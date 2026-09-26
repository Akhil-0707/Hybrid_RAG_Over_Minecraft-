"""Hybrid retriever: dense (semantic) + sparse (BM25) + exact entity match, fused with weighted RRF.

- Dense handles broad, paraphrased intent ("efficient ways to get emeralds").
- Sparse BM25 (with bigrams) rewards exact terms like "Swift Sneak III" or "netherite_ingot".
- The entity matcher finds page titles / redirect aliases verbatim in the query and *guarantees*
  that page's best chunk appears in the results, so a precise item name is never drowned out.
- Optionally, a cross-encoder rescores the fused top-N by reading query and chunk together, and
  its ranking is fused back with the hybrid order. This fixes RRF's blind spot: a chunk only one
  retriever ranked #1 no longer loses to chunks both retrievers ranked mediocrely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import index as idx
from .text import STOPWORDS, chunk_header, normalize_name, tokenize

MAX_ENTITY_NGRAM = 7

# Wiki section headings encode what kind of answer a section holds: "where is X" is answered by
# X's Obtaining section. Deliberately narrow - broader triggers ("get", "avoid") were measured to
# displace the right passages on other questions (e.g. "what can I get from piglins"), and "drops"
# pointing at the whole Obtaining section did too; it now points only at loot/drop sections.
INTENTS = [
    (re.compile(r"\b(where|location|locate|find|found|obtain\w*)\b", re.I),
     re.compile(r"^(Obtaining)|Generated loot|Natural generation", re.I)),
    # "what mobs spawn in a cherry grove" -> the biome page's spawn tables.
    (re.compile(r"\b(mobs?|spawn\w*|animals?|monsters?)\b", re.I), re.compile(r"^Mobs\b", re.I)),
    # "which mob drops the trident" -> Trident > Obtaining > Mob loot; "what does a zombie drop"
    # -> Zombie > Drops.
    (re.compile(r"\bdrop(s|ped|ping)?\b", re.I), re.compile(r"Mob loot|^Drops\b", re.I)),
    # "what command gives me a diamond sword" -> Commands/give > Syntax.
    (re.compile(r"\b(commands?|syntax)\b", re.I), re.compile(r"^Syntax\b", re.I)),
    # "how do I stop creepers from blowing up my house" -> Creeper > Spawning (light level 0):
    # the reliable way to stop a mob is to stop it spawning ("stop villagers despawning" is not).
    (re.compile(r"\b(stop\w*|prevent\w*|keep\w* \w+ away)\b(?!.*\bdespawn)", re.I),
     re.compile(r"^Spawning\b", re.I)),
]
# Spawn tables are split across several chunks (monsters in one, animals in the next); for these
# sections the fast mode takes the whole section, in page order, instead of its best chunk.
WHOLE_SECTIONS = re.compile(r"^Mobs\b", re.I)
RERANK_MODEL = "BAAI/bge-reranker-base"


@dataclass
class Hit:
    chunk: dict
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank
    pinned: bool = False


class HybridRetriever:
    def __init__(self, index_dir: Path | str = "index", weights: dict[str, float] | None = None,
                 rrf_k: int = 60, pool: int = 50, weak_entity_factor: float = 0.5,
                 reranker_model: str = RERANK_MODEL, rerank_depth: int = 20,
                 rerank_weight: float = 1.0, intent_boost: float = 0.5,
                 tutorial_cap: int | None = None):
        """intent_boost: weight of the intent -> section vote in rerank mode (0 = off). When a
        question asks where to find a named page's subject, that page's Obtaining sections join
        the rerank pool and get this extra vote.
        tutorial_cap: max Tutorial: chunks in the final top-k (None = no cap). Long tutorial pages
        otherwise fill most of the top-k for broad questions and push out the canonical page."""
        self.chunks, self.emb, self.entities, model_name = idx.load(Path(index_dir))
        self.bm25 = BM25Okapi([tokenize(f"{chunk_header(c)} {c['text']}") for c in self.chunks])
        self.by_title: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.by_title.setdefault(c["title"], []).append(i)
        self.weights = weights or {"dense": 1.0, "sparse": 1.0, "entity": 1.0}
        self.rrf_k, self.pool, self.weak_entity_factor = rrf_k, pool, weak_entity_factor
        self._model_name, self._model = model_name, None
        self.reranker_model, self.rerank_depth, self._reranker = reranker_model, rerank_depth, None
        self.rerank_weight = rerank_weight
        self.intent_boost = intent_boost
        self.tutorial_cap = tutorial_cap

    @staticmethod
    def intent_sections(query: str) -> list[re.Pattern]:
        return [sections for words, sections in INTENTS if words.search(query)]

    def section_representatives(self, title: str, query: str) -> list[int]:
        """Best chunk (by BM25 against the query) from each section of a page."""
        ids = self.by_title[title]
        scores = self.bm25.get_batch_scores(tokenize(query), ids)
        best: dict[str, tuple[float, int]] = {}
        for i, s in zip(ids, scores):
            sec = self.chunks[i]["section"]
            if sec not in best or s > best[sec][0]:
                best[sec] = (float(s), i)
        return [i for _, i in best.values()]

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model, max_length=512)
        return self._reranker

    # --- individual retrievers ------------------------------------------------------------

    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.model.encode(idx.QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.emb @ q
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return sorted(((int(i), float(scores[i])) for i in top), key=lambda x: -x[1])

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def match_entities(self, query: str) -> list[tuple[str, bool]]:
        """Greedy longest-match of known item/mob/enchantment names inside the query.

        Returns (page title, strong). A match is strong if the name is multi-word, or the query is
        a short name lookup: at most 3 content words, all of them part of matched names
        ("mending", "creeper drops"). Strong matches get full entity weight and are pinned; a lone
        generic title like "speed" or "food" inside a question only gets a reduced boost, since it
        is often incidental - "walking on water by freezing it" is not about the Walking and Water
        pages.
        """
        toks = normalize_name(query).split()
        content = {j for j, t in enumerate(toks) if t not in STOPWORDS}
        matches: list[tuple[str, int]] = []  # (title, words in the matched name)
        covered: set[int] = set()
        i = 0
        while i < len(toks):
            for n in range(min(MAX_ENTITY_NGRAM, len(toks) - i), 0, -1):
                titles = self.entities.get(" ".join(toks[i:i + n]))
                if titles:
                    matches += [(t, n) for t in titles if t in self.by_title]
                    covered.update(range(i, i + n))
                    i += n
                    break
            else:
                i += 1
        name_lookup = len(content) <= 3 and content <= covered
        found: dict[str, bool] = {}
        for t, n in matches:
            found[t] = found.get(t, False) or n > 1 or name_lookup
        return list(found.items())

    def entity(self, query: str) -> list[tuple[int, bool]]:
        """(best chunk id, strong) for each exactly-named page; best = top BM25 chunk of that page."""
        q = tokenize(query)
        out = []
        for title, strong in self.match_entities(query):
            ids = self.by_title[title]
            out.append((ids[int(np.argmax(self.bm25.get_batch_scores(q, ids)))], strong))
        return out

    # --- fusion -----------------------------------------------------------------------------

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        """mode: "dense" | "sparse" | "hybrid" (RRF) | "rerank" (RRF top-N -> cross-encoder)
        | "fast" (hybrid + intent -> section rule, no cross-encoder)."""
        if mode == "dense":
            return [Hit(self.chunks[i], s, {"dense": r}) for r, (i, s) in enumerate(self.dense(query, k), 1)]
        if mode == "sparse":
            return [Hit(self.chunks[i], s, {"sparse": r}) for r, (i, s) in enumerate(self.sparse(query, k), 1)]

        entities = self.entity(query)
        lists = {
            "dense": [(i, 1.0) for i, _ in self.dense(query, self.pool)],
            "sparse": [(i, 1.0) for i, _ in self.sparse(query, self.pool)],
            "entity": [(i, 1.0 if strong else self.weak_entity_factor) for i, strong in entities],
        }
        fused: dict[int, Hit] = {}
        for name, results in lists.items():
            w = self.weights.get(name, 0.0)
            for rank, (i, factor) in enumerate(results, 1):
                hit = fused.setdefault(i, Hit(self.chunks[i], 0.0))
                hit.score += factor * w / (self.rrf_k + rank)
                hit.ranks[name] = rank
        ranked = sorted(fused.values(), key=lambda h: -h.score)

        pinned = [fused[i] for i, strong in entities if strong][:k]
        for hit in pinned:
            hit.pinned = True

        if mode == "rerank":
            # The cross-encoder is one more ranked vote fused with the hybrid order, not the sole
            # judge: when it is unsure (low scores everywhere) it would otherwise throw away a
            # dense/sparse #1. On the eval this beats pure reranking (MRR 0.89 vs 0.83).
            candidates = ranked[:self.rerank_depth]
            candidates += [h for h in pinned if not any(c is h for c in candidates)]
            intents = self.intent_sections(query) if self.intent_boost else []
            # Ordered (not a set): pool order feeds the hybrid-rank vote, and set order of
            # strings changes between processes, which made results non-reproducible.
            named = list(dict.fromkeys(t for t, _ in self.match_entities(query))) if intents else []
            if intents:
                # The section that answers the question's intent may rank low on words alone
                # ("location" vs "found in trail ruins"): pull the best-matching chunk of each
                # intent section of every named page into the pool.
                in_pool = {id(h.chunk) for h in candidates}
                for title in named:
                    for i in self.section_representatives(title, query):
                        sec = self.chunks[i]["section"]
                        if any(p.search(sec) for p in intents) and id(self.chunks[i]) not in in_pool:
                            hit = fused.setdefault(i, Hit(self.chunks[i], 0.0))
                            candidates.append(hit)
                            in_pool.add(id(self.chunks[i]))
            ce = self.reranker.predict(
                [(query, f"{chunk_header(h.chunk)}\n{h.chunk['text']}") for h in candidates],
                batch_size=16,
            )
            by_ce = sorted(range(len(candidates)), key=lambda j: -ce[j])
            for ce_rank, j in enumerate(by_ce, 1):
                h = candidates[j]
                h.ranks["rerank"] = ce_rank
                h.score = 1 / (self.rrf_k + j + 1) + self.rerank_weight / (self.rrf_k + ce_rank)
                # Only the named page's sections: "where is the heavy core" must not promote
                # the Obtaining section of Heavy Weighted Pressure Plate.
                if h.chunk["title"] in named and any(p.search(h.chunk["section"]) for p in intents):
                    h.ranks["intent"] = 1
                    h.score += self.intent_boost / (self.rrf_k + 1)
            ranked = sorted(candidates, key=lambda h: -h.score)
        elif mode == "fast":
            # Hybrid without the cross-encoder (~0.2 s per query instead of several seconds on a
            # laptop CPU, for the in-game mod), keeping the intent -> section rule as a guarantee:
            # "where is the wayfinder trim" keeps the named page's Obtaining section in the top-k.
            for hit in self._intent_hits(query, fused)[:3]:
                hit.pinned = True
                pinned.append(hit)
        elif mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}")

        if self.tutorial_cap is not None:
            # A tutorial the question names ("how does an iron golem farm work" -> Tutorial:Iron
            # golem farming) is what was asked for, so its chunks don't count towards the cap.
            asked = {self.chunks[i]["title"] for i, strong in entities if strong}
            capped, n_tut = [], 0
            for h in ranked:
                if h.chunk["title"].startswith("Tutorial:") and h.chunk["title"] not in asked:
                    if n_tut >= self.tutorial_cap:
                        continue
                    n_tut += 1
                capped.append(h)
            ranked = capped
        if mode == "fast":
            # Without the cross-encoder, word overlap can bury the best meaning match: "walking on
            # water by freezing it" fills the top with the Walking page while Frost Walker is only
            # dense #2 (the words "frost" and "walker" never appear). Like an exact-name pin, the
            # best dense hit left after the tutorial cap keeps a slot.
            kept = {id(h) for h in ranked}
            best = next((fused[i] for i, _ in lists["dense"] if id(fused[i]) in kept), None)
            if best is not None and not best.pinned:
                best.pinned = True
                pinned.append(best)
        return self._keep_pinned(ranked[:k], pinned)

    def search_multi(self, question: str, rewrites: list[str], k: int = 8, mode: str = "rerank",
                     extra_slots: int = 2) -> list[Hit]:
        """The question's own top results, plus up to `extra_slots` passages found by rewrites.

        Rewrites only fill reserved slots at the end; they never reorder the question's ranking.
        (RRF fusion was tried first and measured worse: passages that several rewrites agreed on
        outvoted the question's own #1, turning a hit into a miss.) A rewrite's find must be
        on-topic - from a page the question retrieved or names - so a bad rewrite adds nothing.
        """
        base = self.search(question, 2 * k, mode)
        slots = min(extra_slots, len(rewrites))
        if not slots:
            return base[:k]
        keep = base[:k - slots]
        # The question's exact-name pins must survive even if they sat in the last slots.
        keep += [h for h in base[k - slots:k] if h.pinned]
        have = {id(h.chunk) for h in keep}
        on_topic = {h.chunk["title"] for h in base} | {t for t, _ in self.match_entities(question)}
        found: list[Hit] = []
        lists = [self.search(q, k, mode) for q in rewrites]
        for depth in range(k):  # round-robin: each rewrite's best new find, then its next best
            for n, hits in enumerate(lists, 1):
                if len(found) >= k - len(keep) or depth >= len(hits):
                    continue
                h = hits[depth]
                if id(h.chunk) not in have and h.chunk["title"] in on_topic:
                    h.ranks = {**h.ranks, f"rewrite{n}": depth + 1}
                    found.append(h)
                    have.add(id(h.chunk))
        backfill = [h for h in base[k - slots:] if id(h.chunk) not in have]
        return (keep + found + backfill)[:k]

    def _intent_hits(self, query: str, fused: dict[int, Hit]) -> list[Hit]:
        """For where/find/obtain or spawn questions: the best chunk of each intent section of each
        named page (page order; every chunk for WHOLE_SECTIONS), with the intent vote added."""
        intents = self.intent_sections(query) if self.intent_boost else []
        if not intents:
            return []
        out = []
        for title in dict.fromkeys(t for t, _ in self.match_entities(query)):
            best = set(self.section_representatives(title, query))
            for i in self.by_title[title]:
                sec = self.chunks[i]["section"]
                if (i in best or WHOLE_SECTIONS.search(sec)) and any(p.search(sec) for p in intents):
                    hit = fused.setdefault(i, Hit(self.chunks[i], 0.0))
                    hit.ranks["intent"] = 1
                    hit.score += self.intent_boost / (self.rrf_k + 1)
                    out.append(hit)
        return out

    @staticmethod
    def _keep_pinned(top: list[Hit], pinned: list[Hit]) -> list[Hit]:
        """Exact-hit guarantee: every strongly-named entity keeps its best chunk in the top-k."""
        top = list(top)
        for hit in pinned:
            if not any(h is hit for h in top):
                # Evict the lowest-scoring unpinned hit.
                for j in range(len(top) - 1, -1, -1):
                    if not top[j].pinned:
                        top[j] = hit
                        break
        return sorted(top, key=lambda h: -h.score)
