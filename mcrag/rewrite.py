"""LLM query rewriting: turn a player's question into wiki search queries.

Some questions need a reasoning step before retrieval can work: "how do I stop creepers from
blowing up my house" is answered by *spawning conditions* (creepers only spawn at light level 0),
which no lexical or embedding signal links to the question. A small local model rewrites the
question into a few search queries that name the underlying game mechanic; retrieval then runs
for each, and the rewrites fill two reserved slots after the question's own top results
(HybridRetriever.search_multi).

Rewrites are cached on disk per (model, prompt, question), so repeated evals don't re-query the model.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

from .llm import Ollama, OllamaError

REWRITE_MODEL = "qwen3:4b-instruct"
CACHE = Path(".cache/rewrites.json")

REWRITE_SYSTEM = """\
You write search queries for the Minecraft Wiki. Given a player's question, write up to 3 short \
queries (3-8 words each) that would find the wiki passages answering it.

Think about which game mechanic actually answers the question and name it - for example spawning \
conditions, obtaining or loot sources, crafting, drops, enchantments, or mob behavior - using the \
wiki's own names for items and mobs. Keep one query close to the original wording. Don't answer \
the question and don't add facts.

Reply with only the queries, one per line, with no numbering or other text."""

# Plain lines, not JSON: with Ollama's `format` constraint (a schema or "json") qwen3:4b-instruct
# was measured to spend its whole token budget and return empty content; unconstrained it answers
# in ~25 tokens. Lines are cleaned of any bullets/numbering/quotes the model adds anyway.
_BULLET = re.compile(r"^\s*(?:(?:[-*•]|\d+[.)]|query\s*\d*\s*:)\s*)+", re.I)


def parse_queries(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        q = _BULLET.sub("", line).strip().strip("\"'`").strip()
        # 2-12 words: skips preambles ("Here are the queries:") and one-word chatter.
        if 2 <= len(q.split()) <= 12 and not q.endswith(":"):
            out.append(q)
    return out


class QueryRewriter:
    def __init__(self, model: str = REWRITE_MODEL, client: Ollama | None = None,
                 cache_path: Path | None = CACHE):
        self.model, self.client = model, client or Ollama()
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, list[str]] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def __call__(self, question: str) -> list[str]:
        # The prompt is part of the key: editing it must not silently reuse old rewrites.
        prompt_id = hashlib.sha256(REWRITE_SYSTEM.encode()).hexdigest()[:8]
        key = f"{self.model}::{prompt_id}::{question}"
        if key in self._cache:
            return self._cache[key]
        r = self.client.chat(self.model, [{"role": "system", "content": REWRITE_SYSTEM},
                                          {"role": "user", "content": question}],
                             temperature=0, num_predict=128, num_ctx=2048, think=False)
        queries = parse_queries(r.text)
        if not queries:
            raise OllamaError(f"rewriter returned no usable queries: {r.text[:200]!r}")
        # Drop rewrites identical to the question; the original is always searched anyway.
        queries = [q for q in dict.fromkeys(queries) if q.lower() != question.lower()][:3]
        with self._lock:
            self._cache[key] = queries
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
        return queries
