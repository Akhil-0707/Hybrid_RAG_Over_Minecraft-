"""CLI: python -m mcrag {crawl,index,search,ask,eval,answer-eval}"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from .generate import MODEL
from .retriever import RERANK_MODEL

PAGES = Path("data/pages.jsonl")
TABLES = Path("data/tables.jsonl")
INDEX = Path("index")
MODES = ("dense", "sparse", "hybrid", "rerank")


def cmd_crawl(args):
    from .tables import crawl_tables
    from .wiki import crawl, load_pages
    if not args.tables_only:
        crawl(PAGES, recheck_skipped=args.recheck_skipped)
    if not args.no_tables:
        crawl_tables([p["title"] for p in load_pages(PAGES)], TABLES, workers=args.workers)


def cmd_index(args):
    from .index import build
    build(PAGES, INDEX, tables_path=None if args.no_tables else TABLES)


def print_hits(hits, show_text=True):
    for n, h in enumerate(hits, 1):
        via = ", ".join(f"{k}#{v}" for k, v in h.ranks.items())
        pin = "  [exact-name]" if h.pinned else ""
        print(f"{n}. {h.chunk['title']} > {h.chunk['section']}   ({via}){pin}")
        if show_text:
            print(textwrap.indent(textwrap.shorten(h.chunk["text"], 300), "     "))
        print(f"     {h.chunk['url']}")


def cmd_search(args):
    from .retriever import HybridRetriever
    r = HybridRetriever(INDEX, reranker_model=args.reranker)
    queries = [" ".join(args.query)] if args.query else None
    while True:
        q = queries.pop() if queries else input("\nquery> ").strip()
        if not q:
            break
        if args.compare:
            for mode in MODES:
                print(f"\n=== {mode} ===")
                print_hits(r.search(q, args.k, mode), show_text=False)
        else:
            if args.mode in ("hybrid", "rerank"):
                print(f"exact names matched: {[t + ('' if s else ' (weak)') for t, s in r.match_entities(q)] or '-'}")
            print_hits(r.search(q, args.k, args.mode))
        if args.query:
            break


def cmd_ask(args):
    from .generate import Answerer
    from .llm import OllamaError
    from .retriever import HybridRetriever
    from .text import chunk_header

    r = HybridRetriever(INDEX, reranker_model=args.reranker)
    rewriter = None
    if args.rewrite:
        from .rewrite import QueryRewriter
        rewriter = QueryRewriter(args.rewrite_model)
    answerer = Answerer(r, model=args.model, k=args.k, mode=args.mode, think=args.think,
                        rewriter=rewriter)
    questions = [" ".join(args.question)] if args.question else None
    while True:
        q = questions.pop() if questions else input("\nask> ").strip()
        if not q:
            break
        try:
            hits = answerer.retrieve(q)
        except OllamaError as e:
            print(f"[query rewriting failed - Ollama: {e}]")
            if args.question:
                raise SystemExit(1)
            continue
        if answerer.last_rewrites and (args.show_context or args.dry_run):
            print("rewrites:", " | ".join(answerer.last_rewrites))
        if args.show_context or args.dry_run:
            print(f"--- {len(hits)} retrieved chunks ---")
            print_hits(hits, show_text=args.dry_run)
        if args.dry_run:
            chars = sum(len(h.chunk["text"]) for h in hits)
            print(f"--- would call {answerer.model} via Ollama ({chars} chars of context) ---")
        else:
            print()
            ans = None
            try:
                ans = answerer.ask(q, on_text=lambda s: print(s, end="", flush=True), hits=hits)
            except OllamaError as e:
                print(f"\n[Ollama: {e}]")
            if ans is None:
                if args.question:
                    raise SystemExit(1)
                continue
            print("\n\nSources:")
            for n, i in enumerate(ans.cited, 1):
                h = ans.hits[i]
                print(f"  [{n}] {chunk_header(h.chunk)}  {h.chunk['url']}")
            if not ans.cited:
                print("  (no passages cited)")
            note = ""
            if ans.invalid_citations:
                note = f", {ans.invalid_citations} invalid citation marker(s) dropped"
            print(f"  ({ans.model}: {ans.usage['input_tokens']} in / "
                  f"{ans.usage['output_tokens']} out tokens{note})")
        if args.question:
            break


def evidence_recall(r, k: int, modes=("hybrid", "rerank"), verbose: bool = True,
                    rewriter=None) -> dict:
    """Share of reference facts whose evidence is in the top-k chunks (what the LLM sees).

    With a rewriter, a "rewrite" column runs rerank retrieval over the question + its rewrites.
    """
    if rewriter is not None:
        modes = (*modes, "rewrite")
    import re
    cases = {c["id"]: c for c in json.loads(Path("eval/answers.json").read_text(encoding="utf-8"))}
    ev = {k_: v for k_, v in json.loads(Path("eval/evidence.json").read_text(encoding="utf-8")).items()
          if not k_.startswith("_")}
    totals = {m: {} for m in modes}
    if verbose:
        print(f"{'case':10} " + " ".join(f"{m:>8}" for m in modes) + "   (facts covered in top-%d)" % k)
    for cid, pats in ev.items():
        case, row = cases[cid], []
        for m in modes:
            if m == "rewrite":
                q = case["question"]
                hits = [h.chunk for h in r.search_multi(q, rewriter(q), k)]
            else:
                hits = [h.chunk for h in r.search(case["question"], k, m)]
            got = sum(any(h["title"] in case["relevant"] and re.search(p, h["text"], re.I)
                          for h in hits) for p in pats)
            totals[m].setdefault(case["type"], []).append((got, len(pats)))
            row.append(f"{got}/{len(pats)}")
        if verbose and len(set(row)) > 1 or verbose and any(r_.split("/")[0] != r_.split("/")[1] for r_ in row):
            print(f"{cid:10} " + " ".join(f"{x:>8}" for x in row))
    summary = {}
    for m in modes:
        for t, xs in sorted(totals[m].items()) + [("all", [x for v in totals[m].values() for x in v])]:
            facts = sum(g for g, _ in xs) / sum(n for _, n in xs)
            full = sum(g == n for g, n in xs) / len(xs)
            summary[(m, t)] = (facts, full)
    if verbose:
        print(f"\n{'':14}" + "".join(f"{m + ' facts/all-facts':>28}" for m in modes))
        for t in sorted({t for _, t in summary}, key=lambda t: (t == "all", t)):
            print(f"{t:14}" + "".join(f"{summary[(m, t)][0]:>20.2f} / {summary[(m, t)][1]:.2f}"
                                      for m in modes))
    return summary


def cmd_eval(args):
    from .retriever import HybridRetriever
    r = HybridRetriever(INDEX, reranker_model=args.reranker)
    if args.evidence:
        rewriter = None
        if args.rewrite:
            from .rewrite import QueryRewriter
            rewriter = QueryRewriter(args.rewrite_model)
        evidence_recall(r, args.k if args.k != 5 else 8, rewriter=rewriter)
        return
    cases = json.loads(Path(args.file).read_text(encoding="utf-8"))
    modes = MODES

    def relevant(hit, case) -> bool:
        # With an "answer", the chunk must contain the fact itself, not just be on the right page.
        return hit.chunk["title"] in case["relevant"] and (
            "answer" not in case or case["answer"].lower() in hit.chunk["text"].lower())

    types = sorted({c.get("type", "all") for c in cases})
    stats = {(t, m): {"n": 0, "hit": 0, "rr": 0.0} for t in [*types, "all"] for m in modes}
    print(f"{'query':52} " + " ".join(f"{m:>7}" for m in modes))
    for case in cases:
        row = []
        for m in modes:
            hits = r.search(case["query"], args.k, m)
            rank = next((i for i, h in enumerate(hits, 1) if relevant(h, case)), None)
            for key in ((case.get("type", "all"), m), ("all", m)):
                stats[key]["n"] += 1
                stats[key]["hit"] += rank is not None
                stats[key]["rr"] += 1 / rank if rank else 0
            row.append(f"{rank or '-':>7}")
        print(f"{case['query'][:52]:52} " + " ".join(row))

    print(f"\n{'':52} " + " ".join(f"{m:>7}" for m in modes))
    for t in [*types, "all"] if len(types) > 1 else ["all"]:
        for metric, key in ((f"Hit@{args.k}", "hit"), (f"MRR@{args.k}", "rr")):
            label = f"{t} ({stats[(t, modes[0])]['n'] // 1}) {metric}"
            print(f"{label:>52} " + " ".join(
                f"{stats[(t, m)][key] / stats[(t, m)]['n']:>7.2f}" for m in modes))


def cmd_answer_eval(args):
    from .answer_eval import judge_selftest, run, summarize
    if args.judge_selftest:
        raise SystemExit(0 if judge_selftest(args.judge_model) else 1)
    if args.summary:
        summarize(args.variant)
        return
    run(variant=args.variant, model=args.model, think=args.think,
        rewrite_model=args.rewrite_model if args.rewrite else None,
        judge_model=args.judge_model, phase=args.phase,
        reps=args.reps, concurrency=args.concurrency, timeout_s=args.timeout_s,
        limit=args.limit, ids=args.ids, approve_harness=args.approve_harness, index_dir=INDEX)


def cmd_serve(args):
    from .server import serve
    serve(host=args.host, port=args.port, model=args.model, cpu_only=args.cpu)


def cmd_faq_build(args):
    from .faq import build
    build(model=args.model, only=args.only)


def cmd_recipes_build(args):
    from .recipes import build, default_jar
    build(Path(args.jar) if args.jar else default_jar(args.version))


def main():
    # Wiki text has characters (×, zero-width joiners) that Windows' cp1252 console encoding
    # can't represent; without this, redirected output crashes mid-print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="mcrag", description="Hybrid RAG retrieval over the Minecraft Wiki")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="download wiki pages + tables to data/ (resumable)")
    c.add_argument("--tables-only", action="store_true", help="only fetch tables for cached pages")
    c.add_argument("--no-tables", action="store_true", help="skip the action=parse table step")
    c.add_argument("--workers", type=int, default=2, help="parallel requests for the table step")
    c.add_argument("--recheck-skipped", action="store_true",
                   help="re-fetch previously skipped titles (after changing exclusion rules)")
    c.set_defaults(fn=cmd_crawl)
    i = sub.add_parser("index", help="chunk pages + tables and build dense + sparse + entity index")
    i.add_argument("--no-tables", action="store_true", help="index prose only (for comparison)")
    i.set_defaults(fn=cmd_index)
    s = sub.add_parser("search", help="search (interactive if no query given)")
    s.add_argument("query", nargs="*")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=MODES, default="rerank",
                   help="rerank = hybrid RRF followed by the cross-encoder (default)")
    s.add_argument("--compare", action="store_true", help="show every mode side by side")
    s.add_argument("--reranker", default=RERANK_MODEL, help="cross-encoder model for rerank mode")
    s.set_defaults(fn=cmd_search)
    a = sub.add_parser("ask", help="answer a question with an LLM over retrieved wiki passages")
    a.add_argument("question", nargs="*", help="interactive if omitted")
    a.add_argument("-k", type=int, default=8, help="passages sent to the model")
    a.add_argument("--mode", choices=MODES, default="rerank", help="retrieval mode")
    a.add_argument("--model", default=MODEL, help="Ollama model (server: OLLAMA_HOST)")
    a.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    a.add_argument("--rewrite", action="store_true", help="LLM query rewriting before retrieval")
    a.add_argument("--rewrite-model", default="qwen3:4b-instruct")
    a.add_argument("--reranker", default=RERANK_MODEL, help="cross-encoder model for rerank mode")
    a.add_argument("--show-context", action="store_true", help="print retrieved passages first")
    a.add_argument("--dry-run", action="store_true", help="retrieve and show context, no LLM call")
    a.set_defaults(fn=cmd_ask)
    e = sub.add_parser("eval", help="compare retrievers on labelled queries")
    e.add_argument("--file", default="eval/queries.json")
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--reranker", default=RERANK_MODEL, help="cross-encoder model for rerank mode")
    e.add_argument("--evidence", action="store_true",
                   help="fact-level: are the reference facts' passages in the top-k (default k=8)?")
    e.add_argument("--rewrite", action="store_true",
                   help="with --evidence: add a column using LLM query rewriting (needs Ollama)")
    e.add_argument("--rewrite-model", default="qwen3:4b-instruct")
    e.set_defaults(fn=cmd_eval)
    ae = sub.add_parser("answer-eval", help="answer and grade eval/answers.json (needs Ollama)")
    ae.add_argument("--variant", default="baseline", help="output dir: baseline, v1, v2, ...")
    ae.add_argument("--model", default=MODEL, help="answer model served by Ollama")
    ae.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    ae.add_argument("--rewrite", action="store_true", help="LLM query rewriting before retrieval")
    ae.add_argument("--rewrite-model", default="qwen3:4b-instruct")
    ae.add_argument("--judge-model", default="gemma3:12b", help="judge model served by Ollama")
    ae.add_argument("--phase", choices=["both", "answer", "grade"], default="both",
                    help="answer all cases, grade all answers, or both in sequence")
    ae.add_argument("--reps", type=int, default=1)
    ae.add_argument("--concurrency", type=int, default=1,
                    help="parallel requests (keep 1 for a single local GPU)")
    ae.add_argument("--timeout-s", type=float, default=600, help="hard per-case ceiling (0 = none)")
    ae.add_argument("--limit", type=int, help="only the first N cases (pilot)")
    ae.add_argument("--ids", nargs="+", help="only these case ids")
    ae.add_argument("--summary", action="store_true", help="print the summary of existing results")
    ae.add_argument("--judge-selftest", action="store_true",
                    help="check the judge passes reference answers and fails bad ones (~15 judge calls)")
    ae.add_argument("--approve-harness", action="store_true",
                    help="record the current harness as reviewed (a human decision), then exit")
    ae.set_defaults(fn=cmd_answer_eval)
    sv = sub.add_parser("serve", help="local HTTP backend for the in-game mod (/doubt, /faq)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--model", default="qwen3:4b-instruct", help="Ollama model for answers")
    sv.add_argument("--cpu", action="store_true",
                    help="run the model on the CPU only (slower, no GPU use - for heavy games)")
    sv.set_defaults(fn=cmd_serve)
    fb = sub.add_parser("faq-build", help="pre-generate biome FAQs into assets/biome_faq.json")
    fb.add_argument("--model", default="qwen3:4b-instruct")
    fb.add_argument("--only", nargs="+", help="only these biome page titles")
    fb.set_defaults(fn=cmd_faq_build)
    rb = sub.add_parser("recipes-build",
                        help="extract exact recipes from your Minecraft jar into data/recipes.json")
    rb.add_argument("--jar", help="path to the game jar (default: .minecraft/versions/<version>)")
    rb.add_argument("--version", default="26.2")
    rb.set_defaults(fn=cmd_recipes_build)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
