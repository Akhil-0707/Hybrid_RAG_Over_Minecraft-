# Progress log

A running record of what has been built, measured and decided, so work can pick up where it
stopped. Newest status first.

## Current status (2026-09-26)

**In progress:** full answer eval of the updated retriever on Kaggle
(notebook `minecraft-rag-eval`, version 5):

| variant | answer model | retriever | compared against |
|---|---|---|---|
| `v2` | `qwen3:4b-instruct` | with intent → section rule | `baseline` (same model, earlier retriever) |
| `v3` | `qwen3:8b` | with intent → section rule | `v1` (same model, earlier retriever) |

38 questions × 2 runs each, judged by `gemma3:12b`. The harness change was reviewed and approved
before the run.

**Next steps**

1. When the Kaggle run finishes: download its output, add `eval/results/v2/` and `eval/results/v3/`
   (answers, graded results, change notes), and check the run log (judge self-test passed, 76 graded
   rows per variant, no errors).
2. Compare per question: `baseline` → `v2` and `v1` → `v3`, with 95% intervals over the 38
   questions (the two runs per question are near-copies because Ollama runs with a fixed seed).
3. Update the README's results section with the before/after numbers.
4. Open problems: the creeper question (*stop creepers blowing up my house* → they only spawn at
   light level 0) still isn't retrieved; broad questions are the weakest category (~50% correct).

## Milestones

### Retrieval
- **Hybrid search**: dense (`bge-small-en-v1.5`) + BM25 with a Minecraft-aware tokenizer +
  exact-name matching from page titles and redirects, fused with reciprocal rank fusion.
- **Cross-encoder reranker** (`bge-reranker-base`), blended with the hybrid ranking instead of
  replacing it — page-level Hit@5 1.00, MRR@5 0.84 on 30 queries.
- **Wiki tables and infoboxes** parsed via `action=parse` (trade offers, drop rates, loot chances):
  fact-level Hit@5 on table questions went from 0.20 to 1.00.
- **Intent → section rule** (where / find / obtain → the named page's Obtaining sections): fact
  coverage in the top 8 went from 0.918 to 0.959 and fixed the Wayfinder trim miss.
- **Tried and rejected** (measured, not adopted by default): page expansion and per-section caps
  (no gain); broad intent triggers (hurt other questions); LLM query rewriting (`--rewrite`, 0.94 vs
  0.96 — kept as an opt-in flag).

### Answer generation
- Answers from an open-weight model served by Ollama, citing numbered passages; invalid citation
  numbers are dropped and counted. Runs locally (`qwen3:4b-instruct` fits a 4 GB GPU) or on Kaggle.

### Answer evaluation
- 38 questions (broad, exact-name, table, unanswerable) with reference facts taken from the wiki
  text; metrics: correct, fact recall, grounded, cited, key match.
- Local judge `gemma3:12b` (different model family from the answer models), self-tested on
  reference, empty, "I don't know", wrong-question and invented answers.
- First full run (earlier retriever): `qwen3:4b-instruct` 0.80 correct vs `qwen3:8b` 0.82 —
  statistically tied; 4B is faster (2.8 s vs 3.9 s median).

### Repository
- Code, eval sets and results published with one commit per file; bulky generated data (crawl,
  index, transcripts) is rebuilt locally with `python -m mcrag crawl` and `python -m mcrag index`.
