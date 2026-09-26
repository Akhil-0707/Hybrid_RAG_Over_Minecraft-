# Progress log

A running record of what has been built, measured and decided, so work can pick up where it
stopped. Newest status first.

## Current status (2026-09-26)

**Done:** full answer eval of the updated retriever (Kaggle notebook `minecraft-rag-eval`,
version 5 — judge self-test passed, 76 graded answers per variant, no errors):

| variant | answer model | retriever | correct | vs. same model, earlier retriever |
|---|---|---|---|---|
| `v2` | `qwen3:4b-instruct` | with intent → section rule | 0.83 | `baseline` 0.80 (+0.03, within noise) |
| `v3` | `qwen3:8b` | with intent → section rule | 0.82 | `v1` 0.82 (no change) |

The only question whose grade changed is the one the fix targeted (Wayfinder trim location:
fail → pass for the 4B model). Details are in the README's results section.

**Next steps (open)**

1. Broad questions are the weakest category (~50% correct): most failures leave out one of
   several required facts (e.g. evokers as the totem source), so a prompt asking the model to
   cover every distinct way/source is the next thing to try and measure.
2. The creeper question (*stop creepers blowing up my house* → they only spawn at light level 0)
   is still not retrieved; LLM query rewriting with the 4B model didn't bridge it.
3. The judge can be strict on wording (it wanted "brushing" for suspicious gravel); calibrating it
   against a few dozen hand-labelled answers would tighten the grades.
4. Future eval runs should vary the Ollama seed per run so the two reps per question are
   independent.

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

### In-game mod (prototype)
- Fabric client mod for Minecraft 26.2 with `/doubt` and `/faq`, backed by `python -m mcrag serve`
  on 127.0.0.1:8765; 68 biomes × 4 pre-generated FAQs.
- Game lag fixed: the model is unloaded after every answer instead of held in video memory for
  30 min; `serve --cpu` keeps it off the GPU entirely.
- Slow answers fixed (20–40 s before any text → first line ~5–7 s): `fast` search mode (no
  cross-encoder; as good on the fact eval), model warm-up while the player types, and answers
  streamed into chat sentence by sentence (`/doubt/stream`).
- Wrong crafting patterns fixed: recipe questions are answered from the game's own recipe data
  (`python -m mcrag recipes-build`), not by the model.
- Wrong spawn answers fixed ("creepers don't spawn in cherry groves"): biome spawn questions are
  answered from the game's biome spawn lists (`python -m mcrag spawns-build`), not by the model.
- Not yet decided: whether the expanded crawl (2,014 pages, Tutorial namespace) stays — broad
  questions regressed on it; the server uses it with at most 2 tutorial chunks per answer, and 10
  new eval questions are drafted to measure it.

### Repository
- Code, eval sets and results published with one commit per file; bulky generated data (crawl,
  index, transcripts) is rebuilt locally with `python -m mcrag crawl` and `python -m mcrag index`.
