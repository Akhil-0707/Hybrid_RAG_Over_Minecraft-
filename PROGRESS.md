# Progress log

A running record of what has been built, measured and decided, so work can pick up where it
stopped. Newest status first.

## Project status: completed (2026-09-27)

The project is complete for now. What it delivers:

- **Hybrid RAG over the Minecraft Wiki**: ~2,000 pages (with tables and tutorials) in 31,846
  passages; dense + BM25 + exact-name search with section rules, and an optional cross-encoder
  reranker. Fact coverage in the top 8 passages: **0.984** (61/62 reference facts, 48 questions)
  in the fast setup the mod uses; 0.935 with the reranker.
- **Answers from a local open-weight model** (`qwen3:4b-instruct` via Ollama, 4 GB GPU), grounded
  and cited; answer eval on Kaggle: 0.83 correct (38 questions, judged by `gemma3:12b`).
- **In-game mod** for Minecraft 26.2, released as
  [v0.1.0](https://github.com/Akhil-0707/Hybrid_RAG_Over_Minecraft-/releases/tag/v0.1.0):
  `/doubt` (streamed answers with wiki sources; exact recipes and spawn lists from the game's
  own data) and `/faq` (biome FAQs). Tested in-game without lag or mod errors.
- **Documentation**: the README covers requirements (GPU tiers, RAM, disk), setup, playing,
  troubleshooting and every command; this file records every measurement and decision.

**If work resumes**, these are the optional next steps, none of them blocking:

1. Generation: the 4B model sometimes ignores the most useful retrieved passage (creepers → light
   level 0; `/give` syntax). Try a prompt change or `qwen3:8b`, measured with the answer eval.
2. Re-run the Kaggle answer eval on the current code and 48 questions (needs harness
   re-approval: retriever, client, generation code and `eval/answers.json` changed since v2/v3).
3. The one remaining retrieval miss: "villagers panic" for the iron golem farm (#9, one short).
4. Broad questions: cover every distinct way/source; calibrate the judge on hand-labelled answers.
5. Mod: a hosted backend for the team (the prototype is localhost-only); the *Dappled Forest* FAQ
   (not in 26.2's game data); a v0.1.1 release only if the mod's Java code changes.

To resume: read this file and the README, run `python -m mcrag serve` (after the README's setup
steps on a new machine), and `python -m mcrag eval --evidence` to confirm the numbers above.

## Current status (2026-09-27)

**Mod released:** [v0.1.0](https://github.com/Akhil-0707/Hybrid_RAG_Over_Minecraft-/releases/tag/v0.1.0)
— `mcrag-helper-0.1.0.jar` for Minecraft 26.2 (SHA-256 `e2c8fff2…aaf57`), the build that was
tested in-game: recipe and spawn questions answered instantly from game data, model answers
streaming into chat with the first line after ~5–6 s and the full answer in 6–12 s on a 4 GB
laptop GPU, no crashes or mod errors in the game log. Installing the mod now needs no JDK or
build (the README points at the release); each player still runs the backend locally
(Python, Ollama, one-time wiki index — see the README's Setup).

**Decided: the expanded crawl stays** (2026-09-27). The 10 new eval questions (commands,
experience, light, hardcore, daylight cycle, archaeology, farm tutorials) were approved and added
(48 questions in total). Fact coverage in the top 8 passages, `eval --evidence`:

| setup | broad | exact | table | new (10) | all |
|---|---|---|---|---|---|
| old index, fast | 0.85 | 1.00 | 1.00 | 0.08 | 0.76 |
| old index, rerank | 0.90 | 1.00 | 1.00 | 0.08 | 0.77 |
| expanded, no cap, fast | 0.40 | 1.00 | 0.90 | 0.69 | 0.73 |
| expanded, no cap, rerank | 0.50 | 0.95 | 1.00 | 0.69 | 0.76 |
| **expanded, 2 tutorial chunks max, fast** (the server's setup) | **0.80** | **1.00** | **1.00** | **0.85** | **0.90** |
| expanded, 2 tutorial chunks max, rerank | 0.75 | 1.00 | 1.00 | 0.69 | 0.85 |

The expansion covers questions the old corpus couldn't (new: 0.08 → 0.85), and the tutorial cap
removes nearly all of the broad-question regression (0.40 → 0.80, one fact below the old index).
No code change: the server already runs this setup. Remaining misses with it: right page but not
the passage holding the fact (`/give` syntax, "villagers panic" in the iron golem farm tutorial,
the trident's 8.5% drop, frosted ice for walking on water), plus two misses the old index also
had (creepers spawn only at light level 0; "most saturation" for mining food).

**Retrieval misses fixed** (2026-09-27). Fact coverage in the server setup 0.903 → **0.984**
(61/62; broad 0.80 → 1.00, new 0.85 → 0.92); rerank mode 0.855 → 0.935; page Hit@5 0.83 → 0.87
(fast), 0.93 unchanged (rerank), MRR@5 up in both. No question lost a fact.

| miss | cause | fix |
|---|---|---|
| walking on water by freezing it → Frost Walker | "walking"/"water" pinned as exact page names; Frost Walker only dense #2 | single-word names are strong only when the question is all names; fast mode keeps a slot for the best dense hit |
| which mob drops the trident → 8.5% | no rule for drop questions | drops → *Mob loot* / *Drops* sections |
| what command gives me a diamond sword → `/give` syntax | Examples chosen over Syntax | command/syntax → *Syntax* section |
| stop creepers blowing up my house → light level 0 | reasoning step (stop → prevent spawning) | stop/prevent/keep away (not "despawn") → *Spawning* section |
| best saturation food for mining | label: the retrieved Golden Carrot usage text says "most health via saturation" | pattern widened (not a retrieval change) |
| how does an iron golem farm work → villagers panic | tutorial cap blocked the named tutorial | a tutorial the question names doesn't count towards the cap — now 5 of its chunks are in the top 8, but the "panic" one is #9 (still a miss) |

Answers checked end to end: Frost Walker and the trident are now answered correctly. For `/give`
and creepers the right passage now reaches the model, but the 4B model built its answer from other
passages (`/give` examples; a trench-and-cactus tutorial for creepers) — a generation limit, not
retrieval.

**Open for the mod**

1. The *Dappled Forest* FAQ keeps its generated mob answer (with spawn weights) — the biome has a
   wiki page but isn't in the 26.2 game data.
2. Code changes to `retriever.py`, `llm.py` and `generate.py`, and the 10 new questions in
   `eval/answers.json`, mean the answer-eval harness needs re-approval before the next Kaggle run.
3. Remaining retrieval miss: "villagers panic" for the iron golem farm (#9, one place short).
   The 4B model sometimes ignores the most useful retrieved passage (creepers, `/give`); a prompt
   change or the 8B model could help, measured with the answer eval.
4. A new mod release (v0.1.1+) is only needed if the mod's Java code changes; backend changes
   reach players with a `git pull` and a restart of `serve`.

## Answer eval status (2026-09-26)

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
- **Expanded crawl** (1,259 → 2,014 pages: gameplay, commands, redstone, dimensions and the
  Tutorial namespace; 31,846 chunks) with at most 2 tutorial chunks per answer and `fast` search:
  fact coverage 0.77 → 0.90 over 48 questions (see Current status).
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
- Biome FAQs: mob questions (84 across the biomes) are answered from the same game spawn lists,
  replacing generated answers that had spawn weights, "Sheeps" and wrong claims (endermites in
  Stony Peaks); irregular plurals are corrected in the rest.
- Recipe questions naming a whole tool family ("a pickaxe of any kind") show the wooden tier first.
- Backend fails fast with a clear message when port 8765 is already in use.

### Release and documentation
- **v0.1.0 released** (2026-09-27) on GitHub with the tested mod jar; published with the GitHub CLI
  signed in as the repository owner.
- README rewritten for new users: requirements (software versions; GPU tiers from CPU-only to
  12 GB+, RAM, disk), step-by-step local setup, installing Fabric and the mod, in-game commands
  with examples, troubleshooting and a command reference. The game jar is found automatically on
  Windows, macOS and Linux.

### Repository
- Code, eval sets and results published with one commit per file; bulky generated data (crawl,
  index, transcripts, and the recipe/spawn data extracted from the game) is rebuilt locally with
  `crawl`, `index`, `recipes-build` and `spawns-build`.
