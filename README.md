# Minecraft Hybrid RAG

Hybrid retrieval over the [Minecraft Wiki](https://minecraft.wiki): dense semantic search for broad
questions, sparse BM25 for exact terms, an exact-name matcher that guarantees precise item /
mob / enchantment / armor-trim names are never lost, and a cross-encoder reranker on top. An
open-weight LLM served by Ollama then answers from the retrieved passages, citing each claim back
to its wiki section. Everything runs locally or on Kaggle's free GPU — no paid API.

It also ships as an **in-game assistant**: a Fabric mod for Minecraft 26.2 adds `/doubt <question>`
(answers stream into chat with clickable wiki sources) and `/faq` (FAQs for the biome you're
standing in), backed by a small server on your own computer. Crafting recipes and biome spawn lists
are answered straight from the game's own data files, so they are always exact.

**Contents:** [Requirements](#requirements) · [Setup](#setup-step-by-step) ·
[Playing with the mod](#playing-with-the-mod) · [Troubleshooting](#troubleshooting) ·
[Command reference](#command-reference) · [How it works](#pipeline) ·
[Evaluation](#evaluation)

| Query | Who wins | Why |
|---|---|---|
| `efficient ways to get emeralds` | **dense** | No page says those words; embeddings map it to Trading, Raids, Emerald Ore |
| `Swift Sneak III`, `wind_charge` | **sparse** | Rare exact tokens + bigrams (`swift__sneak`) and snake_case IDs score high in BM25 |
| `Wayfinder trim smithing template location` | **entity** | "wayfinder trim" is a wiki redirect → *Wayfinder Armor Trim* is pinned into the top-k |

## Requirements

**Software**

| What | Version | Needed for |
|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.10 or newer (tested on 3.13) | everything |
| [Git](https://git-scm.com/downloads) | any | downloading the project |
| [Ollama](https://ollama.com/download) | recent (tested on 0.34) | running the answer model locally |
| Minecraft Java Edition | **26.2**, launched once | the in-game mod, recipe and spawn data |
| [Fabric Loader](https://fabricmc.net/use/installer/) + [Fabric API](https://modrinth.com/mod/fabric-api) | Loader 0.19+, Fabric API 0.161.0+26.2 | the in-game mod |
| JDK ([Eclipse Temurin](https://adoptium.net/)) | **25** or newer | only for building the mod yourself — the [release](https://github.com/Akhil-0707/Hybrid_RAG_Over_Minecraft-/releases/latest) has a ready-made jar |

Tested on Windows 11; macOS and Linux should work the same way (nothing is Windows-specific). The
commands below are shown for Windows, with the macOS/Linux form where it differs.

**Hardware**

The model has to share the computer with Minecraft, so what matters is how much video memory
(VRAM) is left while the game runs. Minecraft itself typically takes 1–2 GB of VRAM (more with
shaders or a high render distance).

| Setup | Answer model | What to expect |
|---|---|---|
| **No GPU / weak GPU** | `qwen3:4b-instruct` on the CPU (`serve --cpu`) | Works on any machine with 16 GB RAM. About 30–35 s per model answer; recipe and spawn answers are still instant. The game's GPU is never touched. |
| **4 GB NVIDIA GPU** (minimum GPU; tested on an RTX 3050 Laptop) | `qwen3:4b-instruct` (default) | Model uses ~2.3 GB VRAM only while answering and is unloaded right after, so the game doesn't lag. First line of an answer in ~5–7 s, the rest streams in. |
| **6–8 GB NVIDIA GPU** (recommended) | `qwen3:4b-instruct` with `serve --keep-alive 5m` | The model stays loaded between questions, so follow-ups start in 1–2 s, with room left for the game. |
| **12 GB+ NVIDIA GPU** | `qwen3:8b` (`serve --model qwen3:8b`) | The 8B model needs about 6 GB of VRAM (5.2 GB download; not measured on the test laptop). In the eval it was as accurate as the 4B (0.82 vs 0.83 correct) but stuck closer to the sources (grounded 0.87 vs 0.76). |

Ollama also runs on Apple Silicon Macs (using unified memory) and on some AMD GPUs; see the
[Ollama documentation](https://github.com/ollama/ollama) for supported hardware.

Other resources:

- **RAM**: 8 GB minimum, 16 GB recommended (the backend uses ~1.1 GB, Minecraft 2–4 GB, and CPU mode
  keeps the model in RAM too).
- **Disk**: about 5 GB — Python packages including PyTorch (~2 GB), the answer model (2.5 GB, or
  5.2 GB for `qwen3:8b`), search models (130 MB, plus 1.1 GB if you use the reranker), and the
  wiki index (~100 MB).
- **CPU**: any modern 4-core CPU; search uses at most 4 threads so the game keeps the rest.
- **Internet**: only for the one-time setup (downloads and the wiki crawl). Playing works offline.

## Setup (step by step)

Do steps 1–6 once. After that, only [Playing with the mod](#playing-with-the-mod) is needed each
time you play.

**1. Download the project**

```bash
git clone https://github.com/Akhil-0707/Hybrid_RAG_Over_Minecraft-.git
```

```bash
cd Hybrid_RAG_Over_Minecraft-
```

**2. Create a Python environment and install the dependencies**

```bash
python -m venv .venv
```

Activate it — Windows: `.venv\Scripts\activate` · macOS/Linux: `source .venv/bin/activate` — then:

```bash
pip install -r requirements.txt
```

Activate the environment again in every new terminal before running `python -m mcrag ...`.

**3. Install Ollama and download the answer model**

Install Ollama from [ollama.com/download](https://ollama.com/download). On Windows and macOS it
starts in the background by itself; on Linux run `ollama serve`. Then download the model (2.5 GB):

```bash
ollama pull qwen3:4b-instruct
```

With a 12 GB+ GPU you can also pull `qwen3:8b` (see [Requirements](#requirements)).

**4. Build the wiki index** (one-time, about 1–2 hours)

The wiki text is not in the repository — it is downloaded from the
[Minecraft Wiki](https://minecraft.wiki) API and indexed on your machine. Both commands can be
stopped and re-run; they continue where they left off.

```bash
python -m mcrag crawl
```

```bash
python -m mcrag index
```

`crawl` fetches ~2,000 pages with their tables into `data/`; `index` splits them into ~32,000
passages and embeds them into `index/` (the first run also downloads the 130 MB search model).

**5. Extract recipes and spawn lists from your game** (needs Minecraft 26.2)

Launch Minecraft 26.2 once from the official launcher so the game file
(`.minecraft/versions/26.2/26.2.jar`) exists, then:

```bash
python -m mcrag recipes-build
```

```bash
python -m mcrag spawns-build
```

These write `data/recipes.json` (1,536 recipes) and `data/spawns.json` (66 biomes). If your game
is installed somewhere else, add `--jar <path to 26.2.jar>`.

**6. Check that it works**

```bash
python -m mcrag search "efficient ways to get emeralds"
```

```bash
python -m mcrag ask --model qwen3:4b-instruct "how many emeralds does a novice librarian want for a bookshelf"
```

`search` should list wiki passages; `ask` should print an answer with numbered sources. The biome
FAQs (`assets/biome_faq.json`) are already included, so there is nothing to generate for `/faq`.

## Playing with the mod

**1. Install Fabric for Minecraft 26.2** (once)

1. Download and run the [Fabric installer](https://fabricmc.net/use/installer/), choose
   Minecraft **26.2** and click *Install*. This adds a `fabric-loader-26.2` installation to the
   Minecraft launcher.
2. Download **Fabric API** for 26.2 (0.161.0+26.2 or newer) from
   [Modrinth](https://modrinth.com/mod/fabric-api/versions) and put the jar in your `mods` folder:
   - Windows: `%APPDATA%\.minecraft\mods`
   - macOS: `~/Library/Application Support/minecraft/mods`
   - Linux: `~/.minecraft/mods`

   (Create the folder if it doesn't exist. Use a separate game directory if your `mods` folder
   already holds mods for other Minecraft versions.)

**2. Install the mod** (once, and again after a new mod version)

Download `mcrag-helper-0.1.0.jar` from the
[latest release](https://github.com/Akhil-0707/Hybrid_RAG_Over_Minecraft-/releases/latest) and put
it in the same `mods` folder as Fabric API. That's all — no Java install or build needed.

*Or build it yourself* (e.g. after changing the mod's code). This needs JDK 25+ (`java -version`
should say 25 or higher); the first build downloads Gradle, Minecraft and Fabric and takes a few
minutes.

```bash
cd minecraft-mod
```

Windows (PowerShell or cmd):

```bash
.\gradlew.bat build
```

macOS/Linux: `./gradlew build`. The mod is written to `minecraft-mod/build/libs/mcrag-helper-0.1.0.jar`
— copy it into the same `mods` folder as Fabric API, then go back to the project folder (`cd ..`).

**3. Start the backend** (every time you play)

Make sure Ollama is running, activate the Python environment, and from the project folder run:

```bash
python -m mcrag serve
```

Wait until it prints `Uvicorn running on http://127.0.0.1:8765` (about 25 s) and keep this window
open while you play; stop it with Ctrl+C when you're done. Useful options:

| Option | When to use it |
|---|---|
| `--cpu` | No NVIDIA GPU, or the game still lags: the model runs on the CPU only (slower answers). |
| `--keep-alive 5m` | 6 GB+ VRAM: keep the model loaded between questions for faster follow-ups. |
| `--model qwen3:8b` | 12 GB+ VRAM and the 8B model pulled. |
| `--rerank` | Use the cross-encoder reranker (adds ~7 s per question on a laptop CPU). |
| `--port 8766` | Port 8765 is taken (then also start the game with `-Dmcrag.backend=http://127.0.0.1:8766` in the launcher's JVM arguments). |

To check it's up, open [http://127.0.0.1:8765/health](http://127.0.0.1:8765/health) in a browser.

**4. Play**

Start Minecraft with the **fabric-loader-26.2** installation and open a world (single-player or
any server — the commands run on your own computer, so the server doesn't need the mod). Then type
in chat:

| Command | What it does |
|---|---|
| `/doubt <question>` | Ask anything about the game. The answer appears in chat line by line, followed by numbered wiki sources — click one to open the page. The grey footer shows what answered (the model, *game recipe data* or *game spawn data*) and how long it took. |
| `/faq` | Four quick FAQs about the biome you are standing in. |
| `/faq <biome>` | FAQs for any biome, e.g. `/faq cherry grove`. |

Examples:

- `/doubt how do I craft a piston` — exact crafting grid from the game, instantly
- `/doubt how do I smelt iron ore` — furnace and blast furnace recipes
- `/doubt what mobs spawn here` — this biome's spawn list (hostile, animals, water, ambient)
- `/doubt do wolves spawn here` — yes/no from the game's spawn list
- `/doubt where do I find the wayfinder armor trim` — answered by the model from the wiki
- `/doubt how many sticks does a fletcher want for one emerald` — answered from a wiki trade table

Tips:

- Words like *here*, *nearby* or *this biome* make the answer use the biome you're in; other
  questions are answered generally.
- The model starts loading as soon as you type a space after `/doubt`, so the answer comes faster
  if you type the question at a normal pace rather than pasting it.
- The very first question after starting the backend is a few seconds slower.
- Recipe and spawn answers use Java Edition 26.2's data; wiki answers say when Java and Bedrock
  differ.
- Everything stays on your computer: questions go only to the local backend, and answers are shown
  only in your own chat.

## Troubleshooting

| Problem | Fix |
|---|---|
| Chat says *Couldn't reach the backend at http://127.0.0.1:8765* | Start `python -m mcrag serve` and wait for `Uvicorn running…` before asking. |
| Chat says *Ollama is not available* | Start Ollama (open the app, or `ollama serve`) and check `ollama list` shows `qwen3:4b-instruct`; if not, `ollama pull qwen3:4b-instruct`. |
| `serve` says *Port 8765 … is already in use* | An older backend is still running — press Ctrl+C in its window or close it, then start again. |
| `/doubt` is an unknown command | The mod isn't loaded: start the **fabric-loader-26.2** installation, and check that both Fabric API and `mcrag-helper-0.1.0.jar` are in `mods`. |
| The game lags while an answer is written | Use `serve --cpu`, or lower the render distance. Don't use `--keep-alive` on a 4 GB GPU. |
| Recipe or spawn questions get a long model answer instead of the exact data | Run `recipes-build` / `spawns-build` (step 5) and restart `serve`; check `/health` shows `items_with_recipes` and `biomes_with_spawns` above 0. |
| `recipes-build` says *Minecraft jar not found* | Launch Minecraft 26.2 once, or pass `--jar` with the path to `26.2.jar`. |
| `serve` or `ask` fails with a missing `index/` file | Run step 4 (`crawl`, then `index`). |
| `cd minecraft-mod && .\gradlew.bat build` fails in PowerShell | Windows PowerShell 5 doesn't support `&&`; run the two commands separately. |
| Gradle build fails with *Unsupported class file major version* or a Java version error | Install JDK 25+ and point `JAVA_HOME` at it. |

## Command reference

All commands run from the project folder as `python -m mcrag <command>`:

| Command | What it does |
|---|---|
| `crawl` | Download wiki pages and tables into `data/` (resumable; `--tables-only`, `--no-tables`, `--recheck-skipped`). |
| `index` | Build the search index in `index/` (`--no-tables` for a prose-only index). |
| `recipes-build` / `spawns-build` | Extract recipes / biome spawn lists from the game jar (`--jar`, `--version`). |
| `search "<query>"` | Show the top passages (`--mode dense/sparse/hybrid/rerank/fast`, `--compare`, `-k`). |
| `ask "<question>"` | Answer from the command line (`--model`, `--show-context`, `--dry-run`, `--rewrite`, `--think`). |
| `serve` | Start the backend for the mod (`--cpu`, `--keep-alive`, `--model`, `--rerank`, `--port`). |
| `faq-build` | Regenerate the biome FAQs in `assets/biome_faq.json` (`--model`, `--only`). |
| `eval` | Retrieval evaluation (`--evidence` for fact-level coverage). |
| `answer-eval` | Answer + judge evaluation (see [Answer evaluation](#answer-evaluation)). |

## Pipeline

```
minecraft.wiki API ──crawl──> data/pages.jsonl  ──index──> index/
  TextExtracts +                prose, aliases              chunks.jsonl     prose + table + infobox chunks
  categories + redirects                                    embeddings.npy   BAAI/bge-small-en-v1.5
  action=parse (HTML)  ───────> data/tables.jsonl           entities.json    title/redirect -> page
                                tables as "Header: value" rows, infoboxes
query ─┬─ dense  (cosine, top 50) ──┐
       ├─ sparse (BM25 + bigrams)  ─┼─ weighted RRF ─> top 20 ─> cross-encoder ─> RRF(hybrid rank, ─> pin exact- ─> top-k
       └─ entity (longest n-gram   ─┘                           (bge-reranker-base)  rerank rank)       name hits
                  name match)                                                      │
                                              top-8 passages, numbered ─> LLM (Ollama) ─> answer with [n] citations
```

- **Chunking** (`mcrag/text.py`): splits on wiki `== headings ==`, keeps the `Title > Section` path
  in every chunk, and drops noisy sections (History, Gallery, Data values…).
- **Tables** (`mcrag/tables.py`): TextExtracts drops every table, so trade offers, drop rates, loot
  chances and infobox stats (health, max level, enchantment weight) were unsearchable. The crawl
  also fetches `action=parse` HTML and turns each table row into a self-contained line, expanding
  rowspans/colspans and multi-level headers so no row loses its context:
  `Level: Novice; Probability JE: 67%; Villager wants: 9 × Emerald; Player receives: Bookshelf; …`.
  Rows are grouped into ~180-word chunks (never split mid-row); the infobox is one chunk. Navboxes,
  ID tables, calculators and tables under skipped sections are dropped; hidden JSON is stripped.
  Only extracted rows are cached — raw HTML (up to ~1 MB/page) is discarded.
- **Sparse tokenizer**: keeps roman numerals and short tokens, indexes `netherite_ingot` as both the
  ID and its parts, folds plurals, and adds bigrams so exact multi-word names rank first.
- **Entity matcher**: page titles and multi-word redirects (`"sentry trim"`, `"Sentry Armour Trim"`)
  are matched greedily in the query. Multi-word matches, or any match in a short name-like query,
  are *strong*: full entity weight and *pinned* (their best chunk is guaranteed a slot). A lone
  generic title inside a longer question ("food", "speed") is *weak*: half weight, never pinned.
  Single-word redirects (`price` → Trading) are ignored entirely.
- **Fusion** (`mcrag/retriever.py`): `score = Σ w_r / (60 + rank_r)`; weights are configurable.
- **Reranker** (`--mode rerank`, the default): `BAAI/bge-reranker-base` reads query and chunk
  together for the fused top 20. Its ranking is then fused with the hybrid ranking
  (`1/(60 + hybrid_rank) + w/(60 + rerank_rank)`) instead of replacing it: when the cross-encoder
  is unsure it would otherwise discard a dense #1 hit (pure reranking scored MRR 0.83 vs 0.89 fused).
  Exact-name pins still apply after reranking. Costs ~2–4 s/query on CPU; for speed use
  `--reranker cross-encoder/ms-marco-MiniLM-L-6-v2`.

- **Intent → section** (`INTENTS` in `mcrag/retriever.py`, rerank mode): a question asking *where*
  to find / how to obtain something named by a page ("Wayfinder trim smithing template location")
  pulls that page's *Obtaining* / *Generated loot* sections into the rerank pool and gives them one
  extra vote. The trigger words are deliberately narrow and only the *named* page is boosted —
  broader rules (e.g. "get", "avoid") were measured to push the right passages out elsewhere.
- **Generation** (`mcrag/generate.py`, `mcrag/llm.py`): the top 8 passages are numbered
  `[1]`–`[8]` in the prompt to a model served by Ollama (`qwen3:8b` by default; `OLLAMA_HOST` picks
  the server). The system prompt keeps answers grounded in the excerpts (say so when they don't
  cover it; separate Java vs Bedrock) and asks the model to cite them; markers pointing at no
  passage are dropped and counted, and the rest are renumbered in order of first use. `num_ctx` is
  set explicitly to 8192 — Ollama's short default would silently cut passages off. Thinking mode
  is off unless `--think`.
- **Query rewriting** (`mcrag/rewrite.py`, opt-in `--rewrite`): a small model (`qwen3:4b-instruct`)
  turns the question into up to 3 wiki search queries. The question's own top 6 results are kept
  as-is and the rewrites may only fill the last 2 slots with on-topic passages the question missed,
  so a bad rewrite can't push a good result out (plain rank fusion was measured to do exactly that).
  Rewrites are cached in `.cache/rewrites.json`.

## Command-line usage

After [setup](#setup-step-by-step):

```bash
python -m mcrag search "efficient ways to get emeralds"
```

```bash
python -m mcrag search --compare "how do I avoid dying and keep my items when I take lethal damage"
```

```bash
python -m mcrag ask "how many emeralds does a novice librarian want for a bookshelf"
```

```bash
python -m mcrag eval
```

`ask` needs a running Ollama server with the model pulled (`ollama pull qwen3:8b`; on a 4 GB GPU
use `--model qwen3:4b-instruct`). With no question it starts an interactive prompt.
`--show-context` prints the retrieved passages first, `--dry-run` shows what would be sent without
calling the model, `--rewrite` turns on query rewriting, and `-k`, `--mode`, `--model` tune the
pipeline. Output format (illustrative):

```
<answer text streamed here, with markers after cited claims>[1][2] ...

Sources:
  [1] <Page> > <Section>  https://minecraft.wiki/w/<Page>
  [2] ...
  (<input> in / <output> out tokens)
```

`crawl` is resumable (it skips pages already in `data/pages.jsonl` / `data/tables.jsonl`). The
table step fetches full rendered HTML and takes ~45 min for 1,259 pages at the default 2 parallel
requests; `--tables-only` backfills tables for cached pages, `--no-tables` skips them, and
`index --no-tables` builds a prose-only index for comparison. Edit `DEFAULT_CATEGORIES` /
`SEED_TITLES` in `mcrag/wiki.py` to change coverage; joke/spin-off pages are filtered out.

From Python:

```python
from mcrag.retriever import HybridRetriever
r = HybridRetriever("index", weights={"dense": 1.0, "sparse": 1.2, "entity": 1.0})
for hit in r.search("Silence Armor Trim", k=5, mode="rerank"):  # or "hybrid" to skip the reranker
    print(hit.chunk["title"], hit.chunk["section"], hit.ranks, hit.pinned)
```

## Evaluation

`eval/queries.json` has 30 labelled queries: 10 broad, 10 exact-name, and 10 *table* questions
whose answer only lives in a table or infobox. Table queries carry an `answer` string, so a hit
only counts if the retrieved chunk contains the fact itself (e.g. `94.44%`), not just the right
page. `python -m mcrag eval` prints the rank of the first relevant chunk per mode, then Hit@5 /
MRR@5 per query type. On 1,259 pages / 16,608 chunks (8,403 prose, 6,987 table, 1,218 infobox):

| Hit@5 / MRR@5 | dense       | sparse      | hybrid      | rerank          |
|---------------|-------------|-------------|-------------|-----------------|
| broad (10)    | 0.90 / 0.72 | 0.60 / 0.47 | 1.00 / 0.71 | **1.00 / 0.90** |
| exact (10)    | 1.00 / 1.00 | 0.90 / 0.85 | 1.00 / 1.00 | **1.00 / 1.00** |
| table (10)    | 0.90 / 0.53 | 0.60 / 0.25 | 0.90 / 0.49 | **1.00 / 0.62** |
| all (30)      | 0.93 / 0.75 | 0.70 / 0.53 | 0.97 / 0.73 | **1.00 / 0.84** |

Before table parsing (prose-only index, `index --no-tables`) every mode scored Hit@5 0.20 on the
table queries — facts like trade prices, Looting drop odds and enchantment weight simply weren't
in the corpus. With tables, rerank finds all ten.

Observations:
- The reranker matters most on broad and table queries (MRR 0.71 → 0.90 and 0.49 → 0.62 over
  hybrid); it fixes hybrid's *"avoid dying and keep my items"* miss (Totem of Undying → #1).
- Table chunks cost sparse some precision: its broad Hit@5 fell from 0.70 to 0.60, since
  header-heavy rows ("Villager wants", "Probability JE") match common query words. Hybrid and
  rerank absorb this.
- Table answers often land at #2–#3 rather than #1: the prose chunk for the same page usually
  outranks the table row. Fine for RAG context, but row-level answers could be pushed higher.
- With 30 queries these numbers are indicative, not conclusive — grow `eval/queries.json`
  before tuning further.

## Answer evaluation

`python -m mcrag answer-eval` runs the real `ask` pipeline (retrieval -> answer model) on the 38
questions in `eval/answers.json` and grades every answer. Reference facts were taken from the indexed
wiki text; the 8 *unanswerable* questions (mods, OptiFine, real-world prices, YouTubers…) are not in
the corpus, so the right answer is to say so.

| metric | kind | graded by |
|---|---|---|
| `correct` (headline) | pass/fail | judge (`gemma3:12b` via Ollama): states every reference fact, contradicts none; unanswerable -> declines and invents nothing |
| `fact_recall` | 0–1 | same judge call: share of reference facts stated |
| `grounded` | pass/fail | separate judge call: every game claim is supported by the retrieved passages (catches answering from memory) |
| `cited` | pass/fail | programmatic: a cited passage is from a relevant page (answerable cases only) |
| `key_match` | pass/fail | programmatic, table cases only: the key value (`94.44`, `8–32`, `5 emeralds`…) appears as a whole number — a cross-check on a lenient local judge |

The correctness judge never sees the passages and the groundedness judge never sees the reference
facts. Judges use schema-constrained JSON output and treat the answer as untrusted data. The judge
is a different model family from the answer models (Gemma vs Qwen) to avoid self-preference.
Runs have two phases — `--phase answer` for every case, then `--phase grade` — so an answer model
and a local judge never have to share one GPU; `--phase both` does them in sequence. Latency and
tokens (answer model + judge) are recorded per case.

Output goes to `eval/results/<variant>/` (`answers.jsonl`, `results.jsonl`, `traces/`,
`errors.jsonl`). The runner writes rows as they finish and resumes at the (case, rep) key, retries
an unreachable or failing Ollama server with jittered backoff (counted per row), enforces a hard
per-case wall-clock ceiling, and never scores plumbing as a model failure: timeouts, server/judge
errors and answers served by a different model go to `errors.jsonl`; answers cut off at the token
limit are marked `truncated` and left out of the means.

It also refuses to run until a human has reviewed the harness (the runner, `generate.py`,
`retriever.py` and `answers.json`) and recorded that with `--approve-harness`; any later edit to
those files requires re-approval, so scores are never silently compared across different harnesses.

```bash
python -m mcrag answer-eval --judge-selftest
```

```bash
python -m mcrag answer-eval --model qwen3:8b --reps 2
```

### Results (Kaggle T4, 2026-09-25)

38 questions × 2 runs per model, judged by `gemma3:12b` (judge self-test 15/15; 0 errors, 0
truncated). Per-question means; the delta is paired over the 38 questions with a 95% CI:

| metric | `qwen3:4b-instruct` | `qwen3:8b` | 8B − 4B |
|---|---|---|---|
| correct | 0.80 | 0.82 | +0.01 [−0.05, +0.07] |
| fact recall | 0.84 | 0.84 | +0.00 [−0.04, +0.04] |
| grounded | 0.76 | 0.87 | +0.11 [−0.03, +0.24] |
| cited | 0.93 | 0.95 | +0.02 [−0.08, +0.12] |
| key match (table) | 1.00 | 1.00 | — |
| median latency | 2.8 s | 3.9 s | |

By type (both models): table and unanswerable questions 100% correct, exact names 80%, broad
questions ~50%. No difference between the models is statistically significant; the 8B stays closer
to the passages. Remaining failures are about half *omissions* (a required fact left out — e.g. evokers
as the totem source), a few *wrong answers* (Suspicious Stew over golden carrot), some
*hallucinations* caught by `grounded` (the 8B invented an elytra crafting recipe), and two
*retrieval misses* (creeper spawn light level and Wayfinder trail ruins were never retrieved).

Caveat: Ollama runs with a fixed seed, so the 2 reps are near-copies (grade differs on only 9 of
76 rep pairs) — the effective sample is 38 questions, and the runner's per-type ±CIs, which treat
rows as independent, are too narrow. Use the paired per-question numbers above, or pass a varying
seed for future reps.

### Fact-level retrieval check

`eval/evidence.json` has one regex per reference fact in `eval/answers.json`;
`python -m mcrag eval --evidence` reports how many facts have their evidence in the top 8
passages — exactly what the answer model sees. With the intent → section rule this went from
0.918 to **0.959** of facts (questions with every fact covered 0.90 → 0.93), fixing the Wayfinder
miss, with no change to page-level Hit@5/MRR. The creeper miss (*stop creepers blowing up my house*
→ they only spawn at light level 0) remains: the link is a reasoning step no lexical or
cross-encoder signal captures; LLM query rewriting is the likely fix.

### Retriever update: intent → section rule (2026-09-26)

Same 38 questions × 2 runs and the same judge, re-run with the intent → section retrieval rule;
`v2`/`v3` are the same answer models as `baseline`/`v1`, so the difference is the retriever alone.
Paired per question, 95% CI:

| metric | `qwen3:4b-instruct` (`baseline` → `v2`) | `qwen3:8b` (`v1` → `v3`) |
|---|---|---|
| correct | 0.80 → 0.83 (+0.03 [−0.03, +0.08]) | 0.82 → 0.82 (±0) |
| fact recall | 0.84 → 0.87 (+0.03 [−0.03, +0.10]) | 0.84 → 0.86 (+0.02 [−0.02, +0.05]) |
| grounded | 0.76 → 0.75 (−0.01 [−0.04, +0.01]) | 0.87 → 0.86 (−0.01 [−0.06, +0.03]) |
| cited | 0.93 → 0.97 (+0.03 [−0.03, +0.10]) | 0.95 → 0.97 (+0.02 [−0.02, +0.05]) |

The change is confined to the question it targeted: *Wayfinder trim smithing template location*
went from "the excerpts don't say" to "found in trail ruins, in suspicious gravel" for the 4B
model (fail → pass); the 8B model now names trail ruins too but still adds unsupported claims.
No other question's `correct` grade changed, and no overall difference is statistically
significant — with 38 questions, a single-question fix is expected to be within noise.

### Running it on Kaggle's free GPU

The laptop's 4 GB GPU only fits ~4B models, so the eval is set up to run on Kaggle's T4:

1. `python kaggle/package.py` -> upload `dist/minecraft-rag-kaggle.zip` as a private Kaggle
   Dataset named `minecraft-rag`.
2. Import `kaggle/minecraft_rag_eval.ipynb` into a new notebook; set GPU T4 and Internet on; attach
   the dataset.
3. Run the cells: they install and start Ollama, pull `qwen3:4b-instruct`, `qwen3:8b` and
   `gemma3:12b`, record the harness approval, self-test the judge, run a 7-question pilot for both
   models (`baseline` = 4B, `v1` = 8B), and zip `eval/results/` for download.

`--judge-selftest` checks the judge passes reference answers and fails empty, "I don't know" and
wrong-question answers. `--ids`/`--limit` run a subset, `--variant v1` stores a changed setup
alongside `baseline`, and `--summary` reprints per-type means with 95% CIs. With 38 cases × 2 reps
the noise floor on `correct` is roughly ±11 points.

## In-game mod: how it works

Installation and use are covered in [Setup](#setup-step-by-step) and
[Playing with the mod](#playing-with-the-mod); this section explains the design.

The Fabric client mod (`minecraft-mod/`) registers `/doubt` and `/faq` as client-side commands and
talks to the local backend (`mcrag/server.py`, FastAPI on 127.0.0.1:8765). With each question it
sends the player's biome, dimension and position; they are used only for questions about the
player's surroundings ("what spawns here?"). The backend URL can be changed with
`-Dmcrag.backend=http://host:port` in the game's JVM arguments.

**Recipes come from the game, not the model.** The wiki draws crafting grids as images, so its text
has no pattern or counts, and even with the exact recipe in its context the model garbled rows when
copying them. `recipes-build` reads the game's own recipe files, item tags and English names
(1,536 recipes for 1,005 items in 26.2) and turns each into a readable passage — grid rows, totals,
furnace/blast furnace/smoker inputs, smithing and stonecutting. When a question asks how to craft,
smelt or smith something the game has a recipe for, `/doubt` returns that recipe directly: exact,
instant and without touching the GPU.

**Spawn questions come from the game too.** The wiki's spawn tables lose their category labels in
parsing, and the model mixed up editions, repeated spawn weights, and once said creepers don't
spawn in cherry groves. `spawns-build` reads each biome's natural spawn list from the game jar
(66 biomes, Java Edition). "What mobs spawn here?" (the player's biome), "which hostile mobs spawn
in the plains?" or "do wolves spawn here?" are answered from it directly — grouped as hostile,
animals, water, ambient, with rare mobs marked. The mob questions in the biome FAQs (`/faq`) are
answered from the same lists, replacing the generated answers. Questions about mechanics ("how do slimes spawn?")
or special spawns (the warden) still go to the model. Everything else goes through retrieval and
the model.

**Sharing the GPU with the game.** Measured on a 4 GB laptop GPU: Ollama's default kept the model
loaded for 30 minutes after each answer, holding 2.2 GB of video memory and making the game lag.
The backend unloads the model right after each answer (`keep_alive=0`, 4096-token context).
`serve --cpu` runs the model on the CPU only (about 30–34 s per answer, GPU untouched);
`serve --keep-alive 5m` keeps it loaded between questions instead (faster follow-ups, but it holds
the video memory). Search models are capped at 4 CPU threads.

**Response time.** Measured per answer before these changes: 7–10 s reranking on the CPU, 7–12 s
loading the model, 1–2 s reading the passages, then 15–22 tokens/s of writing — 20–40 s before
anything appeared in chat. Now:

- *Fast search* (`--mode fast`, the server default): hybrid search plus the where/find → Obtaining
  section rule and a mobs/spawn → biome spawn-table rule (the whole Mobs section, since its
  monsters and animals sit in separate chunks), without the cross-encoder — 0.16 s instead of 7.8 s per query. On the fact-level
  eval (`eval --evidence`, expanded index, 2 tutorial chunks max) it covers as many facts in the
  top 8 as reranking (0.918 vs 0.898 on the labelled pages, 0.959 vs 0.939 on any page), though
  its page-level Hit@5 is lower (0.83 vs 0.93). `serve --rerank` brings the cross-encoder back.
- *Warm-up while typing*: the mod calls `/warmup` as soon as the player starts typing after
  `/doubt `, so the model load overlaps the typing.
- *Streaming*: `/doubt/stream` sends each sentence as the model writes it, and the mod prints it
  right away — the first line shows up about 5–7 s after pressing Enter (once warm), the rest
  follow every ~1.5 s. Answers are capped at 350 tokens.
- Recipe and spawn questions never touch the model and answer instantly (see above).

## Next steps

- Calibrate the judges against a few dozen human-labelled answers before hill-climbing on them.
- Multi-turn chat: rewrite follow-up questions into standalone queries before retrieval.
