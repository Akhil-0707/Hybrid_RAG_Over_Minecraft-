"""Generate kaggle/minecraft_rag_eval.ipynb (run from the repo root: python kaggle/build_notebook.py).

Every shell step goes through `sh()`, which streams output to the notebook and appends it to
/kaggle/working/run.log, so a failed batch run on Kaggle can be diagnosed from its output files.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "minecraft_rag_eval.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    ast.parse(s)  # fail here, not on Kaggle
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


PILOT = "broad-02 broad-07 exact-04 table-01 table-04 none-02 none-05"

CELLS = [
    md(r"""
# Minecraft RAG — open-weight answer eval (Ollama on Kaggle)

Runs `mcrag ask` with open-weight models served by Ollama on Kaggle's GPU and grades the answers
with a local judge. Two answer models are benchmarked on the same questions:

| variant | answer model | judge |
|---|---|---|
| `baseline` | `qwen3:4b-instruct` | `gemma3:12b` |
| `v1` | `qwen3:8b` | `gemma3:12b` |

Needs: GPU accelerator, Internet on, and the private dataset `minecraft-rag` attached. Every shell
step's output is also appended to `/kaggle/working/run.log`, kept in the notebook output so a
failed run can be diagnosed.
"""),
    code(r'''
# 1. Helpers, then copy the project out of the read-only input dir and install missing packages.
import os, shutil, subprocess, glob, zipfile, time, pathlib
LOG = "/kaggle/working/run.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def sh(cmd, check=True):
    """Run a bash command, stream its output to the notebook and run.log, fail loudly."""
    log(f"$ {cmd}")
    p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        line = line.rstrip("\n")
        if "warn" in line.lower():
            continue
        log(line)
        tail = (tail + [line])[-30:]
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {cmd}\n" + "\n".join(tail))
    return p.returncode


log(f"=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
PROJECT = "/kaggle/working/minecraft-rag"
if not os.path.exists(PROJECT):
    src = glob.glob("/kaggle/input/**/minecraft-rag/mcrag", recursive=True)
    if src:
        shutil.copytree(os.path.dirname(src[0]), PROJECT)
    else:
        zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
        assert zips, "dataset not found - attach the minecraft-rag dataset"
        zipfile.ZipFile(zips[0]).extractall("/kaggle/working")
assert os.path.exists(os.path.join(PROJECT, "mcrag")), os.listdir("/kaggle/working")
os.chdir(PROJECT)
sh("pip install -q rank_bm25 2>&1 | tail -n 3")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
'''),
    code(r'''
# 2. Install Ollama (official script; it unpacks a .tar.zst, so zstd must exist) and start it.
sh("(command -v zstd || (apt-get update -qq && apt-get install -y -qq zstd)) 2>&1 | tail -n 2")
sh("curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -n 5")
sh("ollama --version")
import requests
server = subprocess.Popen(["ollama", "serve"], stdout=open("/tmp/ollama.log", "w"),
                          stderr=subprocess.STDOUT,
                          env={**os.environ, "OLLAMA_KEEP_ALIVE": "30m",
                               "OLLAMA_MAX_LOADED_MODELS": "1"})  # one model in VRAM at a time
for _ in range(60):
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        log("ollama is up")
        break
    except requests.RequestException:
        time.sleep(1)
else:
    raise RuntimeError(open("/tmp/ollama.log").read()[-3000:])
'''),
    code(r'''
# 3. Download the models (~2.5 GB + 5.2 GB + 8.1 GB).
for m in ["qwen3:4b-instruct", "qwen3:8b", "gemma3:12b"]:
    sh(f"ollama pull {m} 2>&1 | tail -n 1")
sh("ollama list")
'''),
    code(r'''
# 4. Smoke test: one question through the real pipeline (retrieval + rerank + qwen3:4b-instruct).
#    The first run also downloads the embedding and reranker models from Hugging Face.
sh('python -m mcrag ask --model qwen3:4b-instruct "how much health does a creeper have"')
'''),
    md(r"""
## 5. Harness approval

The owner reviewed and approved the eval harness in chat on 2026-09-24 before this run was
pushed; this cell records that approval for these exact files.
"""),
    code(r'''
sh("python -m mcrag answer-eval --approve-harness")
'''),
    code(r'''
# 6. Judge self-test: gemma3:12b must pass reference answers and fail empty, "I don't know",
#    wrong-question and invented answers. A FAIL is recorded, not fatal - check run.log.
rc = sh("python -m mcrag answer-eval --judge-selftest", check=False)
log(f"JUDGE SELFTEST {'PASS' if rc == 0 else 'FAIL'}")
'''),
    code(rf'''
# 7. Pilot: 7 questions (all four types), 1 run each, both answer models, then grading.
PILOT = "{PILOT}"
sh(f"python -m mcrag answer-eval --variant baseline --model qwen3:4b-instruct --phase answer --ids {{PILOT}}")
sh(f"python -m mcrag answer-eval --variant v1 --model qwen3:8b --phase answer --ids {{PILOT}}")
pathlib.Path("eval/results/v1/change.md").write_text(
    "# Answer model qwen3:4b-instruct -> qwen3:8b\n\nSame retrieval, prompt and judge; only the answer model changes.\n")
# Grading loads gemma3:12b once, after both answer phases.
sh(f"python -m mcrag answer-eval --variant baseline --phase grade --ids {{PILOT}}")
sh(f"python -m mcrag answer-eval --variant v1 --phase grade --ids {{PILOT}}")
'''),
    code(r'''
# 8. Package results (answers, grades, traces, errors) plus run.log for download.
shutil.copy(LOG, "eval/results/run.log")
shutil.make_archive("/kaggle/working/answer-eval-results", "zip", "eval", "results")
log("wrote /kaggle/working/answer-eval-results.zip")
'''),
    md(r"""
## 9. Full run — only after the pilot's grading has been reviewed

38 questions x 2 runs per model. Runs resume where they stopped.
"""),
    code(r'''
FULL_RUN = True  # grading signed off 2026-09-25 after pilot 2
if FULL_RUN:
    sh("python -m mcrag answer-eval --variant baseline --model qwen3:4b-instruct --phase answer --reps 2")
    sh("python -m mcrag answer-eval --variant v1 --model qwen3:8b --phase answer --reps 2")
    sh("python -m mcrag answer-eval --variant baseline --phase grade")
    sh("python -m mcrag answer-eval --variant v1 --phase grade")
    shutil.copy(LOG, "eval/results/run.log")
    shutil.make_archive("/kaggle/working/answer-eval-results", "zip", "eval", "results")
'''),
]

if __name__ == "__main__":
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")
