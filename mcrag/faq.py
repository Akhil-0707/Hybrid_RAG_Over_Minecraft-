"""Biome FAQs: a few grounded Q&As per biome, generated ahead of time from the wiki biome pages.

`python -m mcrag faq-build` asks the local model to write FAQs for every Overworld, Nether and End
biome page, using only that page's own excerpts (overview, description, generation, mob spawns,
infobox), and saves them to assets/biome_faq.json. The server serves them instantly, keyed by the
in-game biome id (e.g. minecraft:cherry_grove), so `/faq` needs no model call at play time.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .llm import Ollama, OllamaError
from .text import normalize_name
from .wiki import load_pages

FAQ_PATH = Path("assets/biome_faq.json")
BIOME_CATEGORIES = ("Overworld biomes", "Nether biomes", "End biomes")
# Most useful sections first; Bedrock mob tables duplicate the Java ones, so they come last.
SECTION_ORDER = ["Overview", "Description", "Generation", "Mobs (In Java Edition)", "Infobox"]
MAX_EXCERPT_CHARS = 5000
# In-game ids whose wiki page title isn't just the id in title case.
SPECIAL_IDS = {"the_end": "The End (biome)", "the_void": "The Void"}

FAQ_SYSTEM = """\
You write short FAQs for Minecraft players who are standing in a particular biome. Use only the \
wiki excerpts you are given.

Write 4 questions a player in this biome would likely ask - for example which mobs spawn here, \
what blocks, resources or structures can be found, what is dangerous, or what makes the biome \
special - and answer each in 1-3 sentences using only facts stated in the excerpts. Keep negations \
exactly as the excerpts state them: if the wiki lists things that do NOT spawn or generate in the \
biome, say they don't - never turn such a list into things that do. Skip any question the \
excerpts can't answer. Write plain text, no markdown, in exactly this format:
Q: <question>
A: <answer>"""

_QA = re.compile(r"^\s*Q\s*[:.]\s*(.+?)\s*\n\s*A\s*[:.]\s*(.+?)(?=\n\s*Q\s*[:.]|\Z)", re.S | re.M)


def biome_pages(pages: list[dict]) -> list[dict]:
    return [p for p in pages if any(c in BIOME_CATEGORIES for c in p["categories"])]


def biome_excerpts(title: str, chunks: list[dict]) -> str:
    own = [c for c in chunks if c["title"] == title]
    rank = {s: i for i, s in enumerate(SECTION_ORDER)}
    own.sort(key=lambda c: rank.get(c["section"], len(SECTION_ORDER)))
    parts, used = [], 0
    for c in own:
        block = f"[{c['section']}]\n{c['text']}"
        if used + len(block) > MAX_EXCERPT_CHARS:
            continue
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def parse_faqs(text: str) -> list[dict]:
    out = []
    for q, a in _QA.findall(text.strip()):
        q, a = " ".join(q.split()), " ".join(a.split())
        if q and a:
            out.append({"q": q.rstrip("?") + "?", "a": a})
    return out


def build(model: str = "qwen3:4b-instruct", pages_path: Path = Path("data/pages.jsonl"),
          chunks_path: Path = Path("index/chunks.jsonl"), out_path: Path = FAQ_PATH,
          only: list[str] | None = None) -> None:
    """Generate FAQs for every biome page not yet in out_path (resumable)."""
    pages = biome_pages(load_pages(pages_path))
    if only:
        pages = [p for p in pages if p["title"] in only]
    with chunks_path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    faqs = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    client = Ollama()
    todo = [p for p in pages if p["title"] not in faqs]
    print(f"{len(pages)} biome pages, {len(faqs)} done, {len(todo)} to generate with {model}")
    for n, page in enumerate(todo, 1):
        excerpt = biome_excerpts(page["title"], chunks)
        messages = [{"role": "system", "content": FAQ_SYSTEM},
                    {"role": "user", "content": f"Biome: {page['title']}\n\n<excerpts>\n{excerpt}\n</excerpts>"}]
        t0 = time.time()
        for attempt in range(3):
            try:
                r = client.chat(model, messages, think=False, temperature=0.2, num_predict=700)
                break
            except OllamaError as e:
                if attempt == 2:
                    raise
                print(f"  retry after: {str(e)[:80]}")
                time.sleep(10)
        items = parse_faqs(r.text)
        faqs[page["title"]] = {
            "title": page["title"], "url": page["url"], "aliases": page["aliases"],
            "faqs": items, "model": r.model,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(faqs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  [{n}/{len(todo)}] {page['title']}: {len(items)} FAQs ({time.time() - t0:.1f}s)")


class BiomeFaqs:
    """Look up pre-generated FAQs by in-game biome id or by name."""

    def __init__(self, path: Path = FAQ_PATH):
        self.faqs: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.by_name: dict[str, str] = {}
        for title, entry in self.faqs.items():
            for name in [title, *entry.get("aliases", [])]:
                self.by_name.setdefault(normalize_name(name), title)

    def title_for(self, biome: str) -> str | None:
        """'minecraft:windswept_gravelly_hills', 'Cherry Grove' or 'gravelly mountains' -> title."""
        key = biome.split(":", 1)[-1].strip()
        if key.lower() in SPECIAL_IDS:
            return SPECIAL_IDS[key.lower()]
        return self.by_name.get(normalize_name(key.replace("_", " ")))

    def get(self, biome: str) -> dict | None:
        title = self.title_for(biome)
        return self.faqs.get(title) if title else None
