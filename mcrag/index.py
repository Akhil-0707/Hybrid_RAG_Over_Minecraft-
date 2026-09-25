"""Build and load the on-disk index: chunks, dense embeddings, and the exact-name entity table."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .tables import load_tables
from .text import STOPWORDS, chunk_header, chunk_page, normalize_name
from .wiki import load_pages

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_entity_table(pages: list[dict]) -> dict[str, list[str]]:
    """Map normalized page titles and redirect aliases to canonical page titles.

    Single-word redirects are dropped: the wiki has redirects like "price" -> Trading or
    "job" -> Villager that would fire on ordinary questions. Single-word *titles* are kept.
    """
    table: dict[str, set[str]] = {}
    for page in pages:
        for n, name in enumerate([page["title"], *page["aliases"]]):
            if ":" in name:  # skip namespaced redirects like "Tutorial:..."
                name = name.split(":", 1)[1]
            key = normalize_name(name)
            if len(key) < 3 or key in STOPWORDS or (n > 0 and " " not in key):
                continue
            table.setdefault(key, set()).add(page["title"])
    return {k: sorted(v) for k, v in table.items()}


def build(pages_path: Path, index_dir: Path, tables_path: Path | None = None,
          batch_size: int = 64) -> None:
    from sentence_transformers import SentenceTransformer

    pages = load_pages(pages_path)
    tables = load_tables(tables_path) if tables_path else {}
    for page in pages:
        extra = tables.get(page["title"], {})
        page["infobox"], page["tables"] = extra.get("infobox", []), extra.get("tables", [])
    chunks = [c for p in pages for c in chunk_page(p)]
    kinds = Counter(c["kind"] for c in chunks)
    print(f"{len(pages)} pages ({sum(p['title'] in tables for p in pages)} with tables) -> "
          f"{len(chunks)} chunks {dict(kinds)}")

    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    entities = build_entity_table(pages)
    (index_dir / "entities.json").write_text(json.dumps(entities, ensure_ascii=False), encoding="utf-8")
    print(f"{len(entities)} exact-match names/aliases")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"{chunk_header(c)}\n{c['text']}" for c in chunks]
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(index_dir / "embeddings.npy", emb.astype(np.float32))
    (index_dir / "meta.json").write_text(json.dumps({"model": EMBED_MODEL, "dim": emb.shape[1]}))
    print(f"Saved index to {index_dir}")


def load(index_dir: Path) -> tuple[list[dict], np.ndarray, dict[str, list[str]], str]:
    with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    emb = np.load(index_dir / "embeddings.npy")
    entities = json.loads((index_dir / "entities.json").read_text(encoding="utf-8"))
    model = json.loads((index_dir / "meta.json").read_text())["model"]
    return chunks, emb, entities, model
