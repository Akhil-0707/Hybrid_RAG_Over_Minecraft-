"""Chunking and the sparse tokenizer.

The tokenizer is deliberately *not* aggressive: Minecraft queries hinge on exact identifiers
("Swift Sneak III", "netherite_ingot", "Sentry Armor Trim"), so we keep short tokens like roman
numerals, split snake_case IDs into parts *and* keep the joined form, and add bigrams so an exact
multi-word item name outranks documents that merely contain the words somewhere.
"""
from __future__ import annotations

import re

SKIP_SECTIONS = {
    "history", "gallery", "references", "external links", "navigation", "issues", "videos",
    "see also", "trivia", "achievements", "advancements", "data values", "sounds", "notes",
    "screenshots", "in other media", "mojang screenshots", "concept artwork", "development",
}
STOPWORDS = set("""
a an and are as at be by can do does for from how i if in into is it its me my of on or so
that the their them there these this to was what when where which who why will with you your
""".split())

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
_WORD = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def normalize_token(tok: str) -> str:
    # Minimal plural folding (emeralds -> emerald) without a stemmer mangling item names.
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    out: list[str] = []
    for raw in _WORD.findall(text.lower()):
        parts = raw.split("_")
        if len(parts) > 1:
            out.append(raw)  # keep the full game ID, e.g. netherite_ingot
        out += [normalize_token(p) for p in parts if p not in STOPWORDS]
    if bigrams:
        uni = [t for t in out if "_" not in t]
        out += [f"{a}__{b}" for a, b in zip(uni, uni[1:])]
    return out


def normalize_name(name: str) -> str:
    """Canonical form for exact entity matching: 'Sentry armour trims' -> 'sentry armour trim'."""
    return " ".join(normalize_token(t) for t in _WORD.findall(name.lower().replace("_", " ")))


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a TextExtracts plaintext page into (section path, body) pairs."""
    sections, path = [], []
    last_end, last_path = 0, ""
    for m in _HEADING.finditer(text):
        sections.append((last_path, text[last_end:m.start()]))
        level = len(m.group(1)) - 1
        path = path[: level - 1] + [m.group(2)]
        last_path, last_end = " > ".join(path), m.end()
    sections.append((last_path, text[last_end:]))
    return sections


def chunk_page(page: dict, max_words: int = 180, overlap: int = 40) -> list[dict]:
    chunks = []
    for path, body in split_sections(page["text"]):
        top = path.split(" > ")[0].lower()
        if top in SKIP_SECTIONS:
            continue
        words = body.split()
        if len(words) < 8:
            continue
        step = max_words - overlap
        for start in range(0, max(len(words) - overlap, 1), step):
            piece = " ".join(words[start:start + max_words])
            chunks.append({
                "title": page["title"],
                "section": path or "Overview",
                "url": page["url"],
                "text": piece,
                "kind": "prose",
            })
    return chunks + chunk_tables(page, max_words)


def chunk_tables(page: dict, max_words: int = 180) -> list[dict]:
    """Infobox -> one chunk; each table -> chunks of whole rows (rows are never split)."""
    def make(section: str, lines: list[str], kind: str) -> dict:
        return {"title": page["title"], "section": section, "url": page["url"],
                "text": "\n".join(lines), "kind": kind}

    chunks = []
    if page.get("infobox"):
        chunks.append(make("Infobox", page["infobox"], "infobox"))
    for table in page.get("tables", []):
        section = table["section"] + (f" ({table['caption']})" if table["caption"] else "")
        group, words = [], 0
        for row in table["rows"]:
            n = len(row.split())
            if group and words + n > max_words:
                chunks.append(make(section, group, "table"))
                group, words = [], 0
            group.append(row)
            words += n
        if group:
            chunks.append(make(section, group, "table"))
    return chunks


def chunk_header(chunk: dict) -> str:
    """Title/section prefix used for both indexes, so chunks keep their page context."""
    return f"{chunk['title']} > {chunk['section']}"
