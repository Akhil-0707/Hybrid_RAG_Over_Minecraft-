"""Extract wiki tables and infoboxes via action=parse (TextExtracts drops them).

Each table row becomes one self-contained line of "Header: value" pairs, with rowspans/colspans
expanded so a trade row still says which villager level it belongs to:

    Level: Novice; Probability JE: 40%; Villager wants: 15 × Coal; Player receives: Emerald; ...

Only the extracted rows are cached (data/tables.jsonl); the raw HTML (up to ~1 MB/page) is not.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .text import SKIP_SECTIONS
from .wiki import WikiClient

try:
    import lxml  # noqa: F401
    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"

# navboxes, data-value ID tables and interactive calculators are noise for retrieval.
SKIP_TABLE_CLASSES = {"navbox", "id-table", "calculator-container"}
MAX_CELL_CHARS = 200
_WS = re.compile(r"\s+")
_JSON_BLOB = re.compile(r"\{\s*\"[^{}]*\}")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍­﻿"), None)


def cell_text(cell: Tag) -> str:
    for junk in cell.select("sup.reference, .mw-editsection, style, script, .hidden, "
                            "[style*='display:none'], [style*='display: none']"):
        junk.decompose()
    text = cell.get_text(" ", strip=True).translate(_INVISIBLE)
    text = _JSON_BLOB.sub("", text)  # loot tables embed hidden JSON data for scripts
    return _WS.sub(" ", text).strip()[:MAX_CELL_CHARS]


def _span(cell: Tag, attr: str) -> int:
    try:
        return max(1, min(int(cell.get(attr, 1)), 100))
    except ValueError:
        return 1


def table_grid(table: Tag) -> list[list[tuple[str, bool]]]:
    """Expand rowspan/colspan into a rectangular grid of (text, is_header) cells."""
    grid: list[list[tuple[str, bool] | None]] = []
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    for r, tr in enumerate(rows):
        while len(grid) <= r:
            grid.append([])
        c = 0
        for cell in tr.find_all(["th", "td"], recursive=False):
            while c < len(grid[r]) and grid[r][c] is not None:
                c += 1
            value = (cell_text(cell), cell.name == "th")
            for dr in range(_span(cell, "rowspan")):
                while len(grid) <= r + dr:
                    grid.append([])
                row = grid[r + dr]
                for dc in range(_span(cell, "colspan")):
                    while len(row) <= c + dc:
                        row.append(None)
                    row[c + dc] = value
            c += _span(cell, "colspan")
    return [[cell or ("", False) for cell in row] for row in grid if row]


def table_rows(table: Tag) -> list[str]:
    """Turn a table into one 'Header: value; ...' line per data row."""
    grid = table_grid(table)
    n_head = 0
    while n_head < len(grid) and all(is_th for _, is_th in grid[n_head]):
        n_head += 1
    if n_head == len(grid):  # all-header table: treat the first row as the header
        n_head = min(1, len(grid))
    width = max((len(r) for r in grid), default=0)
    headers = []
    for c in range(width):
        parts: list[str] = []
        for r in range(n_head):
            if c < len(grid[r]) and grid[r][c][0] and grid[r][c][0] not in parts:
                parts.append(grid[r][c][0])
        headers.append(" ".join(parts))

    lines = []
    for row in grid[n_head:]:
        # Adjacent columns under the same header (a colspan header over unnamed sub-columns,
        # or a colspan cell) merge into one "Header: a, b, c" pair.
        groups: list[tuple[str, list[str]]] = []
        for c, (text, _) in enumerate(row):
            head = headers[c] if c < len(headers) else ""
            if groups and groups[-1][0] == head:
                values = groups[-1][1]
            else:
                values = []
                groups.append((head, values))
            if text and text != head and text not in values:
                values.append(text)
        pairs = [f"{head}: {', '.join(vals)}" if head else ", ".join(vals)
                 for head, vals in groups if vals]
        if pairs:
            lines.append("; ".join(pairs))
    return lines


def infobox_rows(table: Tag) -> list[str]:
    lines = []
    for tr in table.find_all("tr"):
        th, td = tr.find("th"), tr.find("td")
        if th and td:
            key, val = cell_text(th), cell_text(td)
            if key and val:
                lines.append(f"{key}: {val}")
    return lines


def extract(html: str) -> dict:
    """Return {"infobox": [lines], "tables": [{"section", "caption", "rows"}]} for one page."""
    soup = BeautifulSoup(html, PARSER)
    out: dict = {"infobox": [], "tables": []}
    path: list[str] = []
    for el in soup.find_all(["h2", "h3", "h4", "h5", "table"]):
        if el.name != "table":
            level = int(el.name[1]) - 1
            path = path[: level - 1] + [cell_text(el)]
            continue
        if el.find_parent("table") is not None:
            continue  # nested layout tables are covered by their parent's cell text
        classes = set(el.get("class", []))
        if "infobox-rows" in classes:
            out["infobox"] += infobox_rows(el)
            continue
        if "wikitable" not in classes or classes & SKIP_TABLE_CLASSES:
            continue
        if path and path[0].lower() in SKIP_SECTIONS:
            continue
        rows = table_rows(el)
        if rows:
            caption = el.find("caption")
            out["tables"].append({
                "section": " > ".join(path) or "Overview",
                "caption": cell_text(caption) if caption else "",
                "rows": rows,
            })
    return out


def fetch(client: WikiClient, title: str) -> dict:
    d = client._get(action="parse", page=title, prop="text", redirects=1,
                    disableeditsection=1, disabletoc=1, disablelimitreport=1)
    html = d.get("parse", {}).get("text", "")
    return {"title": title, **extract(html)} if html else {"title": title, "infobox": [], "tables": []}


def crawl_tables(titles: list[str], out_path: Path, workers: int = 2) -> None:
    """Fetch tables for every title not yet in out_path (resumable, like the page crawl)."""
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["title"] for line in f}
    todo = [t for t in titles if t not in done]
    print(f"Tables: {len(titles)} pages, {len(done)} cached, {len(todo)} to fetch")
    client = WikiClient()
    n_tables = 0
    with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for i, rec in enumerate(pool.map(lambda t: fetch(client, t), todo), 1):
            n_tables += len(rec["tables"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                f.flush()
                print(f"  {i}/{len(todo)} pages, {n_tables} tables so far")
    print(f"Done: {n_tables} tables from {len(todo)} pages")


def load_tables(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {rec["title"]: rec for rec in map(json.loads, f)}
