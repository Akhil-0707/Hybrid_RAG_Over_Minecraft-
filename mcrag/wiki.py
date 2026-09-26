"""Fetch pages from the Minecraft Wiki MediaWiki API and cache them as JSONL."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

API = "https://minecraft.wiki/api.php"
HEADERS = {"User-Agent": "minecraft-rag/0.1 (educational hybrid-RAG project)"}

DEFAULT_CATEGORIES = [
    "Items", "Blocks", "Enchantments", "Smithing templates", "Hostile mobs",
    "Passive mobs", "Neutral mobs", "Effects", "Potions", "Tools", "Armor",
    "Food", "Overworld biomes", "Nether biomes", "Generated structures",
    # Game systems and world: hunger, experience, crafting, light, weather, game modes, ...
    "Gameplay", "Environment", "Dimensions", "End biomes", "Game modes", "Transport",
    "Villager mechanics", "Piglin mechanics", "Illager mechanics", "Plants",
    "Redstone", "Redstone mechanics", "Mechanisms", "Commands",
]
# Tutorial pages live in their own namespace (not category members of the main namespace);
# they answer the "how do I ..." questions that item pages don't.
TUTORIAL_NAMESPACE = 10010
SEED_TITLES = [
    "Trading", "Villager", "Wandering Trader", "Armor trimming", "Smithing Table",
    "Enchanting", "Brewing", "Raid", "Emerald Ore", "Ore", "Tutorial:Mining",
    "Tutorial:Trading hall", "Tutorial:Raid farming", "Hero of the Village",
    "Anvil", "Mending", "Nether", "The End", "Redstone circuits", "Totem of Undying",
    "Bartering", "Conduit", "Vault", "Trial Chambers", "Elytra",
]
# Joke features and spin-off games pollute the corpus; drop pages in these categories.
EXCLUDE_CATEGORY = re.compile(
    r"April Fools|Joke|Fictional|Comic|Book objects|Mini-Series|Dungeons|Legends|"
    r"Minecraft Earth|Story Mode|Spin-off|Education|Removed features|Disambiguation|"
    r"Outdated tutorials",
    re.I,
)
# Tutorial sub-pages that are templates or console-edition duplicates, not content.
EXCLUDE_TITLE = re.compile(r"/header$|/Legacy Console Edition$|/Bedrock Edition$", re.I)


class WikiClient:
    def __init__(self, delay: float = 0.15):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay

    def _get(self, **params) -> dict:
        params |= {"format": "json", "formatversion": "2"}
        for attempt in range(4):
            try:
                r = self.session.get(API, params=params, timeout=30)
                r.raise_for_status()
                time.sleep(self.delay)
                return r.json()
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Wiki API failed for {params}")

    def category_members(self, category: str) -> list[str]:
        titles, cont = [], {}
        while True:
            d = self._get(action="query", list="categorymembers", cmtitle=f"Category:{category}",
                          cmnamespace=0, cmlimit=500, **cont)
            titles += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
            if "continue" not in d:
                return titles
            cont = d["continue"]

    def namespace_pages(self, namespace: int) -> list[str]:
        """All non-redirect page titles in a namespace (e.g. Tutorial:)."""
        titles, cont = [], {}
        while True:
            d = self._get(action="query", list="allpages", apnamespace=namespace, aplimit=500,
                          apfilterredir="nonredirects", **cont)
            titles += [p["title"] for p in d.get("query", {}).get("allpages", [])]
            if "continue" not in d:
                return titles
            cont = d["continue"]

    def page(self, title: str) -> dict | None:
        """Plain-text extract + categories + redirect aliases for one page.

        TextExtracts only returns one full-page extract per request, so this is per-title.
        """
        d = self._get(action="query", prop="extracts|categories|redirects", explaintext=1,
                      cllimit="max", rdlimit="max", redirects=1, titles=title)
        pages = d.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or not pages[0].get("extract"):
            return None
        p = pages[0]
        return {
            "title": p["title"],
            "url": "https://minecraft.wiki/w/" + p["title"].replace(" ", "_"),
            "categories": [c["title"].removeprefix("Category:") for c in p.get("categories", [])],
            "aliases": [r["title"] for r in p.get("redirects", [])],
            "text": p["extract"],
        }


def crawl(out_path: Path, categories=DEFAULT_CATEGORIES, seeds=SEED_TITLES,
          tutorials: bool = True) -> None:
    """Fetch every page in `categories` + `seeds` (+ the Tutorial namespace).

    Resumable: skips titles already in out_path.
    """
    client = WikiClient()
    titles: list[str] = list(seeds)
    for cat in categories:
        members = client.category_members(cat)
        print(f"  Category:{cat}: {len(members)} pages")
        titles += members
    if tutorials:
        tut = client.namespace_pages(TUTORIAL_NAMESPACE)
        print(f"  Tutorial namespace: {len(tut)} pages")
        titles += tut
    titles = list(dict.fromkeys(t for t in titles if not t.startswith(("Category:", "File:"))
                                and not EXCLUDE_TITLE.search(t)))

    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["requested"])
    todo = [t for t in titles if t not in done]
    print(f"{len(titles)} titles, {len(done)} cached, {len(todo)} to fetch")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, title in enumerate(todo, 1):
            page = client.page(title)
            rec = {"requested": title, "page": page}
            if page and any(EXCLUDE_CATEGORY.search(c) for c in page["categories"]):
                rec["page"] = None
            kept += rec["page"] is not None
            skipped += rec["page"] is None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                f.flush()
                print(f"  {i}/{len(todo)} fetched ({kept} kept, {skipped} skipped)")
    print(f"Done: {kept} new pages kept, {skipped} skipped")


def load_pages(path: Path) -> list[dict]:
    """Load cached pages, de-duplicated by canonical title (redirects can collapse requests)."""
    pages: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)["page"]
            if page:
                pages[page["title"]] = page
    return list(pages.values())
