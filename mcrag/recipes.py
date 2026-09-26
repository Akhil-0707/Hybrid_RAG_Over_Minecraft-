"""Exact recipes from the game's own data, for "how do I craft X?" questions.

The wiki draws crafting recipes as picture grids, so the text crawl only keeps "Ingredients:
Diamond + Stick" - no quantities, no layout - and the model invents patterns. The game jar holds
every recipe as data (shaped pattern, ingredients, output count), plus item tags and English names.

`python -m mcrag recipes-build` reads them from the player's own Minecraft jar (default
%APPDATA%/.minecraft/versions/26.2/26.2.jar) into data/recipes.json - Mojang's data, so it is
generated locally and not committed. At question time, RecipeBook.passages() turns a craft/make/
smelt question about a known item into exact recipe passages that go first in the model's context.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

from .text import normalize_name

RECIPES_PATH = Path("data/recipes.json")
STATIONS = {
    "minecraft:smelting": "furnace", "minecraft:blasting": "blast furnace",
    "minecraft:smoking": "smoker", "minecraft:campfire_cooking": "campfire",
}
MAX_NGRAM = 6
_INTENT = re.compile(r"\b(craft\w*|recipes?|make|making|made|build|smelt\w*|cook\w*|stonecutt\w*|"
                     r"smith\w*|how (?:do|can) (?:i|you) get)\b", re.I)
_COOK_INTENT = re.compile(r"\b(smelt\w*|cook\w*|furnace|blast\w*|smoker|campfire)\b", re.I)
# A partial name only counts if the missing words are a variant: "bed" -> "White Bed",
# "pickaxe" -> "Iron Pickaxe", but not "armor" -> "Leather Horse Armor" or "iron" -> "Block of Iron".
_VARIANT_WORDS = set("""
white orange magenta light blue yellow lime pink gray grey cyan purple brown green red black
oak spruce birch jungle acacia dark mangrove cherry bamboo crimson warped pale
wooden stone iron golden gold diamond netherite copper leather chainmail
""".split())
# Representative variant when a family matches, most preferred first: the first tier a player
# makes (a wooden pickaxe, then iron for armour, which has no wooden tier).
_PREFERRED = ("white", "oak", "wooden", "iron")


def default_jar(version: str = "26.2") -> Path:
    return Path(os.environ.get("APPDATA", "")) / ".minecraft" / "versions" / version / f"{version}.jar"


class _Names:
    def __init__(self, lang: dict, tags: dict[str, list[str]]):
        self.lang, self.tags = lang, tags

    def item(self, item_id: str) -> str:
        ns, _, path = item_id.partition(":")
        if path.endswith("_smithing_template"):
            # Every template's in-game name is just "Smithing Template"; the id says which one.
            return path.replace("_", " ").title()
        for kind in ("item", "block"):
            name = self.lang.get(f"{kind}.{ns}.{path}")
            if name:
                return name
        return path.replace("_", " ").title()

    def tag_items(self, tag: str, seen: set | None = None) -> list[str]:
        seen = seen or set()
        if tag in seen:
            return []
        seen.add(tag)
        out = []
        for v in self.tags.get(tag, []):
            v = v["id"] if isinstance(v, dict) else v
            out += self.tag_items(v[1:], seen) if v.startswith("#") else [v]
        return out

    def ingredient(self, ing) -> str:
        """'minecraft:stick' -> 'Stick'; '#minecraft:planks' -> 'any planks'; lists -> 'A or B'."""
        if isinstance(ing, list):
            return " or ".join(dict.fromkeys(self.ingredient(i) for i in ing))
        if isinstance(ing, dict):  # older formats: {"item": ...} / {"tag": ...}
            ing = ing.get("item") or ("#" + ing["tag"] if "tag" in ing else str(ing))
        if ing.startswith("#"):
            items = self.tag_items(ing[1:])
            if len(items) == 1:
                return self.item(items[0])
            return "any " + ing[1:].split(":", 1)[-1].replace("_", " ")
        return self.item(ing)


def _result(recipe: dict) -> tuple[str | None, int]:
    res = recipe.get("result")
    if isinstance(res, str):
        return res, 1
    if isinstance(res, dict):
        return res.get("id") or res.get("item"), int(res.get("count", 1))
    return None, 1


def cook_input(recipe: dict, names: _Names) -> str | None:
    """The ingredient of a furnace/smoker/campfire recipe ("Iron Ore"), for reverse lookup."""
    if recipe.get("type") in STATIONS:
        return names.ingredient(recipe["ingredient"])
    return None


def describe(recipe: dict, names: _Names) -> tuple[str, str] | None:
    """Return (result display name, readable recipe text) or None for unsupported types."""
    rtype = recipe.get("type", "")
    rid, count = _result(recipe)
    if not rid:
        return None
    out = names.item(rid)
    makes = f"makes {count}" if count > 1 else "makes 1"
    if rtype == "minecraft:crafting_shaped":
        key = {k: names.ingredient(v) for k, v in recipe["key"].items()}
        rows, total = [], Counter()
        for n, row in enumerate(recipe["pattern"], 1):
            cells = [key.get(ch, "empty") if ch != " " else "empty" for ch in row]
            total.update(c for c in cells if c != "empty")
            rows.append(f"Row {n}: " + ", ".join(cells))
        width = max(len(r) for r in recipe["pattern"])
        grid = "crafting table" if width > 2 or len(recipe["pattern"]) > 2 else "2x2 inventory grid or crafting table"
        text = (f"{out}: shaped recipe on a {grid} ({makes}). Place exactly in this pattern:\n"
                + "\n".join(rows) + "\nTotal: " + ", ".join(f"{n} {k}" for k, n in total.items()))
    elif rtype == "minecraft:crafting_shapeless":
        total = Counter(names.ingredient(i) for i in recipe["ingredients"])
        grid = "crafting table" if sum(total.values()) > 4 else "2x2 inventory grid or crafting table"
        text = (f"{out}: shapeless recipe on a {grid} ({makes}), ingredients in any arrangement: "
                + ", ".join(f"{n} {k}" for k, n in total.items()))
    elif rtype in STATIONS:
        xp = recipe.get("experience")
        text = (f"{out}: cook {names.ingredient(recipe['ingredient'])} in a {STATIONS[rtype]}"
                + (f" ({xp} experience)" if xp else "") + ".")
    elif rtype == "minecraft:stonecutting":
        text = f"{out}: use a stonecutter on {names.ingredient(recipe['ingredient'])} ({makes})."
    elif rtype == "minecraft:smithing_transform":
        text = (f"{out}: smithing table with {names.ingredient(recipe['template'])} (template) + "
                f"{names.ingredient(recipe['base'])} (base) + {names.ingredient(recipe['addition'])}.")
    elif rtype == "minecraft:crafting_transmute":
        text = (f"{out}: combine {names.ingredient(recipe['input'])} with "
                f"{names.ingredient(recipe['material'])} on a crafting grid.")
    else:
        return None  # special recipes (banners, map cloning, trims ...) are handled by the wiki text
    return out, text


def build(jar: Path | None = None, out_path: Path = RECIPES_PATH) -> None:
    jar = jar or default_jar()
    if not jar.exists():
        raise SystemExit(f"Minecraft jar not found at {jar} - pass --jar to point at your game jar.")
    with zipfile.ZipFile(jar) as z:
        lang = json.loads(z.read("assets/minecraft/lang/en_us.json"))
        tags = {}
        for n in z.namelist():
            m = re.fullmatch(r"data/minecraft/tags/item/(.+)\.json", n)
            if m:
                tags[f"minecraft:{m.group(1)}"] = json.loads(z.read(n)).get("values", [])
        names = _Names(lang, tags)
        book: dict[str, list[str]] = {}
        cooks: dict[str, list[str]] = {}  # "Iron Ore" -> ["Iron Ingot", ...]
        skipped = Counter()
        for n in sorted(z.namelist()):
            if not re.fullmatch(r"data/minecraft/recipe/.+\.json", n):
                continue
            recipe = json.loads(z.read(n))
            try:
                d = describe(recipe, names)
            except (KeyError, TypeError):
                d = None
            if d is None:
                skipped[recipe.get("type", "?")] += 1
                continue
            book.setdefault(d[0], []).append(d[1])
            src = cook_input(recipe, names)
            if src:
                for ingredient in src.split(" or "):
                    cooks.setdefault(ingredient, [])
                    if d[0] not in cooks[ingredient]:
                        cooks[ingredient].append(d[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"source": str(jar), "recipes": book, "cook_inputs": cooks},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(len(v) for v in book.values())} recipes for {len(book)} items -> {out_path}"
          f" (skipped special types: {dict(skipped)})")


class RecipeBook:
    def __init__(self, path: Path = RECIPES_PATH):
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"recipes": {}}
        self.recipes: dict[str, list[str]] = data["recipes"]
        self.cook_inputs: dict[str, list[str]] = {}
        for k, v in data.get("cook_inputs", {}).items():
            key = normalize_name(k)
            self.cook_inputs.setdefault(key, v)
            if key.startswith("raw "):  # "cook beef" means Raw Beef
                self.cook_inputs.setdefault(key[4:], v)
        self.by_name: dict[str, list[str]] = {}
        for name in self.recipes:
            self.by_name.setdefault(normalize_name(name), []).append(name)
        # "bed" -> White Bed, ...: a family suffix maps to names whose extra words are all variant
        # words (colours, woods, materials).
        self.by_family: dict[str, list[str]] = {}
        for name in self.recipes:
            words = normalize_name(name).split()
            for i in range(1, len(words)):
                if all(w in _VARIANT_WORDS for w in words[:i]):
                    self.by_family.setdefault(" ".join(words[i:]), []).append(name)
        for fam, members in self.by_family.items():
            members.sort(key=lambda n: (next((i for i, p in enumerate(_PREFERRED)
                                              if normalize_name(n).startswith(p)), len(_PREFERRED)), n))

    def __len__(self) -> int:
        return len(self.recipes)

    def _scan(self, question: str, table: dict[str, list[str]], family: bool) -> list[str]:
        toks = normalize_name(question).split()
        found: list[str] = []
        i = 0
        while i < len(toks):
            for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                phrase = " ".join(toks[i:i + n])
                hits = table.get(phrase) or (self.by_family.get(phrase, [])[:1] if family else [])
                if hits:
                    found += [h for h in hits if h not in found]
                    i += n
                    break
            else:
                i += 1
        return found

    def match(self, question: str) -> list[str]:
        """Item names whose recipes a craft/make/smelt question is asking about."""
        if not _INTENT.search(question):
            return []
        if _COOK_INTENT.search(question):
            # "smelt iron ore": iron ore is the input - look up what it cooks into.
            cooked = self._scan(question, self.cook_inputs, family=False)
            if cooked:
                return cooked[:2]
        return self._scan(question, self.by_name, family=True)[:2]

    def passages(self, question: str) -> list[dict]:
        """Recipe passages shaped like index chunks, most specific item first."""
        out = []
        cooking = bool(_COOK_INTENT.search(question))
        for name in self.match(question):
            words = normalize_name(name).split()
            family = next((" ".join(words[i:]) for i in range(1, len(words))
                           if " ".join(words[i:]) in self.by_family), None)
            # From-scratch recipes first; ones that recolour/convert an existing item of the same
            # family ("Black Dye + any Bed") last - they are rarely what "how do I make" means.
            noun = words[-1]

            q_words = set(normalize_name(question).split())

            def order(t: str) -> tuple:
                ingredients = t.split(":", 1)[1].lower()
                is_cook = ingredients.lstrip().startswith("cook")
                ing_words = normalize_name(ingredients.split(" in a ")[0]).split()
                # For "smelt iron ore": plain Iron Ore before Deepslate Iron Ore, furnace first.
                extra = len([w for w in ing_words if w not in q_words and w != "cook"])
                return (cooking and not is_cook, cooking and extra, cooking and "in a furnace" not in t,
                        noun in normalize_name(ingredients).split(), "shaped" not in t)

            ranked = sorted(self.recipes[name], key=order)
            fresh = [t for t in ranked if not order(t)[3]]  # not made from another item of its family
            text = "\n\n".join((fresh or ranked)[:2])
            others = [n for n in self.by_family.get(family, []) if n != name] if family else []
            if others:
                text += (f"\n(Other variants such as {', '.join(others[:3])} use the same pattern "
                         f"with the matching material.)")
            out.append({"title": name, "section": "Recipe (game data)",
                        "url": "https://minecraft.wiki/w/" + name.replace(" ", "_"),
                        "text": text, "kind": "recipe"})
        return out
