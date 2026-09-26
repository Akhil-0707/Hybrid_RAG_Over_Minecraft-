"""Biome spawn lists from the game's own data, for "what mobs spawn here?" questions.

The wiki's spawn tables come through the table parser with every column labelled "Monster
category" (the category rows are merged into the header), and the chat model mixed up Java and
Bedrock rows and repeated spawn weights. The game jar holds each biome's natural spawn list as data
(data/minecraft/worldgen/biome/*.json: mob, weight, group size per category), keyed by the same ids
the mod sends (minecraft:cherry_grove), plus English names.

`python -m mcrag spawns-build` reads them from the player's own Minecraft jar into data/spawns.json
(Mojang's data, generated locally and not committed). At question time SpawnBook.answer() turns a
spawn question about a biome - the player's own for "here" questions, or one named in the question -
into a short chat answer, without the model. These are Java Edition lists.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from .recipes import default_jar
from .text import normalize_name

SPAWNS_PATH = Path("data/spawns.json")
BIOME_DIR = "data/minecraft/worldgen/biome/"
MAX_NGRAM = 4
# Game category -> chat label, in display order ("misc" is always empty for biomes).
CATEGORIES = {
    "monster": "Hostile",
    "creature": "Animals",
    "water_creature": "Water animals",
    "water_ambient": "Fish",
    "underground_water_creature": "Underground water",
    "axolotls": "Underground water",
    "ambient": "Ambient",
}
RARE_SHARE = 0.05  # under 5% of its category's spawn weight -> "(rare)"

# A spawn question: "what mobs spawn here", "do creepers spawn in cherry groves", "which animals
# live in the plains". Questions about mechanics ("why/how do slimes spawn") or about doing
# something to mobs ("which mobs drop bones") go to the model instead.
_SPAWN = re.compile(r"\bspawn\w*\b", re.I)
# "Is there any danger here?" is broader than mobs (lava, terrain), so danger words alone don't count.
_MOB_WORD = re.compile(r"\b(mobs?|animals?|monsters?|creatures?|hostiles?|passives?|fish|aquatic|"
                       r"enem(y|ies))\b", re.I)
_WHICH_MOBS = re.compile(r"\b(what|which|any)\b.*" + _MOB_WORD.pattern, re.I)
_WHAT_SPAWNS = re.compile(r"\bwhat (can |does |will )?spawns?\b", re.I)  # "what spawns here"
# Mechanics, doing something to mobs, Bedrock (these lists are Java's) and "only here" questions
# are left to the model.
_OTHER = re.compile(r"\b(why|how(?! many)|drop\w*|attack\w*|breed\w*|tam(e|ing)\w*|kill\w*|fight\w*|"
                    r"avoid\w*|farm\w*|eat\w*|ride|riding|stop\w*|prevent\w*|light\w*|rates?|chance|"
                    r"bedrock|exclusive\w*|unique|only)\b", re.I)
_HOSTILE = re.compile(r"\b(hostiles?|monsters?|dangers?|dangerous|enem(y|ies)|aggressive)\b", re.I)
_ANIMALS = re.compile(r"\b(animals?|passives?|creatures?|friendly|livestock)\b", re.I)
_WATER = re.compile(r"\b(fish|aquatic|water|ocean mobs?)\b", re.I)
WATER_CATEGORIES = ["water_creature", "water_ambient", "underground_water_creature", "axolotls"]


def build(jar: Path | None = None, out_path: Path = SPAWNS_PATH) -> None:
    jar = jar or default_jar()
    if not jar.exists():
        raise SystemExit(f"Minecraft jar not found at {jar} - pass --jar to point at your game jar.")
    with zipfile.ZipFile(jar) as z:
        lang = json.loads(z.read("assets/minecraft/lang/en_us.json"))

        def entity(type_id: str) -> str:
            ns, path = type_id.split(":", 1) if ":" in type_id else ("minecraft", type_id)
            return lang.get(f"entity.{ns}.{path}") or path.replace("_", " ").title()

        biomes = {}
        for n in sorted(z.namelist()):
            if not (n.startswith(BIOME_DIR) and n.endswith(".json")):
                continue
            path = n[len(BIOME_DIR):-len(".json")]
            spawners = json.loads(z.read(n)).get("spawners", {})
            biomes[f"minecraft:{path}"] = {
                "name": lang.get(f"biome.minecraft.{path}") or path.replace("_", " ").title(),
                "spawns": {cat: [{"mob": entity(s["type"]), "weight": s["weight"],
                                  "min": s["minCount"], "max": s["maxCount"]} for s in entries]
                           for cat, entries in spawners.items() if entries},
            }
        # Every entity name, so a question about a mob that is in no biome list (the warden, iron
        # golems) can be recognised and left to the model instead of answered with a list.
        entities = sorted({v for k, v in lang.items()
                           if k.startswith("entity.minecraft.") and k.count(".") == 2})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"source": str(jar), "biomes": biomes, "entities": entities},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"spawn lists for {len(biomes)} biomes -> {out_path}")


def _forms(name: str) -> set[str]:
    """Normalised singular and plural spellings: Wolf -> wolf, wolves; Enderman -> endermen."""
    low = name.lower()
    forms = {low, low + "s", low + "es"}
    if low.endswith("f"):
        forms.add(low[:-1] + "ves")
    if low.endswith("man"):
        forms.add(low[:-3] + "men")
    return {normalize_name(f) for f in forms}


def _groups(e: dict) -> str:
    if e["max"] <= 1:
        return "alone"
    return f"groups of {e['min']}" if e["min"] == e["max"] else f"groups of {e['min']}–{e['max']}"


class SpawnBook:
    def __init__(self, path: Path = SPAWNS_PATH):
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"biomes": {}}
        self.biomes: dict[str, dict] = data["biomes"]
        self.biome_names: dict[str, str] = {}
        for bid, b in self.biomes.items():
            for f in _forms(b["name"]) | {normalize_name(bid.split(":", 1)[1])}:
                self.biome_names.setdefault(f, bid)
        self.spawnable = {e["mob"] for b in self.biomes.values()
                          for es in b["spawns"].values() for e in es}
        self.mob_names: dict[str, str] = {}
        for name in data.get("entities", []):
            for f in _forms(name):
                self.mob_names.setdefault(f, name)

    def __len__(self) -> int:
        return len(self.biomes)

    def resolve(self, biome: str) -> str | None:
        """'minecraft:cherry_grove', 'cherry_grove' or 'Cherry Grove' -> 'minecraft:cherry_grove'."""
        if biome in self.biomes:
            return biome
        key = normalize_name(biome.split(":", 1)[-1])
        return self.biome_names.get(key)

    @staticmethod
    def _scan(words: list[str], table: dict[str, str]) -> tuple[list[str], list[str]]:
        """Greedy longest match of table keys in the words; returns (matches, remaining words)."""
        found, rest, i = [], [], 0
        while i < len(words):
            for n in range(min(MAX_NGRAM, len(words) - i), 0, -1):
                hit = table.get(" ".join(words[i:i + n]))
                if hit:
                    found.append(hit)
                    i += n
                    break
            else:
                rest.append(words[i])
                i += 1
        return list(dict.fromkeys(found)), rest

    def answer(self, question: str, here_biome: str | None = None) -> dict | None:
        """{"lines", "biome", "name"} for a spawn question about a known biome, else None.

        here_biome: the player's biome id, passed only for questions about their surroundings.
        """
        if not (_SPAWN.search(question) or _WHICH_MOBS.search(question)) or _OTHER.search(question):
            return None
        named, rest = self._scan(normalize_name(question).split(), self.biome_names)
        bid = named[0] if named else here_biome
        if bid not in self.biomes:
            return None
        mobs, _ = self._scan(rest, self.mob_names)
        biome = self.biomes[bid]
        if mobs:
            if not set(mobs) <= self.spawnable:
                return None  # a mob with special spawning (warden, golems...): the wiki knows more
            return {"lines": [self._mob_line(biome, m) for m in mobs], "biome": bid, "name": biome["name"]}
        if not (_MOB_WORD.search(question) or _WHAT_SPAWNS.search(question)):
            return None  # "do ocean monuments / trees / snow spawn here" is not about mobs
        cats = list(CATEGORIES)
        if _WATER.search(question) and not (_HOSTILE.search(question) or _ANIMALS.search(question)):
            cats = WATER_CATEGORIES
        elif _HOSTILE.search(question) and not _ANIMALS.search(question):
            cats = ["monster"]
        elif _ANIMALS.search(question) and not _HOSTILE.search(question):
            cats = [c for c in cats if c != "monster"]
        return {"lines": self._list_lines(biome, cats), "biome": bid, "name": biome["name"]}

    def faq_answer(self, question: str, biome: str) -> str | None:
        """The same answer as one paragraph, for a biome FAQ entry ("Hostile: ... Animals: ...")."""
        a = self.answer(question, biome)
        if not a:
            return None
        lines = a["lines"]
        if lines[1:] and lines[1].startswith("• "):  # heading, "• group: mobs" lines, note
            lines = [l[2:] for l in lines[1:-1]] + lines[-1:]
        return " ".join(l if l.endswith((".", "!", "?")) else l + "." for l in lines)

    @staticmethod
    def _mob_line(biome: dict, mob: str) -> str:
        for cat, entries in biome["spawns"].items():
            for e in entries:
                if e["mob"] == mob:
                    kind = CATEGORIES.get(cat, cat).lower()
                    return (f"Yes - {mob} spawns naturally in {biome['name']} ({kind}, {_groups(e)}; "
                            f"Java Edition).")
        owner = biome["name"] + ("'" if biome["name"].endswith("s") else "'s")
        return (f"No - {mob} isn't in {owner} natural spawn list (Java Edition). "
                f"Structures and spawners can still add mobs.")

    @staticmethod
    def _list_lines(biome: dict, cats: list[str]) -> list[str]:
        which = ("Hostile mobs" if cats == ["monster"] else "Water mobs" if cats == WATER_CATEGORIES
                 else "Mobs" if "monster" in cats else "Animals")
        groups: dict[str, list[str]] = {}
        for cat in cats:
            entries = biome["spawns"].get(cat, [])
            total = sum(e["weight"] for e in entries) or 1
            for e in sorted(entries, key=lambda e: -e["weight"]):
                rare = " (rare)" if e["weight"] / total < RARE_SHARE else ""
                groups.setdefault(CATEGORIES[cat], []).append(e["mob"] + rare)
        note = "Structures and spawners in the biome can add other mobs."
        if not groups:
            return [f"No {which.lower()} spawn naturally in {biome['name']} (Java Edition).", note]
        head = f"{which} that spawn naturally in {biome['name']} (Java Edition):"
        if len(groups) == 1:
            return [f"{head} {', '.join(dict.fromkeys(*groups.values()))}", note]
        return [head, *(f"• {label}: {', '.join(dict.fromkeys(mobs))}" for label, mobs in groups.items()),
                note]
