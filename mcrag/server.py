"""Local HTTP backend for the in-game mod: `python -m mcrag serve` (default http://127.0.0.1:8765).

Endpoints
  GET  /health                      -> status, model, corpus size, biomes with FAQs
  POST /doubt   {question, biome?, dimension?, x?, y?, z?}
                                    -> {answer, sources: [{n, title, section, url}], model, seconds}
  GET  /faq?biome=minecraft:cherry_grove
                                    -> {biome, title, url, faqs: [{q, a}]}
  GET  /faq/biomes                  -> list of biome titles that have FAQs

The game sends the player's biome, dimension and position with each /doubt so questions like
"what spawns here?" can be answered; the biome FAQs are pre-generated (`python -m mcrag faq-build`).
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .faq import BiomeFaqs
from .generate import Answerer
from .llm import OllamaError
from .recipes import RecipeBook
from .retriever import HybridRetriever

DEFAULT_MODEL = "qwen3:4b-instruct"  # fits a 4 GB GPU; qwen3:8b answers similarly (see README)
# "what spawns here?" - questions about the player's surroundings get the biome name added to the
# search query; other questions are searched as asked, so the biome doesn't add noise to them.
_HERE = re.compile(r"\b(here|this biome|around me|nearby|near me|where i am|my biome|this area)\b", re.I)

CHAT_STYLE = """\
The answer is shown in the Minecraft chat box, so keep it short: at most 4 sentences, or a list of \
up to 6 short bullet points. No preamble, no closing summary, no notes about the excerpts or the \
player context, and don't repeat yourself. Name things rather than listing statistics: leave out \
numbers such as spawn weights, group sizes or percentages unless the question asks for them."""


class DoubtRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    biome: str | None = None          # in-game id, e.g. "minecraft:cherry_grove"
    dimension: str | None = None      # e.g. "minecraft:overworld"
    x: int | None = None
    y: int | None = None
    z: int | None = None


def plain_text(text: str, truncated: bool = False) -> str:
    """Make an answer chat-ready.

    Minecraft chat has no markdown, so emphasis/headers are dropped and list markers become
    bullets. Spawn-table statistics copied from the wiki ("(spawn weight 1/5 ...; group size 4)")
    are removed - the 4B model copies them even when told not to, and they swamp the chat box.
    A truncated answer is cut back to its last complete line.
    """
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    text = re.sub(r"\s*\((?=[^)]*\b(?:spawn weight|group size)\b)[^)]*\)?", "", text, flags=re.I)
    if truncated and "\n" in text.strip():
        text = text.strip().rsplit("\n", 1)[0] + "\n…"
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def create_app(index_dir: Path = Path("index"), model: str = DEFAULT_MODEL,
               tutorial_cap: int | None = 2, cpu_only: bool = False,
               search_threads: int = 4) -> FastAPI:
    """cpu_only: run the model entirely on the CPU (slower, but never touches the GPU).
    search_threads: CPU threads for the embedding/reranking models, leaving the rest to the game."""
    import torch
    torch.set_num_threads(search_threads)
    app = FastAPI(title="Minecraft RAG", version="0.1")
    retriever = HybridRetriever(index_dir, tutorial_cap=tutorial_cap)
    # Measured on a 4 GB laptop GPU while playing: keeping the model loaded held 2.2 GB of VRAM for
    # 30 minutes after every question (game lag); unloading right after each answer frees it within
    # seconds at no speed cost (~10-13 s per answer). A 4096-token context fits the 8 passages.
    answerer = Answerer(retriever, model=model, style=CHAT_STYLE, num_predict=500, num_ctx=4096,
                        keep_alive="0", num_gpu=0 if cpu_only else None)
    faqs = BiomeFaqs()
    recipes = RecipeBook()  # exact recipes from the game data (`python -m mcrag recipes-build`)
    lock = threading.Lock()  # retriever and a single local GPU: one question at a time

    def biome_name(biome: str | None) -> str | None:
        if not biome:
            return None
        return faqs.title_for(biome) or biome.split(":", 1)[-1].replace("_", " ").title()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model, "device": "cpu" if cpu_only else "gpu",
                "chunks": len(retriever.chunks), "biomes_with_faqs": len(faqs.faqs),
                "items_with_recipes": len(recipes)}

    @app.post("/doubt")
    def doubt(req: DoubtRequest):
        name = biome_name(req.biome)
        # Location is only given to the model for questions about the player's surroundings:
        # the 4B model otherwise applies it to unrelated questions ("diamonds can't be found in
        # the Plains at y=64"), which a prompt instruction alone didn't prevent.
        about_here = bool(name and _HERE.search(req.question))
        context, query = None, req.question
        if about_here:
            dim = (req.dimension or "").split(":", 1)[-1].replace("_", " ") or "unknown dimension"
            context = f"the player is in the {name} biome ({dim})"
            if req.y is not None:
                context += f" at height y={req.y}"
            query = f"{req.question} {name}"
        t0 = time.time()
        # Recipe questions are answered straight from the game's recipe data, without the model:
        # the wiki's recipe grids are images (its text has no pattern or counts), and even with the
        # exact recipe in its context the model garbled rows while copying them. This is exact,
        # instant, and leaves the GPU to the game.
        exact = recipes.passages(req.question)
        if exact:
            return {
                "answer": "\n\n".join(p["text"] for p in exact),
                "sources": [{"n": n, "title": p["title"], "section": p["section"], "url": p["url"]}
                            for n, p in enumerate(exact, 1)],
                "model": "game recipe data", "biome": name,
                "seconds": round(time.time() - t0, 1), "truncated": False,
            }
        with lock:
            try:
                hits = answerer.retrieve(query)
                ans = answerer.ask(req.question, hits=hits, context=context)
            except OllamaError as e:
                raise HTTPException(503, f"Ollama is not available: {e}") from None
        return {
            "answer": plain_text(ans.text, truncated=ans.stop_reason == "max_tokens"),
            "sources": [{"n": n, "title": ans.hits[i].chunk["title"],
                         "section": ans.hits[i].chunk["section"], "url": ans.hits[i].chunk["url"]}
                        for n, i in enumerate(ans.cited, 1)],
            "model": ans.model, "biome": name, "seconds": round(time.time() - t0, 1),
            "truncated": ans.stop_reason == "max_tokens",
        }

    @app.get("/faq")
    def faq(biome: str = Query(..., description="in-game id or name, e.g. minecraft:cherry_grove")):
        entry = faqs.get(biome)
        if not entry or not entry["faqs"]:
            raise HTTPException(404, f"No FAQs for biome '{biome}' (run `python -m mcrag faq-build`).")
        return {"biome": biome, "title": entry["title"], "url": entry["url"], "faqs": entry["faqs"]}

    @app.get("/faq/biomes")
    def faq_biomes():
        return sorted(t for t, e in faqs.faqs.items() if e["faqs"])

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, model: str = DEFAULT_MODEL,
          index_dir: Path = Path("index"), cpu_only: bool = False) -> None:
    import uvicorn
    # 127.0.0.1 only: the prototype is for this machine; expose it deliberately when hosting.
    uvicorn.run(create_app(index_dir, model, cpu_only=cpu_only), host=host, port=port,
                log_level="info")
