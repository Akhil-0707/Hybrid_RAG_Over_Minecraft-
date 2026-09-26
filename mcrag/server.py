"""Local HTTP backend for the in-game mod: `python -m mcrag serve` (default http://127.0.0.1:8765).

Endpoints
  GET  /health                      -> status, model, corpus size, biomes with FAQs
  POST /doubt/stream {question, biome?, dimension?, x?, y?, z?}
                                    -> NDJSON, one event per line as the answer is written:
                                       {"line": "..."} ... then {"done": true, sources, model,
                                       seconds, truncated}, or {"error": "..."}
  POST /doubt   (same body)         -> {answer, sources: [{n, title, section, url}], model, seconds}
  POST /warmup                      -> starts loading the model (the mod calls it while the
                                       player is still typing a question)
  GET  /faq?biome=minecraft:cherry_grove
                                    -> {biome, title, url, faqs: [{q, a}]}
  GET  /faq/biomes                  -> list of biome titles that have FAQs

The game sends the player's biome, dimension and position with each question so ones like
"what spawns here?" can be answered; the biome FAQs are pre-generated (`python -m mcrag faq-build`).
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .faq import BiomeFaqs
from .generate import Answerer, renumber_citations
from .llm import ChatResult, OllamaError
from .recipes import RecipeBook
from .retriever import HybridRetriever
from .spawns import SpawnBook

DEFAULT_MODEL = "qwen3:4b-instruct"  # fits a 4 GB GPU; qwen3:8b answers similarly (see README)
# "what spawns here?" - questions about the player's surroundings get the biome name added to the
# search query; other questions are searched as asked, so the biome doesn't add noise to them.
_HERE = re.compile(r"\b(here|this biome|around me|nearby|near me|where i am|my biome|this area)\b", re.I)
# Where a streamed answer can be cut into chat lines: a line break, or the end of a sentence (with
# its citation markers) once the next sentence has started. "1." list numbers are not sentence ends.
_BREAK = re.compile(r"\n+|(?<!\d)[.!?](?:\s*\[\d{1,2}(?:\s*,\s*\d{1,2})*\])*(?P<gap>[ \t]+)(?=[A-Z(•*\-\"])")

CHAT_STYLE = """\
The answer is shown in the Minecraft chat box, so keep it short: at most 4 sentences, or a list of \
up to 6 short bullet points. No preamble, no closing summary, no notes about the excerpts or the \
player context, and don't repeat yourself. Name things rather than listing statistics: leave out \
numbers such as spawn weights, group sizes or percentages unless the question asks for them. \
The excerpts are only part of the wiki: never say that something doesn't spawn, exist or happen \
just because the excerpts don't mention it."""


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


def split_complete(buffer: str) -> tuple[list[str], str]:
    """Cut the finished lines/sentences off the front of a partly written answer."""
    pieces, start = [], 0
    for m in _BREAK.finditer(buffer):
        end = m.start("gap") if m.group("gap") else m.start()
        pieces.append(buffer[start:end])
        start = m.end()
    return pieces, buffer[start:]


def create_app(index_dir: Path = Path("index"), model: str = DEFAULT_MODEL,
               tutorial_cap: int | None = 2, cpu_only: bool = False,
               search_threads: int = 4, search_mode: str = "fast",
               keep_alive: str = "0") -> FastAPI:
    """cpu_only: run the model entirely on the CPU (slower, but never touches the GPU).
    search_threads: CPU threads for the embedding/reranking models, leaving the rest to the game.
    search_mode: "fast" (no cross-encoder, ~0.2 s) or "rerank" (cross-encoder, ~7 s on a laptop
    CPU; on the fact-coverage eval "fast" scored as well).
    keep_alive: how long the model stays loaded after an answer ("0" frees the GPU right away;
    "5m" makes follow-up questions skip the ~7 s load but holds ~2 GB of video memory)."""
    import torch
    torch.set_num_threads(search_threads)
    app = FastAPI(title="Minecraft RAG", version="0.2")
    retriever = HybridRetriever(index_dir, tutorial_cap=tutorial_cap)
    retriever.search("warm up", 8, search_mode)  # load the search models now, not on the 1st question
    # Measured on a 4 GB laptop GPU while playing: keeping the model loaded held 2.2 GB of VRAM for
    # 30 minutes after every question (game lag); unloading right after each answer frees it within
    # seconds. A 4096-token context fits the 8 passages.
    num_gpu = 0 if cpu_only else None
    answerer = Answerer(retriever, model=model, mode=search_mode, style=CHAT_STYLE,
                        num_predict=350, num_ctx=4096, keep_alive=keep_alive, num_gpu=num_gpu)
    faqs = BiomeFaqs()
    recipes = RecipeBook()  # exact recipes from the game data (`python -m mcrag recipes-build`)
    spawns = SpawnBook()  # biome spawn lists from the game data (`python -m mcrag spawns-build`)
    lock = threading.Lock()  # retriever and a single local GPU: one question at a time
    warming: list[threading.Thread] = []

    def biome_name(biome: str | None) -> str | None:
        if not biome:
            return None
        return faqs.title_for(biome) or biome.split(":", 1)[-1].replace("_", " ").title()

    def event(**fields) -> str:
        return json.dumps(fields, ensure_ascii=False) + "\n"

    def answer_events(req: DoubtRequest) -> Iterator[str]:
        t0 = time.time()
        name = biome_name(req.biome)
        # Recipe questions are answered straight from the game's recipe data, without the model:
        # the wiki's recipe grids are images (its text has no pattern or counts), and even with
        # the exact recipe in its context the model garbled rows while copying them.
        exact = recipes.passages(req.question)
        if exact:
            for p in exact:
                for line in p["text"].split("\n"):
                    yield event(line=line)
            yield event(done=True, model="game recipe data", biome=name, truncated=False,
                        seconds=round(time.time() - t0, 1),
                        sources=[{"n": n, "title": p["title"], "section": p["section"],
                                  "url": p["url"]} for n, p in enumerate(exact, 1)])
            return
        # Spawn questions likewise come from the game's biome spawn lists: the wiki's spawn tables
        # lose their category labels in parsing, and the model mixed up editions and categories.
        here = bool(req.biome and _HERE.search(req.question))
        spawn = spawns.answer(req.question, req.biome if here else None)
        if spawn:
            for line in spawn["lines"]:
                yield event(line=line)
            title = faqs.title_for(spawn["biome"]) or spawn["name"]
            yield event(done=True, model="game spawn data", biome=name, truncated=False,
                        seconds=round(time.time() - t0, 1),
                        sources=[{"n": 1, "title": title, "section": "Mobs (game data)",
                                  "url": "https://minecraft.wiki/w/" + title.replace(" ", "_")}])
            return
        # Location is only given to the model for questions about the player's surroundings:
        # the 4B model otherwise applies it to unrelated questions ("diamonds can't be found in
        # the Plains at y=64"), which a prompt instruction alone didn't prevent.
        context, query = None, req.question
        if name and _HERE.search(req.question):
            dim = (req.dimension or "").split(":", 1)[-1].replace("_", " ") or "unknown dimension"
            context = f"the player is in the {name} biome ({dim})"
            if req.y is not None:
                context += f" at height y={req.y}"
            query = f"{req.question} {name}"
        with lock:
            try:
                hits = answerer.retrieve(query)
                buffer, cited, final = "", [], None
                pieces = answerer.client.chat_stream(
                    model, answerer.messages(req.question, hits, context),
                    num_ctx=answerer.num_ctx, num_predict=answerer.num_predict,
                    keep_alive=keep_alive, num_gpu=num_gpu)
                for piece in pieces:
                    if isinstance(piece, ChatResult):
                        final = piece
                        break
                    done, buffer = split_complete(buffer + piece)
                    for line in done:
                        line = plain_text(renumber_citations(line, len(hits), cited)[0])
                        if line:
                            yield event(line=line)
            except OllamaError as e:
                yield event(error=f"Ollama is not available: {e}")
                return
        truncated = final is not None and final.done_reason == "length"
        # A cut-off answer ends mid-sentence: show "…" instead of the unfinished sentence.
        last = "…" if truncated else plain_text(renumber_citations(buffer, len(hits), cited)[0])
        if last:
            yield event(line=last)
        yield event(done=True, model=final.model if final else model, biome=name,
                    truncated=truncated, seconds=round(time.time() - t0, 1),
                    sources=[{"n": n, "title": hits[i].chunk["title"],
                              "section": hits[i].chunk["section"], "url": hits[i].chunk["url"]}
                             for n, i in enumerate(cited, 1)])

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model, "device": "cpu" if cpu_only else "gpu",
                "search": search_mode, "keep_alive": keep_alive,
                "chunks": len(retriever.chunks), "biomes_with_faqs": len(faqs.faqs),
                "items_with_recipes": len(recipes), "biomes_with_spawns": len(spawns)}

    @app.post("/doubt/stream")
    def doubt_stream(req: DoubtRequest):
        return StreamingResponse(answer_events(req), media_type="application/x-ndjson")

    @app.post("/doubt")
    def doubt(req: DoubtRequest):
        lines, done = [], {}
        for e in map(json.loads, answer_events(req)):
            if "error" in e:
                raise HTTPException(503, e["error"])
            if "line" in e:
                lines.append(e["line"])
            else:
                done = e
        done.pop("done", None)
        return {"answer": "\n".join(lines), **done}

    @app.post("/warmup")
    def warmup():
        # Loading takes ~7 s; the mod calls this as soon as the player starts typing a question,
        # so the load overlaps the typing. The model unloads again 2 minutes later if no question
        # comes, or right after the answer when one does.
        if warming and warming[-1].is_alive():
            return {"status": "loading"}

        def load():
            try:
                answerer.client.load(model, num_ctx=answerer.num_ctx, num_gpu=num_gpu, keep_alive="2m")
            except OllamaError:
                pass  # the question itself will report it

        warming[:] = [threading.Thread(target=load, daemon=True)]
        warming[0].start()
        return {"status": "loading"}

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
          index_dir: Path = Path("index"), cpu_only: bool = False, search_mode: str = "fast",
          keep_alive: str = "0") -> None:
    import uvicorn
    # Check the port before the ~40 s of model loading, so a backend that is still running from
    # before gives a clear message instead of uvicorn's bind error at the end.
    import socket
    with socket.socket() as s:
        try:
            s.bind((host, port))
        except OSError:
            raise SystemExit(f"Port {port} on {host} is already in use - probably another "
                             f"`python -m mcrag serve` is still running. Stop it with Ctrl+C in its "
                             f"window (or close that window), or pick another --port.") from None
    # 127.0.0.1 only: the prototype is for this machine; expose it deliberately when hosting.
    uvicorn.run(create_app(index_dir, model, cpu_only=cpu_only, search_mode=search_mode,
                           keep_alive=keep_alive), host=host, port=port, log_level="info")
