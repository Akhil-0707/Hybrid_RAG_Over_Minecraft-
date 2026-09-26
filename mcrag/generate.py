"""Answer generation: retrieved chunks -> an open-weight LLM served by Ollama, with citations.

The top-k passages are numbered [1]..[k] in the prompt and the model is asked to cite them; the
markers it writes are validated (numbers pointing at no passage are dropped and counted) and
renumbered in order of first use, so the CLI can print a matching source list with wiki URLs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .llm import Ollama
from .retriever import Hit, HybridRetriever
from .text import chunk_header

MODEL = "qwen3:8b"

SYSTEM = """\
You answer Minecraft questions using excerpts from the Minecraft Wiki that are numbered [1], [2], \
... in the user message. Some excerpts are prose; others are table rows written as \
"Header: value; Header: value" or infobox facts.

Base the answer on the excerpts. If they don't contain the answer, say so plainly and share only \
what they do support, rather than filling gaps from memory - game mechanics change between \
versions, and the wiki excerpts are the source of truth here. When Java Edition and Bedrock \
Edition differ, say which is which. Keep answers direct and practical: lead with the answer, then \
the details a player needs (numbers, requirements, where to find things).

Cite the excerpts you use with their numbers in square brackets right after the sentence they \
support, like this: "Creepers have 20 HP [2]." Only cite numbers that appear in the excerpt list. \
If no excerpt answers the question, say that the excerpts don't cover it and cite nothing."""

_MARKER = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)  # 0-based indices into hits, in citation order
    stop_reason: str | None = None
    model: str | None = None
    usage: dict = field(default_factory=dict)
    invalid_citations: int = 0  # [n] markers pointing at no passage, dropped from the text


def retrieve_passages(retriever: HybridRetriever, question: str, k: int, mode: str,
                      rewriter=None) -> tuple[list[Hit], list[str]]:
    """Top-k passages; with a rewriter, rewrites also fill reserved slots (search_multi)."""
    if rewriter is None:
        return retriever.search(question, k, mode), []
    rewrites = rewriter(question)
    return retriever.search_multi(question, rewrites, k, mode), rewrites


def number_passages(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{n}] {chunk_header(h.chunk)} ({h.chunk['url']})\n{h.chunk['text']}"
                       for n, h in enumerate(hits, 1))


def renumber_citations(text: str, n_passages: int) -> tuple[str, list[int], int]:
    """Map the model's [n] (1-based passage numbers) to citation order; drop invalid markers.

    Returns (text with markers renumbered [1], [2], ... in order of first use, cited passage
    indices (0-based) in that order, number of invalid markers dropped).
    """
    cited: list[int] = []
    invalid = 0

    def repl(m: re.Match) -> str:
        nonlocal invalid
        out = []
        for part in m.group(1).split(","):
            n = int(part)
            if not 1 <= n <= n_passages:
                invalid += 1
                continue
            if n - 1 not in cited:
                cited.append(n - 1)
            out.append(f"[{cited.index(n - 1) + 1}]")
        return "".join(dict.fromkeys(out))

    return _MARKER.sub(repl, text), cited, invalid


class Answerer:
    """Retrieve passages for a question and answer it with a model served by Ollama."""

    def __init__(self, retriever: HybridRetriever, model: str = MODEL, k: int = 8,
                 mode: str = "rerank", client: Ollama | None = None, think: bool = False,
                 num_ctx: int = 8192, num_predict: int = 1024, rewriter=None,
                 style: str | None = None):
        """style: extra instructions appended to the system prompt (e.g. short chat answers for
        the in-game mod). None keeps the prompt exactly as evaluated."""
        self.retriever, self.model, self.k, self.mode = retriever, model, k, mode
        self.client = client or Ollama()
        self.rewriter, self.last_rewrites = rewriter, []
        self.think, self.num_ctx, self.num_predict = think, num_ctx, num_predict
        self.system = SYSTEM + ("\n\n" + style if style else "")

    def retrieve(self, question: str) -> list[Hit]:
        hits, self.last_rewrites = retrieve_passages(self.retriever, question, self.k, self.mode,
                                                     self.rewriter)
        return hits

    def messages(self, question: str, hits: list[Hit], context: str | None = None) -> list[dict]:
        user = f"<excerpts>\n{number_passages(hits)}\n</excerpts>\n\nQuestion: {question}"
        if context:
            # Where the player is (from the game), for questions like "what spawns here?".
            user += f"\n\nPlayer context: {context}"
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def ask(self, question: str, on_text: Callable[[str], None] | None = None,
            hits: list[Hit] | None = None, context: str | None = None) -> Answer:
        hits = hits if hits is not None else self.retrieve(question)
        r = self.client.chat(self.model, self.messages(question, hits, context), think=self.think,
                             num_ctx=self.num_ctx, num_predict=self.num_predict)
        text, cited, invalid = renumber_citations(r.text.strip(), len(hits))
        stop = "max_tokens" if r.done_reason == "length" else "end_turn"
        if on_text:
            on_text(text)
        return Answer(
            text=text, hits=hits, cited=cited, stop_reason=stop, model=r.model,
            usage={"input_tokens": r.input_tokens, "output_tokens": r.output_tokens},
            invalid_citations=invalid,
        )
