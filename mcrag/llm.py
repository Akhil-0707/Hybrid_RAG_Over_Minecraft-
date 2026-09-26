"""Minimal Ollama chat client (POST /api/chat), used for open-weight generation and judging.

Works the same against a local Ollama or one started inside a Kaggle notebook; the host comes
from OLLAMA_HOST (default http://127.0.0.1:11434 - the IPv4 address, so Windows doesn't try IPv6
"localhost" first).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterator

import requests

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = "http://" + DEFAULT_HOST


class OllamaError(RuntimeError):
    pass


class OllamaBusy(OllamaError):
    """Server unreachable or 5xx: worth retrying with backoff."""


@dataclass
class ChatResult:
    text: str
    model: str
    done_reason: str          # "stop", or "length" when num_predict was hit
    input_tokens: int
    output_tokens: int
    seconds: float            # server-side total_duration


class Ollama:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 600):
        self.host, self.timeout = host.rstrip("/"), timeout

    def chat(self, model: str, messages: list[dict], *, num_ctx: int = 8192,
             num_predict: int = 1024, temperature: float = 0.2, seed: int | None = 0,
             think: bool | None = None, fmt: dict | None = None,
             keep_alive: str = "30m", num_gpu: int | None = None) -> ChatResult:
        """think=None uses the model's default; False disables thinking on models that have it.

        keep_alive: how long Ollama keeps the model in (video) memory after the call; "0" unloads
        it immediately, freeing the GPU for a game. num_gpu: layers to put on the GPU (0 = CPU
        only); None lets Ollama decide.
        """
        # num_ctx must be set explicitly: Ollama's default context is short and silently
        # truncates the retrieved passages from the front of the prompt.
        options = self._options(num_ctx, num_predict, temperature, seed, num_gpu)
        body = {"model": model, "messages": messages, "stream": False, "options": options,
                "keep_alive": keep_alive}
        if think is not None:
            body["think"] = think
        if fmt is not None:
            body["format"] = fmt
        r = self._post(body)
        if r.status_code == 400 and think is False and "think" in r.text.lower():
            # Models without a thinking mode reject the flag; "no thinking" is their default anyway.
            body.pop("think")
            r = self._post(body)
        if r.status_code >= 500:
            raise OllamaBusy(f"HTTP {r.status_code}: {r.text[:500]}")
        if r.status_code != 200:
            raise OllamaError(f"HTTP {r.status_code}: {r.text[:500]}")
        d = r.json()
        return ChatResult(
            text=d.get("message", {}).get("content", ""),
            model=d.get("model", model),
            done_reason=d.get("done_reason", "stop"),
            input_tokens=d.get("prompt_eval_count", 0),
            output_tokens=d.get("eval_count", 0),
            seconds=d.get("total_duration", 0) / 1e9,
        )

    def chat_stream(self, model: str, messages: list[dict], *, num_ctx: int = 8192,
                    num_predict: int = 1024, temperature: float = 0.2, seed: int | None = 0,
                    keep_alive: str = "30m", num_gpu: int | None = None) -> Iterator[str | ChatResult]:
        """Like chat(), but yields the answer text piece by piece as the model writes it, then a
        final ChatResult (with an empty text) carrying the stop reason and token counts."""
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive,
                "options": self._options(num_ctx, num_predict, temperature, seed, num_gpu)}
        r = self._post(body, stream=True)
        if r.status_code != 200:
            err = OllamaBusy if r.status_code >= 500 else OllamaError
            raise err(f"HTTP {r.status_code}: {r.text[:500]}")
        with r:
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if "error" in d:
                    raise OllamaError(d["error"])
                if d.get("message", {}).get("content"):
                    yield d["message"]["content"]
                if d.get("done"):
                    yield ChatResult(text="", model=d.get("model", model),
                                     done_reason=d.get("done_reason", "stop"),
                                     input_tokens=d.get("prompt_eval_count", 0),
                                     output_tokens=d.get("eval_count", 0),
                                     seconds=d.get("total_duration", 0) / 1e9)

    def load(self, model: str, *, num_ctx: int = 8192, num_gpu: int | None = None,
             keep_alive: str = "2m") -> None:
        """Load the model without generating anything, so a question that follows skips the load.
        num_ctx/num_gpu must match the later call, or Ollama reloads the model for it."""
        body = {"model": model, "messages": [], "keep_alive": keep_alive, "stream": False,
                "options": self._options(num_ctx, 1, 0.0, None, num_gpu)}
        r = self._post(body)
        if r.status_code != 200:
            raise OllamaError(f"HTTP {r.status_code}: {r.text[:500]}")

    @staticmethod
    def _options(num_ctx, num_predict, temperature, seed, num_gpu) -> dict:
        options = {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        return options

    def _post(self, body: dict, stream: bool = False, attempts: int = 3) -> requests.Response:
        # A local connection can fail once while the machine is short on memory (seen in-game with
        # ~1 GB of RAM free: the request never reached Ollama, and asking again worked), so retry
        # a failed connect briefly before reporting Ollama as unreachable.
        for attempt in range(attempts):
            try:
                return requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout,
                                     stream=stream)
            except requests.ConnectionError as e:
                if attempt == attempts - 1:
                    raise OllamaBusy(f"cannot reach Ollama at {self.host} - "
                                     f"is `ollama serve` running?") from e
                time.sleep(1)

    def has_model(self, model: str) -> bool:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=10).json().get("models", [])
        except requests.RequestException:
            return False
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        return model in names or f"{model}:latest" in names
