"""OpenAI-compatible generation client — one class for all servers."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field

import httpx

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import ModelSpec


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_time_s: float


@dataclass
class StreamResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    itl_ms: list[float] = field(default_factory=list)
    total_time_s: float = 0.0


def build_client(spec: ModelSpec) -> OpenAIGenerateClient:
    server = get_server(spec.server)
    return OpenAIGenerateClient(spec, server)


class OpenAIGenerateClient:
    def __init__(self, spec: ModelSpec, server: dict):
        self.model = spec.model_name
        host = server["host"]
        self.client = httpx.Client(base_url=host, timeout=300)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"Server health check failed (status {r.status_code}). "
                f"Ensure server is running on {host}."
            )

    def health(self) -> bool:
        try:
            r = self.client.get("/v1/models")
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, messages: list[dict], max_tokens: int = 256,
                 temperature: float = 0.0) -> GenerateResult:
        t0 = time.perf_counter()
        r = self.client.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GenerateResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_time_s=elapsed,
        )

    def generate_stream(self, messages: list[dict], max_tokens: int = 256,
                        temperature: float = 0.0) -> StreamResult:
        t0 = time.perf_counter()
        ttft = 0.0
        itl_ms: list[float] = []
        chunks_text: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        last_token_time = t0
        first_token_seen = False

        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content is None:
                    continue
                now = time.perf_counter()
                if not first_token_seen:
                    ttft = now - t0
                    first_token_seen = True
                else:
                    itl_ms.append((now - last_token_time) * 1000)
                last_token_time = now
                chunks_text.append(content)

        total_time = time.perf_counter() - t0
        return StreamResult(
            text="".join(chunks_text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens or len(chunks_text),
            ttft_s=ttft,
            itl_ms=itl_ms,
            total_time_s=total_time,
        )
