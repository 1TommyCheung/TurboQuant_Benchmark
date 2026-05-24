"""Model registry + sentence-transformers loader.

Reads config/models.yaml, exposes typed accessors, provides a single
load_embedder(model_id) entry-point used by Phase 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: Literal["api", "hf", "ollama", "vllm", "llamacpp"]
    dim: int
    max_ctx_tokens: int
    precision: str
    hf_repo: str | None = None
    ollama_model: str | None = None
    ollama_host: str = "http://127.0.0.1:11434"
    vllm_host: str = "http://127.0.0.1:8800"
    llamacpp_model: str | None = None
    llamacpp_host: str = "http://127.0.0.1:8080"
    provider: str | None = None
    quantization: str | None = None
    vram_estimate_gb: float | None = None
    notes: str | None = None


def load_registry() -> list[ModelSpec]:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return [ModelSpec(**c) for c in raw["candidates"]]


def get_candidate(model_id: str) -> ModelSpec:
    for c in load_registry():
        if c.id == model_id:
            return c
    raise KeyError(model_id)


def baseline_dim() -> int:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return int(raw["baseline_dim"])


def load_embedder(model_id: str):
    """Return an object with .encode(texts: list[str], batch_size=int) -> np.ndarray.

    For HF kind: returns a sentence-transformers model with appropriate
    precision/quantization. For API kind: returns a wrapper around the
    Google Generative AI client.
    """
    spec = get_candidate(model_id)
    if spec.kind == "api":
        return _GeminiEmbedder(spec)
    if spec.kind == "ollama":
        return _OllamaEmbedder(spec)
    if spec.kind == "vllm":
        return _VLLMEmbedder(spec)
    if spec.kind == "llamacpp":
        return _LlamaCppEmbedder(spec)
    return _HFEmbedder(spec)


class _GeminiEmbedder:
    def __init__(self, spec: ModelSpec):
        import google.generativeai as genai
        import os
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.spec = spec
        self.genai = genai

    def encode(self, texts: list[str], batch_size: int = 100):
        import numpy as np
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            out.extend(self._encode_batch_recursive(batch))
        return np.array(out, dtype="float32")

    def _encode_batch_recursive(self, batch: list[str], rate_attempt: int = 0) -> list:
        """Recursive batch encoder that:
          - On 429 / rate / 503: sleeps with exponential backoff and retries SAME batch.
          - On 504 / deadline: splits the batch in half and processes each piece.
            (Some V2 batches exceed Google's gRPC timeout; splitting converges.)
          - On other errors: raises.
        """
        import time
        if not batch:
            return []
        try:
            resp = self.genai.embed_content(
                model=f"models/{self.spec.id}",
                content=batch,
                task_type="retrieval_document",
            )
            return list(resp["embedding"])
        except Exception as e:
            msg = str(e).lower()
            is_rate = "429" in msg or "resource exhausted" in msg or "rate" in msg or "503" in msg
            is_deadline = "deadline" in msg or "504" in msg
            if is_rate:
                if rate_attempt >= 8:
                    raise
                time.sleep(min(2.0 * (2 ** rate_attempt), 60.0))
                return self._encode_batch_recursive(batch, rate_attempt + 1)
            if is_deadline:
                if len(batch) == 1:
                    # Can't split further — propagate
                    raise
                mid = len(batch) // 2
                return (
                    self._encode_batch_recursive(batch[:mid])
                    + self._encode_batch_recursive(batch[mid:])
                )
            raise


class _HFEmbedder:
    def __init__(self, spec: ModelSpec):
        from sentence_transformers import SentenceTransformer
        import torch
        kwargs: dict = {"device": "cuda", "trust_remote_code": True}
        if spec.precision == "int8":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["model_kwargs"] = {"quantization_config": bnb_config}
        elif spec.precision == "fp16":
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        elif spec.precision == "bf16":
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        self.model = SentenceTransformer(spec.hf_repo, **kwargs)
        # Cap effective context to 2048 to keep VRAM bounded across all
        # model sizes on the 24GB RTX 4090. 99.7% of the 50K sample is
        # <=2048 tokens; only 165 chunks (long + very_long buckets) get
        # truncated. This also matches the Gemini baseline's 2048-tok
        # ceiling, so candidates are evaluated on the same effective
        # context as the production model.
        self.model.max_seq_length = min(spec.max_ctx_tokens, 2048)
        self.spec = spec

    def encode(self, texts: list[str], batch_size: int = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


class _OllamaEmbedder:
    """Embed via Ollama's /api/embed endpoint (GGUF model under llama.cpp).

    Assumes `ollama serve` is running and the model has been pulled via
    `ollama pull <ollama_model>`.
    """
    def __init__(self, spec: ModelSpec):
        import httpx
        self.spec = spec
        self.client = httpx.Client(base_url=spec.ollama_host, timeout=300)
        # Smoke probe to confirm the daemon is up + model is pulled
        r = self.client.post("/api/embed", json={"model": spec.ollama_model, "input": "smoke"})
        if r.status_code != 200:
            raise RuntimeError(
                f"Ollama /api/embed smoke failed (status {r.status_code}): {r.text}. "
                f"Ensure `ollama serve` is running and `ollama pull {spec.ollama_model}` has completed."
            )

    def encode(self, texts: list[str], batch_size: int = 8):
        import numpy as np
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            r = self.client.post(
                "/api/embed",
                json={"model": self.spec.ollama_model, "input": batch},
            )
            r.raise_for_status()
            embs = r.json().get("embeddings") or []
            out.extend(embs)
        arr = np.array(out, dtype="float32")
        # L2-normalize to match how other embedders return normalized vectors
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class _VLLMEmbedder:
    """Embed via vLLM's OpenAI-compatible /v1/embeddings endpoint.

    Assumes `vllm serve <hf_repo> --runner pooling --convert embed --port <vllm_host port>` is running.
    Inputs are truncated to `max_input_tokens` (default 8000, safely below vllm's
    8192 max_model_len) using the model's own tokenizer.
    """
    # Hard cap below vllm's --max-model-len 8192 (leave headroom for any added tokens).
    MAX_INPUT_TOKENS = 8000

    def __init__(self, spec: ModelSpec):
        import httpx
        self.spec = spec
        # Bump connection pool to comfortably fit our 16 concurrent workers.
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self.client = httpx.Client(base_url=spec.vllm_host, timeout=180, limits=limits)
        # Smoke probe via /v1/models (vLLM responds with the served model list)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"vLLM /v1/models smoke failed (status {r.status_code}). "
                f"Ensure `vllm serve {spec.hf_repo} --runner pooling --convert embed --port {spec.vllm_host.split(':')[-1]}` is running."
            )
        # Local tokenizer is only for client-side truncation. Keep it optional so
        # the lean benchmark env can query an already-running vLLM server.
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, trust_remote_code=True)
        except ModuleNotFoundError:
            self.tokenizer = None

    def _truncate(self, text: str) -> str:
        if self.tokenizer is None:
            return text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.MAX_INPUT_TOKENS:
            return text
        ids = ids[: self.MAX_INPUT_TOKENS]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def encode(self, texts: list[str], batch_size: int = 8):
        """Send batches CONCURRENTLY so vLLM's dynamic batcher has work to coalesce.

        Sequential POSTs only give vLLM one request at a time (batch_size inputs
        per forward pass). With 16 in-flight requests, the scheduler can fuse up
        to ~128 inputs per step — closer to the FP8 model's true throughput.
        """
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        # Truncate every input up-front to stay within vllm's max_model_len.
        safe_texts = [self._truncate(t) for t in texts]

        # Slice into HTTP batches.
        slices = [safe_texts[i:i + batch_size] for i in range(0, len(safe_texts), batch_size)]

        def _embed_batch(batch: list[str]) -> list[list[float]]:
            r = self.client.post(
                "/v1/embeddings",
                json={"model": self.spec.hf_repo, "input": batch},
            )
            r.raise_for_status()
            data = r.json().get("data") or []
            return [d["embedding"] for d in data]

        # 16 concurrent workers — httpx.Client is thread-safe for requests.
        out: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for vecs in ex.map(_embed_batch, slices):
                out.extend(vecs)

        arr = np.array(out, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class _LlamaCppEmbedder:
    """Embed via llama.cpp's OpenAI-compatible /v1/embeddings endpoint.

    Used by the Windows TurboQuant build served with:
    llama-server -m Qwen3-Embedding-8B-Q8_0.gguf --embedding --pooling last -ub 8192
    """
    def __init__(self, spec: ModelSpec):
        import httpx
        self.spec = spec
        self.model_name = spec.llamacpp_model or spec.id
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self.client = httpx.Client(base_url=spec.llamacpp_host, timeout=300, limits=limits)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"llama.cpp /v1/models smoke failed (status {r.status_code}). "
                f"Start TurboQuant llama-server on {spec.llamacpp_host} with --embedding --pooling last."
            )

    def encode(self, texts: list[str], batch_size: int = 8):
        import numpy as np
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            r = self.client.post(
                "/v1/embeddings",
                json={"model": self.model_name, "input": batch},
            )
            r.raise_for_status()
            data = r.json().get("data") or []
            out.extend([d["embedding"] for d in data])
        arr = np.array(out, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
