"""Embedding client classes for the embeddings benchmark.

Each client takes a (spec: ModelSpec, server: dict) and exposes
.encode(texts: list[str], batch_size: int) -> np.ndarray (L2-normalized float32).

Use build_client(spec) as the single entry-point — it resolves the server
config via tqbench.config.get_server() and dispatches to the correct class.
"""
from __future__ import annotations

from tqbench.benchmarks.embeddings.models import ModelSpec


def build_client(spec: ModelSpec):
    """Return an embedder for *spec*.

    Resolves the server dict from tqbench.config.get_server(spec.server),
    then dispatches by server["type"].
    """
    from tqbench.config import get_server

    server = get_server(spec.server)
    stype = server["type"]

    if stype == "gemini_api":
        return GeminiEmbedClient(spec, server)
    if stype == "hf":
        return HFEmbedClient(spec, server)
    if stype == "ollama":
        return OllamaEmbedClient(spec, server)
    if stype == "vllm":
        return VLLMEmbedClient(spec, server)
    if stype == "llamacpp":
        return LlamaCppEmbedClient(spec, server)
    raise ValueError(f"Unknown server type '{stype}' for server '{spec.server}'")


class GeminiEmbedClient:
    """Embed via the Google Generative AI API (gemini-embedding-* models).

    Requires GEMINI_API_KEY environment variable.
    """

    def __init__(self, spec: ModelSpec, server: dict):
        import os

        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.spec = spec
        self.server = server
        self.genai = genai

    def encode(self, texts: list[str], batch_size: int = 100):
        import numpy as np

        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out.extend(self._encode_batch_recursive(batch))
        return np.array(out, dtype="float32")

    def _encode_batch_recursive(self, batch: list[str], rate_attempt: int = 0) -> list:
        """Recursive batch encoder with exponential back-off on rate limits.

        - On 429 / rate / 503: sleep with exponential backoff and retry same batch.
        - On 504 / deadline: split the batch in half and process each piece.
        - On other errors: raise.
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
            is_rate = (
                "429" in msg
                or "resource exhausted" in msg
                or "rate" in msg
                or "503" in msg
            )
            is_deadline = "deadline" in msg or "504" in msg
            if is_rate:
                if rate_attempt >= 8:
                    raise
                time.sleep(min(2.0 * (2**rate_attempt), 60.0))
                return self._encode_batch_recursive(batch, rate_attempt + 1)
            if is_deadline:
                if len(batch) == 1:
                    raise
                mid = len(batch) // 2
                return self._encode_batch_recursive(
                    batch[:mid]
                ) + self._encode_batch_recursive(batch[mid:])
            raise


class HFEmbedClient:
    """Embed via sentence-transformers (local HF model on CUDA)."""

    def __init__(self, spec: ModelSpec, server: dict):
        from sentence_transformers import SentenceTransformer
        import torch

        self.spec = spec
        self.server = server
        device = server.get("device", "cuda")
        kwargs: dict = {"device": device, "trust_remote_code": True}
        if spec.precision == "int8":
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["model_kwargs"] = {"quantization_config": bnb_config}
        elif spec.precision == "fp16":
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        elif spec.precision == "bf16":
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        self.model = SentenceTransformer(spec.hf_repo, **kwargs)
        # Cap effective context to 2048 to keep VRAM bounded on the 24GB RTX 4090.
        self.model.max_seq_length = min(spec.max_ctx_tokens, 2048)

    def encode(self, texts: list[str], batch_size: int = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


class OllamaEmbedClient:
    """Embed via Ollama's /api/embed endpoint (GGUF model under llama.cpp).

    Requires `ollama serve` running and the model pulled via `ollama pull`.
    """

    def __init__(self, spec: ModelSpec, server: dict):
        import httpx

        self.spec = spec
        self.server = server
        self.ollama_model = spec.ollama_model
        host = server["host"]
        self.client = httpx.Client(base_url=host, timeout=300)
        # Smoke probe to confirm daemon is up and model is available.
        r = self.client.post(
            "/api/embed", json={"model": self.ollama_model, "input": "smoke"}
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"Ollama /api/embed smoke failed (status {r.status_code}): {r.text}. "
                f"Ensure `ollama serve` is running and `ollama pull {self.ollama_model}` has completed."
            )

    def encode(self, texts: list[str], batch_size: int = 8):
        import numpy as np

        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = self.client.post(
                "/api/embed",
                json={"model": self.ollama_model, "input": batch},
            )
            r.raise_for_status()
            embs = r.json().get("embeddings") or []
            out.extend(embs)
        arr = np.array(out, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class VLLMEmbedClient:
    """Embed via vLLM's OpenAI-compatible /v1/embeddings endpoint.

    Assumes `vllm serve <hf_repo> --runner pooling --convert embed` is running.
    Inputs are truncated to MAX_INPUT_TOKENS using the model's own tokenizer.
    Uses 16 concurrent workers so vLLM's dynamic batcher has work to coalesce.
    """

    MAX_INPUT_TOKENS = 8000

    def __init__(self, spec: ModelSpec, server: dict):
        import httpx

        self.spec = spec
        self.server = server
        host = server["host"]
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self.client = httpx.Client(base_url=host, timeout=180, limits=limits)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"vLLM /v1/models smoke failed (status {r.status_code}). "
                f"Ensure `vllm serve {spec.hf_repo} --runner pooling --convert embed` is running on {host}."
            )
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                spec.hf_repo, trust_remote_code=True
            )
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
        """Send batches concurrently so vLLM's dynamic batcher has work to coalesce."""
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        safe_texts = [self._truncate(t) for t in texts]
        slices = [
            safe_texts[i : i + batch_size] for i in range(0, len(safe_texts), batch_size)
        ]

        def _embed_batch(batch: list[str]) -> list[list[float]]:
            r = self.client.post(
                "/v1/embeddings",
                json={"model": self.spec.hf_repo, "input": batch},
            )
            r.raise_for_status()
            data = r.json().get("data") or []
            return [d["embedding"] for d in data]

        out: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for vecs in ex.map(_embed_batch, slices):
                out.extend(vecs)

        arr = np.array(out, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class LlamaCppEmbedClient:
    """Embed via llama.cpp's OpenAI-compatible /v1/embeddings endpoint.

    Used by the TurboQuant build served with:
    llama-server -m <model>.gguf --embedding --pooling last -ub 8192
    """

    def __init__(self, spec: ModelSpec, server: dict):
        import httpx

        self.spec = spec
        self.server = server
        self.model_name = spec.llamacpp_model or spec.id
        host = server["host"]
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self.client = httpx.Client(base_url=host, timeout=300, limits=limits)
        r = self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(
                f"llama.cpp /v1/models smoke failed (status {r.status_code}). "
                f"Start llama-server on {host} with --embedding --pooling last."
            )

    def encode(self, texts: list[str], batch_size: int = 8):
        import numpy as np

        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
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
