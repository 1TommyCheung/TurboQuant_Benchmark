"""Quick quality check — 10 complex queries, compare responses across configs.

Sends the same 10 prompts to the currently running server, saves responses.
Run once per config, then diff the outputs.

Usage:
    python -m tqbench.benchmarks.generation.runners.eval_quick_quality --model qwen3.5-9b-fp8-vllm
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import httpx

from tqbench.config import get_server
from tqbench.benchmarks.generation.models import get_candidate

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "raw"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

QUERIES = [
    {"id": "math_1", "q": "What is 127 × 43? Show your work.", "expect_contains": "5461"},
    {"id": "math_2", "q": "A train travels 240km in 3 hours. It then travels 180km in 2 hours. What is its average speed for the entire journey?", "expect_contains": "84"},
    {"id": "code_1", "q": "Write a Python function that checks if a string is a valid palindrome, ignoring spaces and punctuation. Include type hints.", "expect_contains": "def"},
    {"id": "code_2", "q": "Write a Python function to find the longest common subsequence of two strings. Return the length.", "expect_contains": "def"},
    {"id": "reason_1", "q": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.", "expect_contains": "cannot"},
    {"id": "reason_2", "q": "There are 5 houses in a row. The red house is to the left of the green house. The blue house is in the middle. The yellow house is at one end. The white house is next to the blue house. What position is the red house in?", "expect_contains": ""},
    {"id": "knowledge_1", "q": "Explain the difference between TCP and UDP in networking. Give one use case for each.", "expect_contains": ""},
    {"id": "knowledge_2", "q": "What causes tides on Earth? Be specific about the roles of the Moon and Sun.", "expect_contains": "gravitational"},
    {"id": "creative_1", "q": "Write a haiku about a programmer debugging code at 3am.", "expect_contains": ""},
    {"id": "extract_1", "q": "Extract all the numbers from this text and sum them: 'The 3 cats ate 12 fish and drank 2 bowls of milk over 7 days.'", "expect_contains": "24"},
]


def run(model_id: str) -> None:
    spec = get_candidate(model_id)
    server = get_server(spec.server)
    host = server["host"]
    client = httpx.Client(base_url=host, timeout=120)

    r = client.get("/v1/models")
    if r.status_code != 200:
        raise RuntimeError(f"Server not reachable at {host}")

    results = []
    for q in QUERIES:
        t0 = time.perf_counter()
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": spec.model_name,
                "messages": [{"role": "user", "content": q["q"]}],
                "max_tokens": 512,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["completion_tokens"]
        passed = q["expect_contains"].lower() in text.lower() if q["expect_contains"] else None

        results.append({
            "id": q["id"],
            "query": q["q"],
            "response": text,
            "tokens": tokens,
            "latency_s": round(elapsed, 2),
            "expect": q["expect_contains"] or "(no check)",
            "passed": passed,
        })
        status = "PASS" if passed else ("FAIL" if passed is False else "—")
        log.info(f"  {q['id']:15s}  {status:4s}  tok={tokens:4d}  lat={elapsed:.2f}s  {text[:80]}")

    passed = sum(1 for r in results if r["passed"] is True)
    checked = sum(1 for r in results if r["passed"] is not None)
    log.info(f"  Score: {passed}/{checked} passed")

    date = dt.date.today().isoformat()
    out_path = REPORTS / f"{date}_{model_id}_quick_quality.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model_id": model_id, "results": results}, indent=2))
    log.info(f"Wrote {out_path}")
    client.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
