"""Build per-(scenario × backend × replicate) prompts for Phase B+ complex scenarios.

Writes one prompt file per cell under /tmp/phase_b_scenarios/<scenario>__<backend>__rep<N>.prompt.txt
"""
from __future__ import annotations
import os
from pathlib import Path

from bench_phase_b.complex_scenarios import ALL_SCENARIOS, Scenario


CASE_CONTEXT = """## Case Context (Singapore FC/OAD 22/2025: Tommy Cheung v Tracy Cheuk)
- **Plaintiff:** Tommy Cheung. Currently represented by Gloria James-Civetta & Co (GJC). Resides at 10 Shanghai Road #06-02, Charleston, Singapore 248184 (since 1 Sep 2025).
- **Defendant:** Tracy Cheuk ("ex_wife"). Represented by Lee & Lee (L&L). Resides at 20 Kay Poh Road #06-05, Kasturina Lodge, Singapore 248969.
- **Children:** Taran (in Alexandra Primary), Tristan (kindergarten).
- **Matrimonial home:** 31 Alexandra Road #06-04, Singapore 159967 (the asset, not Tommy's residence).
- **GJC personnel:** Sheryl Keith, Pang Chen, Noelle Goh, Yuqi Wu.
- **HEP (prior firm):** Carrie Gill et al. Represented Tommy June–November 2024.
- **Today's date:** 2026-05-16.
"""


def build_prompt(scenario: Scenario, backend: str, replicate: int) -> str:
    turns_md = "\n".join(
        f"### Turn {t['idx']}{' (needs search)' if t['needs_search'] else ' (no search expected)'}\n"
        f"User: {t['user_text']}"
        for t in scenario["turns"]
    )
    schema_backend = backend
    schema_scenario = scenario["id"]
    return f"""You are **Athena**, a legal evidence research assistant for Singapore Family Court case FC/OAD 22/2025.

This is **Phase B+ scenario `{scenario['id']}`** — a designed complex multi-turn benchmark probing real agent workflows. The user is actively working a contested point; play through each turn as Athena would.

## Replay parameters
- **Backend (retrieval):** `{backend}`
- **Scenario:** {scenario['title']}
- **Skill probed:** {scenario['skill']}
- **Replicate:** {replicate}

{CASE_CONTEXT}
## Scenario description
{scenario['description']}

## Search tool (Bash-driven shim)
You do NOT have a `search_evidence` function. Run searches via Bash:

```bash
cd /mnt/i/dev/Legal/case_kb/bench_embeddings && \\
conda run -n evidence-lake --no-capture-output python -m bench_phase_b.bench_search \\
  --backend {backend} \\
  --query "<query>" \\
  --k 20 \\
  --mode hybrid
```

Output is JSON: `{{backend, mode, query, k, n_hits, hits:[{{chunk_id, source_type, party_from, date_sgt, is_privileged, in_scope, snippet}}]}}`.

You may also use `--mode vector` (vector-only, no BM25) or `--mode bm25` (BM25 only), and adjust `--k` (typical 10-30). The `is_privileged` and `is_wp` flags on each hit are reliable — use them when the user asks for privilege-aware filtering.

## The conversation to play

Play each turn in order with full conversation context carrying forward. For turns marked "needs search", do real searches with the shim. For acknowledgement turns, give a brief Athena-style response.

{turns_md}

## What to do
1. Play Turn 1, deliver a final Athena response.
2. Then Turn 2, with full conversation history. And so on through the last turn.
3. After each turn, record what searches you ran and what you cited.
4. Be honest. If retrieval misses something the scenario is testing, say so in `notes`. Don't fabricate.
5. Cap each Athena response at ~300 words.

## CRITICAL: structured output

When you finish all turns, end your entire response with **exactly one** fenced JSON block (no text after the closing fence):

```json
{{
  "scenario": "{schema_scenario}",
  "backend": "{schema_backend}",
  "replicate": {replicate},
  "skill": "{scenario['skill']}",
  "turns": [
    {{
      "turn_idx": 1,
      "user_text": "<verbatim>",
      "searches": [
        {{"query": "<q>", "mode": "hybrid", "k": 20,
          "top_chunk_ids": ["<id1>", "<id2>"]}}
      ],
      "final_response_text": "<your Athena response, <=300 words>",
      "cited_chunk_ids": ["<ids you cited>"],
      "notes": "<honest 1-line observation>"
    }}
  ]
}}
```

The `turns` array must contain every turn in the scenario, in order.

Begin Turn 1 now.
"""


def main() -> None:
    out_dir = Path("/tmp/phase_b_scenarios")
    out_dir.mkdir(exist_ok=True)
    backends = ["gemini-embedding-001", "gemini-embedding-2", "qwen3-embedding-8b-fp8-vllm"]
    for scenario in ALL_SCENARIOS:
        for backend in backends:
            for rep in (1,):  # N=1 first pass
                p = build_prompt(scenario, backend, rep)
                short_backend = backend.replace("gemini-embedding-001", "v1").replace("gemini-embedding-2", "v2").replace("qwen3-embedding-8b-fp8-vllm", "qwen")
                fname = f"{scenario['id']}__{short_backend}__rep{rep}.prompt.txt"
                (out_dir / fname).write_text(p)
                print(f"{fname}: {len(p)} chars")
    print(f"\nWrote {4*3*1} prompts to {out_dir}")


if __name__ == "__main__":
    main()
