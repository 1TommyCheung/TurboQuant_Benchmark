# bench_embeddings — Embedding Model Bake-Off Harness

> **FROZEN at case_kb commit `1fe458f` (2026-05-16).** Reads data from a 5.6 GB
> snapshot at `/mnt/i/dev/Legal/data/evidence_lake/snapshots/bench_2026-05-16_1fe458f/`
> + `/home/tommy/evidence_lake_indexes/bench_snapshots/2026-05-16_1fe458f/`.
> Pipeline iteration on case_kb (re-chunking, re-embedding, new sources) does
> NOT affect this benchmark. See [SNAPSHOT.md](SNAPSHOT.md) for the manifest +
> re-snapshot procedure. All snapshot paths are defined in `src/bench/snapshot.py`.

## Purpose
Evaluate whether to replace `gemini-embedding-001` (production embedding model used by Stage 6 of the Evidence Lake pipeline) with a local model running on the RTX 4090. Decides cost/quality/latency trade-offs across 6 candidate models.

## Relationship to the parent project (case_kb)

> **This package is a child of case_kb but is strictly read-only on production data.** It exists to benchmark the embedding choice that drives Stage 6 of the pipeline.

| Direction | How it relates |
|---|---|
| Inputs | Reads production LanceDB at `/home/tommy/evidence_lake_indexes/` **read-only**. Reads `chunks.parquet` and `agent_verified_facts.jsonl` from the case_kb data lake. Reuses the same chunk schema (`ChunkRecord` in `core/schemas.py`). |
| Outputs | Writes only to `bench_embeddings/indexes/` (per-model LanceDB tables) and `bench_embeddings/reports/` (HTML + decision MD). |
| Code isolation | Does **NOT** modify `tools/`, `services/`, `eval/`, or `agent/`. Has its own `pyproject.toml` and dependency graph. |
| Cross-refs | If a model wins, the decision is applied by updating `config/embeddings.yaml:active_provider` in the case_kb project. No code in `bench_embeddings/` is meant to ship to production. |

**Design spec:** `../docs/superpowers/specs/2026-05-15-embeddings-benchmark-design.md`
**Plan:** `../docs/superpowers/plans/2026-05-15-embeddings-benchmark.md`

## Architecture
- **Input data:** **frozen snapshot** of production chunks.parquet + LanceDB + FTS + facts (paths in `src/bench/snapshot.py`) + `data/eval_queries/` (handcrafted queries + Layer 1 pool-and-judge ground truth + Layer 2 cross-LLM bias check)
- **Candidates:** 6 models (Gemini API + 5 local variants on the 4090) — see `config/models.yaml`
- **Phases:** Phase 0 smoke → Phase 1 quality (vector-only + end-to-end MRR/recall@k) → Phase 2 speed (throughput/latency/VRAM/cold-start) → Report builder
- **Orchestrator:** `runners/run_all.py` with hard 72h timebox and per-stage runner dispatch
- **Output:** HTML report (Plotly + Mermaid via Jinja2) + 1-page decision MD

## Layout
- `src/bench/` — pure-function library (sampling, perturbations, metrics, scoring)
- `runners/` — per-stage CLI scripts (phase 0-2, layer 1-2, report)
- `config/models.yaml` — 6-model lineup with API/local + dimensions
- `data/` — eval queries + chunk samples (mostly gitignored; `layer3_handcrafted.jsonl` committed)
- `indexes/` — per-model LanceDB tables (gitignored)
- `reports/` — HTML report + decision MD (gitignored)
- `tests/` — pytest suite

## CLI
```bash
conda activate evidence-lake
cd bench_embeddings && pip install -e ".[serve,test]"

# Smoke check
pytest tests/

# Full pipeline (72h timebox)
python -m runners.run_all

# Individual phases
python -m runners.phase0_smoke
python -m runners.phase1_quality
python -m runners.phase2_speed
python -m runners.report
```

## Key parent-repo files this benchmark targets
- `tools/shared/gemini_client.py` — production embedding client (the baseline being challenged)
- `tools/shared/gte_client.py`, `tools/shared/qwen3_client.py` — local candidates
- `tools/chunker/embed_build.py` — production embedding pipeline (the integration point if a new model wins)
- `config/embeddings.yaml` — production provider switch (`active_provider`)
- `core/schemas.py:ChunkRecord` — shared chunk schema

## Production integration path (if a model wins)
1. Update `config/embeddings.yaml:active_provider` to the winning provider
2. Update `tools/shared/<provider>_client.py` if needed
3. Re-run case_kb pipeline Stage 6 (`python -m scripts.refresh_pipeline --stage 6 --provider <new>`)
4. Verify hybrid search quality on production via `tools/search/hybrid_search.py`

## Caveats
- The benchmark's ground truth (`data/eval_queries/`) is derived from case_kb evidence. Adding new sources to case_kb may invalidate the eval set — re-run Layer 1 pool-and-judge if so.
- Production LanceDB path is hardcoded to `/home/tommy/evidence_lake_indexes/` (WSL). Mac users will need to symlink or repath.
