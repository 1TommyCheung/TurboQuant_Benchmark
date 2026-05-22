# bench_embeddings

Embedding model bake-off for Evidence Lake. Decides whether to replace
`gemini-embedding-001` with a local model on the RTX 4090.

> ### Relationship to case_kb
>
> This is a **child project of [case_kb](../)** that benchmarks the embedding
> choice driving **Stage 6 (EMBED)** of the production pipeline. It is
> strictly read-only on production data — reads `/home/tommy/evidence_lake_indexes/`
> and `chunks.parquet` from the case_kb data lake; writes only to its own
> `indexes/` and `reports/`. **No code here ships to production**; a winning
> model is applied by updating `case_kb/config/embeddings.yaml:active_provider`.
>
> Full dependency map and integration path: [`CLAUDE.md`](CLAUDE.md).
> Production pipeline reference: [`../docs/PIPELINE.md`](../docs/PIPELINE.md) — see Stage 4 EMBED.

**Design spec:** [../docs/superpowers/specs/2026-05-15-embeddings-benchmark-design.md](../docs/superpowers/specs/2026-05-15-embeddings-benchmark-design.md)

## Quick start

```bash
conda activate evidence-lake
pip install -e ".[serve,test]"
pytest tests/
python -m runners.run_all  # full pipeline
```

## Layout

- `src/bench/` — pure-function library (sampling, perturbations, metrics, scoring)
- `runners/` — per-stage CLI scripts
- `config/models.yaml` — 6 model lineup
- `data/` — eval queries + chunk samples (gitignored except `layer3_handcrafted.jsonl`)
- `indexes/` — per-model LanceDB tables (gitignored)
- `reports/` — HTML report + decision MD

## Isolation

This package does NOT modify `tools/`, `services/`, `eval/`, or `agent/`.
It reads production LanceDB at `/home/tommy/evidence_lake_indexes/`
read-only. All writes go to `bench_embeddings/indexes/` and `bench_embeddings/reports/`.
