# TurboQuant Qwen3 Embedding Bench

This folder is a Windows-native copy of the `case_kb` embedding benchmark for
comparing:

- `qwen3-embedding-8b-q8-ollama`
- `qwen3-embedding-8b-q8-turbo`

The copied benchmark reads frozen case data from:

- `I:\dev\Legal\data\evidence_lake\snapshots\bench_2026-05-16_1fe458f\chunks.parquet`
- `I:\dev\Legal\data\evidence_lake\snapshots\bench_2026-05-16_1fe458f\search.duckdb`
- `I:\dev\Legal\data\evidence_lake\snapshots\bench_2026-05-16_1fe458f\agent_verified_facts.jsonl`

All generated indexes and reports stay in:

- `I:\dev\LLM\TurboQuant_Benchmark\bench_embeddings_turbo\indexes`
- `I:\dev\LLM\TurboQuant_Benchmark\bench_embeddings_turbo\reports`

## Environment

Local Miniconda:

```powershell
I:\dev\LLM\TurboQuant_Benchmark\miniconda
```

Benchmark env:

```powershell
I:\dev\LLM\TurboQuant_Benchmark\envs\tq-bench\python.exe
```

Python version:

```text
Python 3.12.13
```

## Model

Official GGUF:

```text
Qwen/Qwen3-Embedding-8B-GGUF
Qwen3-Embedding-8B-Q8_0.gguf
```

Download:

```powershell
cd I:\dev\LLM\TurboQuant_Benchmark
.\Download-Qwen3Q8.ps1
```

## TurboQuant Server

Qwen's GGUF instructions recommend:

```text
--embedding --pooling last -ub 8192
```

TurboQuant guidance says Q8_0 weights are suitable for symmetric turbo cache:

```text
--cache-type-k turbo3 --cache-type-v turbo3
```

Start:

```powershell
cd I:\dev\LLM\TurboQuant_Benchmark
.\Start-TurboQuantQwen.ps1
```

## Ollama Server

Pull the comparison model:

```powershell
ollama pull qwen3-embedding:8b-q8_0
```

Ollama must be running before the Ollama benchmark starts.

## Run Both

In another PowerShell:

```powershell
cd I:\dev\LLM\TurboQuant_Benchmark
.\Run-BenchCompare.ps1 -Overwrite
```

Run the bounded 5K comparison used for the first Windows report:

```powershell
.\Run-BenchCompare.ps1 -BatchSize 64 -Limit 5000 -Overwrite
```

Run one side only:

```powershell
.\Run-BenchCompare.ps1 -Models qwen3-embedding-8b-q8-turbo -Overwrite
.\Run-BenchCompare.ps1 -Models qwen3-embedding-8b-q8-ollama -Overwrite
```

Regenerate the HTML comparison report from the latest raw outputs:

```powershell
cd I:\dev\LLM\TurboQuant_Benchmark\bench_embeddings_turbo
I:\dev\LLM\TurboQuant_Benchmark\envs\tq-bench\python.exe -m runners.build_turboquant_compare_report
```
