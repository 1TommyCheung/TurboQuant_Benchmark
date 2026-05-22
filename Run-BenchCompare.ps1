param(
    [string[]]$Models = @("qwen3-embedding-8b-q8-ollama", "qwen3-embedding-8b-q8-turbo"),
    [int]$BatchSize = 64,
    [int]$Limit = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$bench = Join-Path $root "bench_embeddings_turbo"
$python = Join-Path $root "envs\tq-bench\python.exe"

if (!(Test-Path $python)) {
    throw "Python env not found at $python"
}
if (!(Test-Path $bench)) {
    throw "Benchmark copy not found at $bench"
}

Set-Location $bench

foreach ($model in $Models) {
    Write-Host "=== Embedding corpus: $model ==="
    $embedArgs = @("-m", "runners.embed_corpus", "--model", $model, "--batch-size", "$BatchSize")
    if ($Limit -gt 0) {
        $embedArgs += @("--limit", "$Limit")
    }
    if ($Overwrite) {
        $embedArgs += "--overwrite"
    }
    & $python @embedArgs

    Write-Host "=== Evaluating quality: $model ==="
    & $python -m runners.eval_quality --model $model
}

Write-Host "Done. Outputs:"
Write-Host "  indexes: $root\bench_embeddings_turbo\indexes"
Write-Host "  reports: $root\bench_embeddings_turbo\reports\raw"
