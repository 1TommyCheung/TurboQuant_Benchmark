param(
    [string]$ModelPath = "I:\dev\LLM\TurboQuant_Benchmark\models\Qwen3-Embedding-8B-Q8_0.gguf",
    [int]$Port = 8080,
    [int]$Context = 32768,
    [int]$Batch = 8192,
    [int]$UBatch = 8192
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$serverDir = Join-Path $root "turboquant-plus-tqp-v0.1.1-windows-x64-cuda12.4"
$server = Join-Path $serverDir "llama-server.exe"

if (!(Test-Path $server)) {
    throw "TurboQuant llama-server.exe not found at $server"
}
if (!(Test-Path $ModelPath)) {
    throw "Model not found at $ModelPath. Run .\Download-Qwen3Q8.ps1 first."
}

Set-Location $serverDir

# Qwen's GGUF card recommends embedding mode with last-token pooling and -ub 8192.
# TurboQuant guidance says Q8_0 weights are suitable for symmetric turbo cache.
& $server `
    -m $ModelPath `
    --embedding `
    --pooling last `
    --alias qwen3-embedding-8b-q8-turbo `
    -ngl all `
    -c $Context `
    -b $Batch `
    -ub $UBatch `
    -fa on `
    --cache-type-k turbo3 `
    --cache-type-v turbo3 `
    -np 1 `
    --metrics `
    --host 0.0.0.0 `
    --port $Port
