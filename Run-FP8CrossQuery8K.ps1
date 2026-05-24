param(
    [int]$Port = 8080,
    [int]$Context = 8192,
    [int]$Batch = 8192,
    [int]$UBatch = 8192
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$bench = Join-Path $root "bench_embeddings_turbo"
$python = Join-Path $root "envs\tq-bench\python.exe"
$modelPath = Join-Path $root "models\Qwen3-Embedding-8B-Q8_0.gguf"
$serverDir = Join-Path $root "turboquant-plus-tqp-v0.1.1-windows-x64-cuda12.4"
$server = Join-Path $serverDir "llama-server.exe"
$runLogs = Join-Path $bench "run_logs"
$fp8Index = "I:\dev\Legal\case_kb\bench_embeddings\indexes\qwen3-embedding-8b-fp8-vllm.lance"

if (!(Test-Path $python)) { throw "Python env not found at $python" }
if (!(Test-Path $modelPath)) { throw "Model not found at $modelPath" }
if (!(Test-Path $server)) { throw "TurboQuant server not found at $server" }
if (!(Test-Path $fp8Index)) { throw "FP8 vLLM index not found at $fp8Index" }
New-Item -ItemType Directory -Force -Path $runLogs | Out-Null

$configs = @(
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-turbo3";    Alias = "qwen3-embedding-8b-q8-tq-turbo3";    CacheK = "turbo3"; CacheV = "turbo3" },
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-turbo4";    Alias = "qwen3-embedding-8b-q8-tq-turbo4";    CacheK = "turbo4"; CacheV = "turbo4" },
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-q8-turbo4"; Alias = "qwen3-embedding-8b-q8-tq-q8-turbo4"; CacheK = "q8_0";   CacheV = "turbo4" },
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-q8-q8";     Alias = "qwen3-embedding-8b-q8-tq-q8-q8";     CacheK = "q8_0";   CacheV = "q8_0" }
)

function Stop-PortProcess {
    param([int]$LocalPort)
    $connections = Get-NetTCPConnection -LocalPort $LocalPort -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        if ($conn.OwningProcess -and $conn.OwningProcess -ne 0) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

function Wait-ServerReady {
    param([int]$LocalPort, [int]$TimeoutSeconds = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/v1/models" -TimeoutSec 5 | Out-Null
            return
        } catch {
            Start-Sleep -Seconds 2
        }
    } while ((Get-Date) -lt $deadline)
    throw "llama-server did not become ready on port $LocalPort within $TimeoutSeconds seconds"
}

foreach ($cfg in $configs) {
    Write-Host "=== 8K FP8-index cross-eval: query=$($cfg.Id) index=qwen3-embedding-8b-fp8-vllm ==="
    Stop-PortProcess -LocalPort $Port
    Start-Sleep -Seconds 2
    $serverOut = Join-Path $runLogs "$($cfg.Id).fp8-cross-8k.server.out.log"
    $serverErr = Join-Path $runLogs "$($cfg.Id).fp8-cross-8k.server.err.log"
    $args = @(
        "-m", $modelPath,
        "--embedding",
        "--pooling", "last",
        "--alias", $cfg.Alias,
        "-ngl", "all",
        "-c", "$Context",
        "-b", "$Batch",
        "-ub", "$UBatch",
        "-fa", "on",
        "--cache-type-k", $cfg.CacheK,
        "--cache-type-v", $cfg.CacheV,
        "-np", "1",
        "--metrics",
        "--host", "0.0.0.0",
        "--port", "$Port"
    )
    $proc = Start-Process -FilePath $server -WorkingDirectory $serverDir -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr
    try {
        Wait-ServerReady -LocalPort $Port
        $out = Join-Path $runLogs "$($cfg.Id).on-fp8-vllm-8k.eval.out.log"
        $err = Join-Path $runLogs "$($cfg.Id).on-fp8-vllm-8k.eval.err.log"
        $pyArgs = @(
            "-m", "runners.eval_quality",
            "--model", $cfg.Id,
            "--index-model", "qwen3-embedding-8b-fp8-vllm",
            "--index-path", $fp8Index
        )
        $p = Start-Process -FilePath $python -ArgumentList $pyArgs -WorkingDirectory $bench -RedirectStandardOutput $out -RedirectStandardError $err -Wait -PassThru -WindowStyle Hidden
        if (($p.ExitCode -ne 0) -and ($null -ne $p.ExitCode)) {
            throw "cross-eval failed for $($cfg.Id), see $err"
        }
    } finally {
        if (!$proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
        Stop-PortProcess -LocalPort $Port
    }
}

Push-Location $bench
& $python -m runners.build_fp8_cross_query_report
Pop-Location
