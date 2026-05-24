param(
    [int]$Limit = 5000,
    [int]$BatchSize = 64,
    [int]$Context = 8192,
    [int]$Batch = 8192,
    [int]$UBatch = 8192,
    [int]$Port = 8080,
    [switch]$SkipOllama,
    [switch]$ReuseExistingIndexes
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$bench = Join-Path $root "bench_embeddings_turbo"
$python = Join-Path $root "envs\tq-bench\python.exe"
$modelPath = Join-Path $root "models\Qwen3-Embedding-8B-Q8_0.gguf"
$serverDir = Join-Path $root "turboquant-plus-tqp-v0.1.1-windows-x64-cuda12.4"
$server = Join-Path $serverDir "llama-server.exe"
$runLogs = Join-Path $bench "run_logs"

if (!(Test-Path $python)) { throw "Python env not found at $python" }
if (!(Test-Path $modelPath)) { throw "Model not found at $modelPath" }
if (!(Test-Path $server)) { throw "TurboQuant server not found at $server" }
New-Item -ItemType Directory -Force -Path $runLogs | Out-Null

$turboConfigs = @(
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-turbo3";     Alias = "qwen3-embedding-8b-q8-tq-turbo3";     CacheK = "turbo3"; CacheV = "turbo3" },
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-turbo4";     Alias = "qwen3-embedding-8b-q8-tq-turbo4";     CacheK = "turbo4"; CacheV = "turbo4" },
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-q8-turbo4";  Alias = "qwen3-embedding-8b-q8-tq-q8-turbo4";  CacheK = "q8_0";   CacheV = "turbo4" },
    [pscustomobject]@{ Id = "qwen3-embedding-8b-q8-tq-q8-q8";      Alias = "qwen3-embedding-8b-q8-tq-q8-q8";      CacheK = "q8_0";   CacheV = "q8_0" }
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

function Start-VramMonitor {
    param([string]$OutPath)
    Start-Job -ScriptBlock {
        param($Path)
        "timestamp,memory_used_mib" | Set-Content -Path $Path -Encoding utf8
        while ($true) {
            $line = nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits
            Add-Content -Path $Path -Value $line -Encoding utf8
            Start-Sleep -Seconds 1
        }
    } -ArgumentList $OutPath
}

function Stop-VramMonitor {
    param($Job)
    if ($Job) {
        Stop-Job $Job -ErrorAction SilentlyContinue
        Receive-Job $Job -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $Job -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-BenchStep {
    param(
        [string]$ModelId,
        [string]$Step,
        [string[]]$PyArgs
    )
    $logPath = Join-Path $runLogs "$ModelId.$Step.matrix.out.log"
    $errPath = Join-Path $runLogs "$ModelId.$Step.matrix.err.log"
    $vramPath = Join-Path $runLogs "$ModelId.$Step.matrix.vram.csv"
    $monitor = Start-VramMonitor -OutPath $vramPath
    try {
        $proc = Start-Process -FilePath $python -ArgumentList $PyArgs -WorkingDirectory $bench -RedirectStandardOutput $logPath -RedirectStandardError $errPath -Wait -PassThru -WindowStyle Hidden
        $exitCode = $proc.ExitCode
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
        if ($exitCode -ne 0) {
            throw "$Step failed for $ModelId with exit code $exitCode. See $logPath and $errPath"
        }
    } finally {
        Stop-VramMonitor -Job $monitor
    }
}

function Run-BenchmarkForModel {
    param([string]$ModelId)
    Write-Host "=== Embedding corpus: $ModelId ==="
    $indexPath = Join-Path (Join-Path $bench "indexes") "$ModelId.lance"
    if ($ReuseExistingIndexes -and (Test-Path $indexPath)) {
        Write-Host "Reusing existing index at $indexPath"
    } else {
        $embedArgs = @("-m", "runners.embed_corpus", "--model", $ModelId, "--batch-size", "$BatchSize", "--limit", "$Limit", "--overwrite")
        Invoke-BenchStep -ModelId $ModelId -Step "embed" -PyArgs $embedArgs
    }

    Write-Host "=== Evaluating quality/query speed: $ModelId ==="
    $evalArgs = @("-m", "runners.eval_quality", "--model", $ModelId)
    Invoke-BenchStep -ModelId $ModelId -Step "eval" -PyArgs $evalArgs
}

function Run-CrossEval {
    param(
        [string]$QueryModelId,
        [string]$IndexModelId
    )
    Write-Host "=== Cross-eval: query=$QueryModelId index=$IndexModelId ==="
    $indexPath = Join-Path (Join-Path $bench "indexes") "$IndexModelId.lance"
    if (!(Test-Path $indexPath)) {
        Write-Host "Skipping cross-eval; index not found at $indexPath"
        return
    }
    $step = "eval-on-$IndexModelId"
    $evalArgs = @("-m", "runners.eval_quality", "--model", $QueryModelId, "--index-model", $IndexModelId)
    Invoke-BenchStep -ModelId $QueryModelId -Step $step -PyArgs $evalArgs
}

if (!$SkipOllama) {
    Write-Host "=== Ollama baseline ==="
    Run-BenchmarkForModel -ModelId "qwen3-embedding-8b-q8-ollama"
}

foreach ($cfg in $turboConfigs) {
    Write-Host "=== TurboQuant server: $($cfg.Id) K=$($cfg.CacheK) V=$($cfg.CacheV) ==="
    Stop-PortProcess -LocalPort $Port
    Start-Sleep -Seconds 2

    $serverOut = Join-Path $runLogs "$($cfg.Id).server.matrix.out.log"
    $serverErr = Join-Path $runLogs "$($cfg.Id).server.matrix.err.log"
    $serverArgs = @(
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
    $proc = Start-Process -FilePath $server -WorkingDirectory $serverDir -ArgumentList $serverArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr
    try {
        Wait-ServerReady -LocalPort $Port
        Run-BenchmarkForModel -ModelId $cfg.Id
        if (!$SkipOllama) {
            Run-CrossEval -QueryModelId $cfg.Id -IndexModelId "qwen3-embedding-8b-q8-ollama"
        }
    } finally {
        if (!$proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        Stop-PortProcess -LocalPort $Port
        Start-Sleep -Seconds 2
    }
}

Push-Location $bench
& $python -m runners.build_turboquant_matrix_report
Pop-Location
