param(
    [int]$Port = 8800,
    [int]$ContainerPort = 8000,
    [int]$Context = 8192,
    [string]$Image = "vllm/vllm-openai:gemma4-cu130",
    [string]$ContainerName = "tq-vllm-fp8-8k"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$bench = Join-Path $root "bench_embeddings_turbo"
$python = Join-Path $root "envs\tq-bench\python.exe"
$runLogs = Join-Path $bench "run_logs"
$hfCache = Join-Path $root "hf_cache"
$fp8Index = "I:\dev\Legal\case_kb\bench_embeddings\indexes\qwen3-embedding-8b-fp8-vllm.lance"
$modelId = "qwen3-embedding-8b-fp8-vllm-docker-8k"
$indexModelId = "qwen3-embedding-8b-fp8-vllm"
$hfRepo = "maywell/Qwen3-Embedding-8B-FP8-Dynamic"

if (!(Test-Path $python)) { throw "Python env not found at $python" }
if (!(Test-Path $fp8Index)) { throw "FP8 vLLM index not found at $fp8Index" }
New-Item -ItemType Directory -Force -Path $runLogs | Out-Null
New-Item -ItemType Directory -Force -Path $hfCache | Out-Null

function Stop-PortProcess {
    param([int]$LocalPort)
    $connections = Get-NetTCPConnection -LocalPort $LocalPort -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        if ($conn.OwningProcess -and $conn.OwningProcess -ne 0) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

function Stop-Container {
    param([string]$Name)
    $existing = docker ps -a -q -f "name=^$Name$"
    if ($existing) {
        docker rm -f $Name | Out-Null
    }
}

function Wait-VLLMReady {
    param([int]$LocalPort, [int]$TimeoutSeconds = 900)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/v1/models" -TimeoutSec 5 | Out-Null
            return
        } catch {
            Start-Sleep -Seconds 5
        }
    } while ((Get-Date) -lt $deadline)
    throw "vLLM did not become ready on port $LocalPort within $TimeoutSeconds seconds"
}

function Start-VramMonitor {
    param([string]$Path)
    $cmd = "while (`$true) { nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits; Start-Sleep -Seconds 1 }"
    return Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $cmd) -RedirectStandardOutput $Path -WindowStyle Hidden -PassThru
}

function Save-DockerLogs {
    param([string]$Name, [string]$Path)
    $existing = docker ps -a -q -f "name=^$Name$"
    if ($existing) {
        cmd /c "docker logs $Name > `"$Path`" 2>&1"
    }
}

$serverLog = Join-Path $runLogs "$modelId.server.docker.log"
$evalOut = Join-Path $runLogs "$modelId.on-fp8-vllm-8k.eval.out.log"
$evalErr = Join-Path $runLogs "$modelId.on-fp8-vllm-8k.eval.err.log"
$vramCsv = Join-Path $runLogs "$modelId.on-fp8-vllm-8k.eval.vram.csv"

Write-Host "=== 8K FP8-index cross-eval: query=$modelId index=$indexModelId ==="
Stop-Container -Name $ContainerName
Stop-PortProcess -LocalPort $Port

$mountCache = "${hfCache}:/root/.cache/huggingface"
$dockerArgs = @(
    "run", "-d",
    "--name", $ContainerName,
    "--gpus", "all",
    "--ipc", "host",
    "-p", "${Port}:${ContainerPort}",
    "-v", $mountCache,
    "-e", "HF_HUB_ENABLE_HF_TRANSFER=0",
    "-e", "VLLM_USAGE_SOURCE=production-docker-image",
    $Image,
    $hfRepo,
    "--runner", "pooling",
    "--convert", "embed",
    "--max-model-len", "$Context",
    "--served-model-name", $hfRepo,
    "--host", "0.0.0.0",
    "--port", "$ContainerPort",
    "--trust-remote-code",
    "--gpu-memory-utilization", "0.90"
)

docker @dockerArgs | Out-File -FilePath $serverLog -Encoding utf8
$monitor = $null
try {
    Write-Host "Waiting for vLLM server on http://127.0.0.1:$Port ..."
    Wait-VLLMReady -LocalPort $Port
    Save-DockerLogs -Name $ContainerName -Path $serverLog

    $monitor = Start-VramMonitor -Path $vramCsv
    $pyArgs = @(
        "-m", "runners.eval_quality",
        "--model", $modelId,
        "--index-model", $indexModelId,
        "--index-path", $fp8Index
    )
    $p = Start-Process -FilePath $python -ArgumentList $pyArgs -WorkingDirectory $bench -RedirectStandardOutput $evalOut -RedirectStandardError $evalErr -Wait -PassThru -WindowStyle Hidden
    if (($p.ExitCode -ne 0) -and ($null -ne $p.ExitCode)) {
        throw "vLLM cross-eval failed, see $evalErr"
    }
} finally {
    if ($monitor -and !$monitor.HasExited) {
        Stop-Process -Id $monitor.Id -Force -ErrorAction SilentlyContinue
    }
    Save-DockerLogs -Name $ContainerName -Path $serverLog
    Stop-Container -Name $ContainerName
}

Push-Location $bench
& $python -m runners.build_fp8_cross_query_report
Pop-Location
