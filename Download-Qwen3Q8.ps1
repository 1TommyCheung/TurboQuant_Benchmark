$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$models = Join-Path $root "models"
$out = Join-Path $models "Qwen3-Embedding-8B-Q8_0.gguf"

New-Item -ItemType Directory -Force -Path $models | Out-Null

if (Test-Path $out) {
    Write-Host "Model already exists: $out"
    Get-Item $out | Select-Object FullName,Length
    exit 0
}

$url = "https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q8_0.gguf"
curl.exe -L --fail --retry 3 -C - --output $out $url
Get-Item $out | Select-Object FullName,Length
