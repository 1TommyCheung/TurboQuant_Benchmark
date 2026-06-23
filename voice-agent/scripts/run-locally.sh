#!/usr/bin/env bash
# One-command boot for the Pattern C voice agent.
# After save: chmod +x scripts/run-locally.sh
#
# - Pre-flight: nvidia-smi shows both GPUs (4090 + 3090 Ti)
# - Pull the beellama image
# - docker compose up -d
# - Wait for /health on beellama (8083) and audio-service (8090)
# - Print connection info
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BEELLAMA_IMAGE="ghcr.io/1tommycheung/beellama-server:stable"
BEELLAMA_URL="http://localhost:8083/v1/models"
AUDIO_URL="http://localhost:8090/health"
WS_URL="ws://localhost:8090/ws"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"

log()  { printf '\033[1;36m[run]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

# ---- Pre-flight: GPUs --------------------------------------------------------
log "Pre-flight: checking GPUs via nvidia-smi"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found on PATH. Is the NVIDIA driver installed in WSL2?"
    exit 1
fi

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_COUNT=${#GPU_NAMES[@]}
log "Detected ${GPU_COUNT} GPU(s):"
for i in "${!GPU_NAMES[@]}"; do
    printf '       [%d] %s\n' "$i" "${GPU_NAMES[$i]}"
done

if [ "$GPU_COUNT" -lt 2 ]; then
    err "Pattern C requires 2 GPUs (4090 for LLM, 3090 Ti for audio). Found ${GPU_COUNT}."
    exit 1
fi

# Soft sanity check that the expected models are present
if ! printf '%s\n' "${GPU_NAMES[@]}" | grep -qi '4090'; then
    warn "No RTX 4090 detected — beellama may not hit its 160ms TTFT target."
fi
if ! printf '%s\n' "${GPU_NAMES[@]}" | grep -qiE '3090( ti)?'; then
    warn "No RTX 3090 Ti detected — audio stack may share the LLM GPU and stutter."
fi

# ---- Pull image --------------------------------------------------------------
log "Pulling ${BEELLAMA_IMAGE}"
docker pull "$BEELLAMA_IMAGE"

# ---- Compose up --------------------------------------------------------------
log "docker compose up -d"
docker compose up -d

# ---- Wait for health ---------------------------------------------------------
wait_for_url() {
    local name="$1" url="$2" deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
    log "Waiting on ${name} @ ${url} (timeout ${WAIT_TIMEOUT}s)"
    while : ; do
        if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
            log "${name} is healthy"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            err "${name} did not become healthy within ${WAIT_TIMEOUT}s"
            docker compose logs --tail=80 || true
            return 1
        fi
        sleep 2
    done
}

wait_for_url "beellama"      "$BEELLAMA_URL"
wait_for_url "audio-service" "$AUDIO_URL"

# ---- Connection info ---------------------------------------------------------
cat <<EOF

==============================================================================
  Voice agent up.

  beellama LLM (4090)          : http://localhost:8083/v1
  audio-service health (3090Ti): ${AUDIO_URL}
  audio-service WebSocket      : ${WS_URL}

  Smoke test:    scripts/smoke-test.sh
  Benchmarks:    scripts/bench-run.sh
  Tail logs:     docker compose logs -f
  Stop:          docker compose down
==============================================================================
EOF
