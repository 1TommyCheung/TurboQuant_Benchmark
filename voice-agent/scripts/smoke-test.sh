#!/usr/bin/env bash
# Smoke-test the audio-service WS endpoint after `docker compose up`.
# After save: chmod +x scripts/smoke-test.sh
#
# - Sends bench/voice/audio/fixtures/short_hi.wav via wscat to ws://localhost:8090/ws
# - Asserts that at least one binary audio frame comes back within 10s.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WS_URL="${WS_URL:-ws://localhost:8090/ws}"
FIXTURE="${FIXTURE:-$PROJECT_DIR/bench/voice/audio/fixtures/short_hi.wav}"
TIMEOUT_SECS="${TIMEOUT_SECS:-10}"

log()  { printf '\033[1;36m[smoke]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[smoke]\033[0m %s\n' "$*" >&2; }

# ---- Dependencies ------------------------------------------------------------
if ! command -v wscat >/dev/null 2>&1; then
    err "wscat not found. Install with: npm install -g wscat"
    exit 127
fi

if [ ! -f "$FIXTURE" ]; then
    err "Fixture WAV not found: $FIXTURE"
    err "Generate it first: scripts/record-test-audio.py"
    exit 1
fi

# ---- Send the WAV, capture the reply ----------------------------------------
OUT_FILE="$(mktemp -t voice-smoke.XXXXXX.bin)"
trap 'rm -f "$OUT_FILE"' EXIT

log "POST ${FIXTURE} -> ${WS_URL} (timeout ${TIMEOUT_SECS}s)"

# wscat can read a file payload via --file and writes binary frames to stdout.
# We give the server up to $TIMEOUT_SECS to produce any response bytes.
set +e
timeout "${TIMEOUT_SECS}s" wscat \
    --connect "$WS_URL" \
    --file "$FIXTURE" \
    --no-color \
    > "$OUT_FILE" 2>/dev/null
RC=$?
set -e

BYTES=$(wc -c < "$OUT_FILE" | tr -d ' ')
log "received ${BYTES} bytes from server (wscat rc=${RC})"

if [ "$BYTES" -lt 64 ]; then
    err "FAIL: no/insufficient audio frames returned (expected >= 64 bytes)"
    exit 1
fi

log "PASS: audio-service responded with audio data."
