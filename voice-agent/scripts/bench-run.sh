#!/usr/bin/env bash
# Run the voice-agent pytest suite and emit results.json.
# After save: chmod +x scripts/bench-run.sh
#
# Usage: scripts/bench-run.sh [pytest-args...]
#   RESULTS_JSON=path/to/out.json scripts/bench-run.sh -k T02
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BENCH_DIR="$PROJECT_DIR/bench/voice"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/bench/voice/results}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RESULTS_JSON="${RESULTS_JSON:-$RESULTS_DIR/results-${TIMESTAMP}.json}"
LATEST_JSON="$RESULTS_DIR/results-latest.json"
JUNIT_XML="$RESULTS_DIR/junit-${TIMESTAMP}.xml"

mkdir -p "$RESULTS_DIR"

log() { printf '\033[1;36m[bench]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[bench]\033[0m %s\n' "$*" >&2; }

if ! command -v pytest >/dev/null 2>&1; then
    err "pytest not found. Install bench/voice/requirements.txt first."
    exit 127
fi

log "running pytest in ${BENCH_DIR}"
log "results -> ${RESULTS_JSON}"

export VOICE_RESULTS_JSON="$RESULTS_JSON"

set +e
pytest "$BENCH_DIR" \
    -v \
    --tb=short \
    --junitxml="$JUNIT_XML" \
    "$@"
PYTEST_RC=$?
set -e

# If conftest didn't already write results.json (e.g. tests imported `report`),
# fall back to deriving one from the JUnit XML so callers always get JSON.
if [ ! -f "$RESULTS_JSON" ] && [ -f "$JUNIT_XML" ] && command -v python3 >/dev/null 2>&1; then
    log "no results.json from harness — synthesizing from JUnit XML"
    python3 - "$JUNIT_XML" "$RESULTS_JSON" <<'PY'
import json, sys, xml.etree.ElementTree as ET, datetime as dt
junit, out = sys.argv[1], sys.argv[2]
root = ET.parse(junit).getroot()
suites = root.findall(".//testsuite") or [root]
cases = []
for s in suites:
    for tc in s.findall("testcase"):
        status = "passed"
        if tc.find("failure") is not None:
            status = "failed"
        elif tc.find("error") is not None:
            status = "error"
        elif tc.find("skipped") is not None:
            status = "skipped"
        cases.append({
            "id": tc.attrib.get("name"),
            "classname": tc.attrib.get("classname"),
            "duration_s": float(tc.attrib.get("time", "0") or 0),
            "status": status,
        })
data = {
    "run_id": dt.datetime.utcnow().isoformat() + "Z",
    "source": "junit-fallback",
    "topology": "split_4090_llm_3090ti_audio",
    "cases": cases,
    "summary": {
        "total": len(cases),
        "passed": sum(c["status"] == "passed" for c in cases),
        "failed": sum(c["status"] in ("failed", "error") for c in cases),
        "skipped": sum(c["status"] == "skipped" for c in cases),
    },
}
with open(out, "w") as fh:
    json.dump(data, fh, indent=2)
PY
fi

if [ -f "$RESULTS_JSON" ]; then
    cp -f "$RESULTS_JSON" "$LATEST_JSON"
    log "results written: $RESULTS_JSON"
    log "latest symlink:  $LATEST_JSON"
else
    err "no results.json produced (pytest rc=${PYTEST_RC})"
fi

exit "$PYTEST_RC"
