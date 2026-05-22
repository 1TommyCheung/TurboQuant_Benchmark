# Phase A — Static retrieval diff (snapshot-aligned)

Replays each `search_evidence` call from the pi-session HTML through both
Gemini and Harrier backends, reading from the frozen snapshot defined in
`bench_embeddings/src/bench/snapshot.py`. Computes overlap metrics. Runs a
Sonnet 4-bucket judge on all 11 turns. Verdict: A1 / A2 / A3.

**Spec:** `../../docs/superpowers/specs/2026-05-16-agent-level-evaluation-design.md`

**Plan:** `../../docs/superpowers/plans/2026-05-16-agent-level-evaluation-phase-a.md`

**Snapshot:** `../SNAPSHOT.md` (ID `2026-05-16_1fe458f`, anchored at case_kb commit `1fe458fa6`)

## Run order

```bash
cd /mnt/i/dev/Legal/case_kb/bench_embeddings
conda activate evidence-lake

# 1. Augment bench LanceDB tables with chunks the agent has referenced
python -m replay.runners.augment_corpus

# 2. Extract search_evidence calls from the pi-session HTML
python -m replay.runners.extract_session_calls

# 3. Compute cache_keys.json (SHAs over snapshot + bench code)
python -m replay.runners.update_cache_keys

# 4. Run Gemini baseline re-eval on the augmented corpus (one-time)
python -m replay.runners.rerun_gemini_baseline

# 5. Static retrieval diff Gemini vs Harrier across all search_evidence calls
python -m replay.runners.replay_search_static

# 6. Sonnet judge on all 11 turns
python -m replay.runners.judge_all_turns

# 7. Apply §5.4 decision rule → verdict A1/A2/A3
python -m replay.runners.apply_phase_a_verdict

# 8. Rebuild report (extends v1 HTML report with §16)
python -m runners.build_report
```

## Cost / wall time

- ~$5 LLM (Sonnet via Claude OAuth) for step 6
- ~5-6 hours dev + ~45 min compute
