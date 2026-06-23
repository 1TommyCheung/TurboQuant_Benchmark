# Voice benchmark audio fixtures

These WAV files are **not** checked into git. Each test case in
`bench/voice/test_*.py` marks itself with `@pytest.mark.wav_fixture("name.wav")`
and is auto-skipped if the matching file is missing (see `conftest.py`).

## Required files

All fixtures are **16 kHz, mono, 16-bit PCM**.

| File | Duration | Spoken content | Used by |
|---|---|---|---|
| `short_hi.wav`                       | ~0.4s | "hi"                                  | `T01_short_greeting` |
| `medium_time.wav`                    | ~1.2s | "what time is it"                     | `T02_medium_factual`, concurrency, chaos |
| `long_explain.wav`                   | ~3.0s | "explain how neural networks work"    | `T03_long_explanation` |
| `barge_in_pair/utterance_a.wav`      | ~2.0s | any longer utterance ("tell me a long story about a dog") | `T04_barge_in` (the interrupted) |
| `barge_in_pair/utterance_b.wav`      | ~0.6s | a short interrupting phrase ("wait, stop") | `T04_barge_in` (the interrupter) |

## How to record

### Option 1 — record live (recommended)

```bash
# Linux / WSL2 with sox installed
sox -d -r 16000 -c 1 -b 16 short_hi.wav
# Speak, then Ctrl+C
```

### Option 2 — synthesize from an existing voice

If you don't want to record yourself, generate the prompts via Kokoro (already
in the audio-service stack) at 24 kHz, then resample to 16 kHz:

```bash
python - <<'PY'
import soundfile as sf
from kokoro import KPipeline
pipe = KPipeline(lang_code='a')
for text, name in [
    ("hi", "short_hi.wav"),
    ("what time is it", "medium_time.wav"),
    ("explain how neural networks work", "long_explain.wav"),
]:
    audio_chunks = [c for c, _ in pipe(text, voice="af_heart")]
    audio = sum((list(c) for c in audio_chunks), [])
    sf.write(name, audio, 24000)
PY

# resample to 16 kHz mono PCM16
for f in short_hi medium_time long_explain; do
  sox "${f}.wav" -r 16000 -c 1 -b 16 "${f}.16k.wav" && mv "${f}.16k.wav" "${f}.wav"
done
```

### Option 3 — use Common Voice clips

Pull short matching utterances from the Mozilla Common Voice corpus and
resample/trim to the durations above with `sox` or `ffmpeg`.

## Verifying

```bash
python -c "import soundfile as sf; d, sr = sf.read('short_hi.wav'); print(sr, len(d)/sr)"
# expect: 16000 ~0.4
```

The test harness will raise `ValueError: expected 16000 Hz, got <X>` if the
sample rate is wrong.
