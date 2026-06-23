#!/usr/bin/env python3
"""Generate or record T01-T05 WAV fixtures for the voice-agent benchmark.

After save: chmod +x scripts/record-test-audio.py

Modes
-----
default : synthesize each phrase via pyttsx3 (falls back to `espeak` /
          `espeak-ng` if pyttsx3 is missing) and save as 16kHz mono PCM WAV.
--mic   : prompt the user and record 3s of audio per phrase via sounddevice.

Outputs land in bench/voice/audio/fixtures/ alongside the test harness.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
RECORD_SECONDS = 3.0

# IDs match bench/voice/test_latency.py T01-T05.
TEST_PHRASES: dict[str, tuple[str, str]] = {
    "T01": ("short_hi.wav",        "hi"),
    "T02": ("medium_time.wav",     "what time is it"),
    "T03": ("long_explain.wav",    "explain how neural networks work in detail"),
}

FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent
    / "bench" / "voice" / "audio" / "fixtures"
)


def _resample_to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Linear-interp resample + mono-mix. Good enough for fixture WAVs."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * SAMPLE_RATE / sr))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _write_wav_int16(path: Path, audio_f32: np.ndarray) -> None:
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm = (audio_f32 * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- #
# TTS synth
# --------------------------------------------------------------------------- #

def _synth_pyttsx3(text: str, out_path: Path) -> bool:
    try:
        import pyttsx3  # type: ignore
    except Exception:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        engine.save_to_file(text, str(tmp_path))
        engine.runAndWait()
        with wave.open(str(tmp_path), "rb") as r:
            sr = r.getframerate()
            n = r.getnframes()
            frames = r.readframes(n)
            sw = r.getsampwidth()
            ch = r.getnchannels()
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        audio /= float(np.iinfo(dtype).max)
        if ch > 1:
            audio = audio.reshape(-1, ch).mean(axis=1)
        audio = _resample_to_16k_mono(audio, sr)
        _write_wav_int16(out_path, audio)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return True
    except Exception as exc:
        print(f"  pyttsx3 failed: {exc}", file=sys.stderr)
        return False


def _synth_espeak(text: str, out_path: Path) -> bool:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        return False
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [binary, "-s", "170", "-w", str(tmp_path), text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(str(tmp_path), "rb") as r:
            sr = r.getframerate()
            n = r.getnframes()
            frames = r.readframes(n)
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        audio = _resample_to_16k_mono(audio, sr)
        _write_wav_int16(out_path, audio)
        return True
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def synth_phrase(text: str, out_path: Path) -> None:
    if _synth_pyttsx3(text, out_path):
        return
    if _synth_espeak(text, out_path):
        return
    raise RuntimeError(
        "No TTS backend available. Install pyttsx3 (pip) or espeak-ng (apt)."
    )


# --------------------------------------------------------------------------- #
# Mic capture
# --------------------------------------------------------------------------- #

def record_phrase(text: str, out_path: Path, seconds: float = RECORD_SECONDS) -> None:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sounddevice not installed. `pip install sounddevice` for --mic mode."
        ) from exc

    print(f"\n>>> Say: {text!r}")
    for n in (3, 2, 1):
        print(f"    recording in {n}...", end="\r", flush=True)
        time.sleep(1.0)
    print("    RECORDING            ")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("    done.")
    _write_wav_int16(out_path, audio.flatten())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mic", action="store_true", help="record from microphone instead of TTS")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=FIXTURES_DIR,
        help=f"output directory (default: {FIXTURES_DIR})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing WAV fixtures",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing fixtures to {args.out_dir}")

    for tid, (fname, phrase) in TEST_PHRASES.items():
        out = args.out_dir / fname
        if out.exists() and not args.force:
            print(f"  [{tid}] {fname} exists, skipping (--force to overwrite)")
            continue
        print(f"  [{tid}] {fname}  <-  {phrase!r}")
        if args.mic:
            record_phrase(phrase, out)
        else:
            synth_phrase(phrase, out)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
