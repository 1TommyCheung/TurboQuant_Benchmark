"""Environment-driven configuration for the audio service.

All settings can be overridden via environment variables of the same name
(case-insensitive). See ``docs/voice-agent/2026-06-23-pattern-c-design.md``
for the broader Pattern C architecture.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Pattern C voice agent audio service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM (OpenAI-compatible endpoint exposed by beellama)
    LLM_BASE_URL: str = "http://beellama:8083/v1"
    LLM_MODEL: str = "qwen3.5-9b"
    LLM_SYSTEM_PROMPT: str = (
        "You are a helpful voice assistant. Always begin your replies with "
        "a brief acknowledgement like 'Sure,' or 'Right,' followed by a comma. "
        "Keep responses conversational and under 80 words unless asked for detail."
    )

    # Audio
    SAMPLE_RATE_IN: int = 16000   # Whisper/Silero expect 16kHz
    SAMPLE_RATE_OUT: int = 24000  # Kokoro outputs 24kHz
    CHANNELS: int = 1

    # VAD (Silero)
    VAD_THRESHOLD: float = 0.45
    VAD_MIN_SILENCE_MS: int = 450
    VAD_MIN_SPEECH_MS: int = 100
    VAD_SPEECH_PAD_MS: int = 100
    VAD_WINDOW_SAMPLES: int = 512  # 32ms at 16kHz

    # STT (faster-whisper)
    STT_MODEL: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    STT_DEVICE: str = "cuda"
    STT_COMPUTE_TYPE: str = "int8"
    STT_LANGUAGE: str = "en"
    STT_RECYCLE_EVERY: int = 500  # subprocess recycle

    # TTS (Kokoro)
    TTS_LANG_CODE: str = "a"  # American English
    TTS_VOICE: str = "af_heart"
    TTS_SPEED: float = 1.0

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8090

    # CUDA
    CUDA_VISIBLE_DEVICES: str = "1"  # 3090 Ti for audio stack


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""

    return Settings()
