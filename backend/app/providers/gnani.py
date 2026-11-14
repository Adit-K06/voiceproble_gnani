"""GnaniProvider — real SDK wrapper with timing, retry, and budget awareness.

IMPORTANT: This module is imported lazily. It is NEVER imported in tests —
tests always use FakeProvider. The real API is only called when the user
runs run_demo.py or the pipeline with --provider gnani after explicit approval.

Retry policy
------------
- 3 attempts, exponential back-off: 1 s, 2 s (jittered ±10 %).
- Retried errors: HTTP 429 (rate-limit) and 5xx (server fault).
- NOT retried: 401/403 (auth), 400 (bad request), InvalidAudioError.

Logging
-------
Every call is logged at INFO level with model, voice/language, and latency.
Errors are logged at ERROR level before re-raising.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from app.providers.base import BaseProvider, SynthResult, TranscribeResult

logger = logging.getLogger(__name__)

# HTTP status codes that are worth retrying.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.0
_JITTER = 0.10   # ±10 %


def _backoff(attempt: int) -> float:
    """Return sleep duration for *attempt* (0-indexed) with jitter."""
    base = _BACKOFF_BASE_S * (2 ** attempt)
    jitter = base * _JITTER * (2 * random.random() - 1)
    return max(0.0, base + jitter)


def _is_retryable(exc: Exception) -> bool:
    """Return True only if *exc* is a temporary network/rate-limit error."""
    # Import here so the module can be imported without gnani installed.
    try:
        from gnani.stt.exceptions import APIError as STTAPIError
        from gnani.tts.exceptions import APIError as TTSAPIError
    except ImportError:
        return False

    for api_err_cls in (STTAPIError, TTSAPIError):
        if isinstance(exc, api_err_cls):
            status = getattr(exc, "status_code", None)
            if status is not None:
                return status in _RETRYABLE_STATUSES
    return False


class GnaniProvider(BaseProvider):
    """Live Gnani TTS + STT provider.

    Parameters
    ----------
    api_key:
        Your Gnani API key. Falls back to the ``GNANI_API_KEY`` environment
        variable (loaded from ``backend/.env`` by the caller).
    tts_audio_config:
        :class:`gnani.tts.AudioConfig` instance. Defaults to 24 kHz WAV PCM.
    """

    def __init__(
        self,
        api_key: str | None = None,
        tts_audio_config: Any = None,
        *,
        max_attempts: int = _MAX_ATTEMPTS,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self._max_attempts = max_attempts

        # Resolve the key — never log or print it.
        self._api_key = api_key or os.environ.get("GNANI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "GNANI_API_KEY is not set. "
                "Add it to backend/.env and load it before constructing GnaniProvider."
            )

        # Lazy-import the SDK so this module can still be linted without it.
        from gnani.tts import AudioConfig, GnaniTTSClient
        from gnani.stt import GnaniSTTClient

        self._tts_client = GnaniTTSClient(api_key=self._api_key)
        self._stt_client = GnaniSTTClient(api_key=self._api_key)

        self._audio_config = tts_audio_config or AudioConfig(
            sample_rate=24000,
            encoding="linear_pcm",
            container="wav",
        )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: str,
        model: str,
        language: str,
    ) -> SynthResult:
        """Call Gnani TTS with retry on temporary errors."""
        logger.info(
            "[TTS] synthesize | model=%s voice=%s language=%s text_len=%d",
            model, voice, language, len(text),
        )
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            t0 = time.monotonic()
            try:
                audio_bytes: bytes = self._tts_client.synthesize(
                    text,
                    voice=voice,
                    model=model,
                    language=language,
                    audio_config=self._audio_config,
                )
                latency = time.monotonic() - t0
                logger.info(
                    "[TTS] OK | model=%s voice=%s latency=%.2fs bytes=%d",
                    model, voice, latency, len(audio_bytes),
                )
                return SynthResult(
                    audio_bytes=audio_bytes,
                    model=model,
                    voice=voice,
                    latency_s=latency,
                )
            except Exception as exc:
                latency = time.monotonic() - t0
                last_exc = exc
                if _is_retryable(exc) and attempt < self._max_attempts - 1:
                    sleep = _backoff(attempt)
                    logger.warning(
                        "[TTS] attempt %d/%d failed (retryable) | sleeping %.1fs | %s",
                        attempt + 1, self._max_attempts, sleep, exc,
                    )
                    time.sleep(sleep)
                else:
                    logger.error(
                        "[TTS] FAILED | model=%s voice=%s attempt=%d | %s",
                        model, voice, attempt + 1, exc,
                    )
                    raise
        # Should never reach here, but satisfies type-checkers.
        raise last_exc  # type: ignore[misc]

    def transcribe(
        self,
        wav_path: Path,
        language: str,
        model: str | None = None,
        **options: Any,
    ) -> TranscribeResult:
        """Call Gnani STT with retry on temporary errors.

        Accepted **options** keys (all optional):
            format        : "verbatim" | "transcribe" (default "verbatim")
            keywords      : list[str]  (prisma-v3.0 only)
            topic         : str        (prisma-v3.0 only)
            previous_text : str        (prisma-v3.0 only)
            speaker_names : list[str]  (prisma-v3.0 only)
        """
        fmt = options.pop("format", "verbatim")
        logger.info(
            "[STT] transcribe | model=%s language=%s format=%s file=%s",
            model, language, fmt, wav_path.name,
        )
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            t0 = time.monotonic()
            try:
                raw: dict = self._stt_client.transcribe(
                    wav_path,
                    language_code=language,
                    model=model,
                    format=fmt,
                    **options,
                )
                latency = time.monotonic() - t0
                transcript = raw.get("transcript", "")
                used_model = raw.get("model", model or "prisma-v2.5")
                logger.info(
                    "[STT] OK | model=%s latency=%.2fs transcript_len=%d",
                    used_model, latency, len(transcript),
                )
                return TranscribeResult(
                    text=transcript,
                    model=used_model,
                    latency_s=latency,
                    raw=raw,
                )
            except Exception as exc:
                latency = time.monotonic() - t0
                last_exc = exc
                if _is_retryable(exc) and attempt < self._max_attempts - 1:
                    sleep = _backoff(attempt)
                    logger.warning(
                        "[STT] attempt %d/%d failed (retryable) | sleeping %.1fs | %s",
                        attempt + 1, self._max_attempts, sleep, exc,
                    )
                    time.sleep(sleep)
                else:
                    logger.error(
                        "[STT] FAILED | model=%s language=%s attempt=%d | %s",
                        model, language, attempt + 1, exc,
                    )
                    raise
        raise last_exc  # type: ignore[misc]

