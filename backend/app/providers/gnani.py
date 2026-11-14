"""GnaniProvider — one-attempt real SDK wrapper with timing and safe errors.

IMPORTANT: This module is imported lazily. It is NEVER imported in tests —
tests always use FakeProvider. The real API is only called when the user
runs run_demo.py or the pipeline with --provider gnani after explicit approval.

The pipeline owns retries so that every provider attempt is charged to the
run's BudgetGuard. This wrapper makes exactly one SDK call per method call.

Logging
-------
Every call is logged at INFO level with model, voice/language, and latency.
Errors are logged at ERROR level before re-raising.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.providers.base import BaseProvider, SynthResult, TranscribeResult

if TYPE_CHECKING:
    from app.db.models import User

logger = logging.getLogger(__name__)

def _safe_error_metadata(exc: Exception) -> tuple[int | None, str, bool]:
    """Extract safe error metadata without retaining the SDK message."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    error_type = str(getattr(exc, "source_error_type", type(exc).__name__))
    timed_out = isinstance(exc, (TimeoutError,)) or "timeout" in type(exc).__name__.lower()
    return status, error_type, timed_out


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
        max_attempts: int = 1,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        # Retained as a backwards-compatible argument. Retries belong to the
        # pipeline so each individual network attempt consumes budget.

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

        self._tts_client = GnaniTTSClient(api_key=self._api_key, timeout=60)
        self._stt_client = GnaniSTTClient(api_key=self._api_key, timeout=60)

        self._audio_config = tts_audio_config or AudioConfig(
            sample_rate=24000,
            encoding="linear_pcm",
            container="wav",
        )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def _safe_log_value(self, value: object) -> object:
        if not isinstance(value, str) or not self._api_key:
            return value
        return value.replace(self._api_key, "[redacted]")

    def synthesize(
        self,
        text: str,
        voice: str,
        model: str,
        language: str,
    ) -> SynthResult:
        """Call Gnani TTS once; retry policy is owned by the pipeline."""
        logger.info(
            "[TTS] synthesize | model=%s voice=%s language=%s text_len=%d",
            self._safe_log_value(model),
            self._safe_log_value(voice),
            self._safe_log_value(language),
            len(text),
        )
        t0 = time.monotonic()
        try:
            audio_bytes: bytes = self._tts_client.synthesize(
                text,
                voice=voice,
                model=model,
                language=language,
                audio_config=self._audio_config,
            )
        except Exception as exc:
            status, source_type, timed_out = _safe_error_metadata(exc)
            logger.error(
                "[TTS] FAILED | model=%s voice=%s error_type=%s http_status=%s",
                self._safe_log_value(model), self._safe_log_value(voice),
                source_type, status,
            )
            raise GnaniProviderError(
                "Gnani TTS request failed.", status_code=status,
                source_error_type=source_type, timed_out=timed_out,
            ) from None
        latency = time.monotonic() - t0
        logger.info(
            "[TTS] OK | model=%s latency=%.2fs bytes=%d",
            self._safe_log_value(model), latency, len(audio_bytes),
        )
        return SynthResult(audio_bytes=audio_bytes, model=model, voice=voice, latency_s=latency)

    def transcribe(
        self,
        wav_path: Path,
        language: str,
        model: str | None = None,
        **options: Any,
    ) -> TranscribeResult:
        """Call Gnani STT once; retry policy is owned by the pipeline.

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
            self._safe_log_value(model),
            self._safe_log_value(language),
            self._safe_log_value(fmt),
            self._safe_log_value(wav_path.name),
        )
        t0 = time.monotonic()
        try:
            raw: dict = self._stt_client.transcribe(
                wav_path,
                language_code=language,
                model=model,
                format=fmt,
                **options,
            )
        except Exception as exc:
            status, source_type, timed_out = _safe_error_metadata(exc)
            logger.error(
                "[STT] FAILED | model=%s language=%s error_type=%s http_status=%s",
                self._safe_log_value(model), self._safe_log_value(language),
                source_type, status,
            )
            raise GnaniProviderError(
                "Gnani STT request failed.", status_code=status,
                source_error_type=source_type, timed_out=timed_out,
            ) from None
        latency = time.monotonic() - t0
        transcript = raw.get("transcript", "")
        used_model = raw.get("model", model or "prisma-v2.5")
        logger.info(
            "[STT] OK | model=%s latency=%.2fs transcript_len=%d",
            self._safe_log_value(used_model), latency, len(transcript),
        )
        return TranscribeResult(
            text=transcript, model=used_model, latency_s=latency, raw=raw
        )


class GnaniProviderError(RuntimeError):
    """Sanitized provider failure with machine-readable safe metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        source_error_type: str = "GnaniProviderError",
        timed_out: bool = False,
    ) -> None:
        self.status_code = status_code
        self.http_status = status_code
        self.source_error_type = source_error_type
        self.timed_out = timed_out
        super().__init__(message)


def get_gnani_provider_for(user: User) -> GnaniProvider:
    """Build a provider using the current user's decrypted stored Gnani key."""
    from app.core.crypto import (
        CryptoConfigurationError,
        SecretDecryptionError,
        decrypt_secret,
    )
    if user.gnani_key is None:
        raise RuntimeError("No Gnani key is configured for this user.")
    try:
        api_key = decrypt_secret(user.gnani_key.ciphertext)
    except (CryptoConfigurationError, SecretDecryptionError):
        raise RuntimeError("The stored Gnani key could not be decrypted.") from None
    try:
        return GnaniProvider(api_key=api_key)
    except Exception:
        raise GnaniProviderError("Unable to initialize the Gnani provider.") from None

