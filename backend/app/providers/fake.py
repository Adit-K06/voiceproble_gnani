"""Fake provider for unit tests.

Returns deterministic, predictable results without any network calls.
The TTS output is a minimal valid WAV (silence at 24 kHz, 0.1 s).
The STT output echoes back text looked up from an optional map.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any

from app.providers.base import BaseProvider, SynthResult, TranscribeResult


def _silent_wav(sample_rate: int = 24000, duration_s: float = 0.1) -> bytes:
    """Return a minimal valid RIFF/WAV file containing silence."""
    num_samples = int(sample_rate * duration_s)
    pcm = b"\x00\x00" * num_samples          # 16-bit zero samples
    data_size = len(pcm)
    riff_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1,                      # PCM
        1,                      # mono
        sample_rate,
        sample_rate * 2,        # byte rate
        2,                      # block align
        16,                     # bits per sample
        b"data", data_size,
    )
    return header + pcm


class FakeProvider(BaseProvider):
    """Deterministic stub — zero network calls, zero API credits.

    Parameters
    ----------
    transcript_map:
        Optional dict mapping ``str(wav_path)`` → transcript string, so
        callers can inject specific STT outputs (including deliberate errors).
    stt_model_name:
        The ``model`` field returned in every :class:`TranscribeResult`.
    tts_model_name:
        The ``model`` field returned in every :class:`SynthResult`.
    """

    def __init__(
        self,
        transcript_map: dict[str, str] | None = None,
        stt_model_name: str = "fake-stt-v1",
        tts_model_name: str = "fake-tts-v1",
    ) -> None:
        self._transcript_map = transcript_map or {}
        self._stt_model_name = stt_model_name
        self._tts_model_name = tts_model_name
        # Public call logs — tests can assert on these.
        self.synth_calls: list[dict] = []
        self.transcribe_calls: list[dict] = []

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
        self.synth_calls.append(
            {"text": text, "voice": voice, "model": model, "language": language}
        )
        return SynthResult(
            audio_bytes=_silent_wav(),
            model=self._tts_model_name,
            voice=voice,
            latency_s=0.001,
        )

    def transcribe(
        self,
        wav_path: Path,
        language: str,
        model: str | None = None,
        **options: Any,
    ) -> TranscribeResult:
        key = str(wav_path)
        transcript = self._transcript_map.get(key, "फेक ट्रांसक्रिप्ट")
        self.transcribe_calls.append(
            {"wav_path": key, "language": language, "model": model, "options": options}
        )
        return TranscribeResult(
            text=transcript,
            model=self._stt_model_name,
            latency_s=0.001,
            raw={},
        )
