"""Abstract base class and result types for TTS/STT providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SynthResult:
    """Return value of a TTS call.

    Attributes
    ----------
    audio_bytes : bytes
        A complete WAV file (RIFF header + 16-bit mono PCM).
    model : str
        The TTS model that produced the audio.
    voice : str
        The voice used.
    latency_s : float
        Wall-clock seconds from request start to response complete.
    """

    audio_bytes: bytes
    model: str
    voice: str
    latency_s: float


@dataclass
class TranscribeResult:
    """Return value of an STT call.

    Attributes
    ----------
    text : str
        The recognised transcript.
    model : str
        The STT model that produced the transcript.
    latency_s : float
        Wall-clock seconds from request start to response complete.
    raw : dict
        The full response payload from the provider (empty dict for fakes).
    """

    text: str
    model: str
    latency_s: float
    raw: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Interface that every TTS/STT backend must implement."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str,
        model: str,
        language: str,
    ) -> SynthResult:
        """Synthesise *text* and return a :class:`SynthResult`.

        Parameters
        ----------
        text:
            The sentence to speak.
        voice:
            Voice identifier (e.g. ``"Nalini"``).
        model:
            TTS model identifier (e.g. ``"timbre-v2.5"``).
        language:
            BCP-47 language code (e.g. ``"hi-IN"``).
        """

    @abstractmethod
    def transcribe(
        self,
        wav_path: Path,
        language: str,
        model: str | None = None,
        **options: Any,
    ) -> TranscribeResult:
        """Transcribe *wav_path* and return a :class:`TranscribeResult`.

        Parameters
        ----------
        wav_path:
            Path to a WAV file (≤ 30 s for REST endpoints).
        language:
            BCP-47 code (e.g. ``"hi-IN"``).
        model:
            STT model name or ``None`` for the server default.
        **options:
            Extra provider-specific kwargs, e.g. ``format="transcribe"``,
            ``keywords=[...]``.  Unknown keys should be silently ignored by
            fake providers and forwarded to the SDK by real ones.
        """
