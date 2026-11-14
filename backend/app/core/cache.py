"""Disk caches for synthesized and degraded audio.

The TTS cache key is SHA-256 of (text + voice + tts_model + language). The
degraded-audio cache extends that identity with condition name, parameters,
and seed. Both cache values are WAV files stored under cache_dir/.

This means the same sentence is never billed twice, even across runs.
The cache is intentionally NOT cleared automatically; delete the cache_dir
to force a fresh synthesis run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _cache_key(text: str, voice: str, tts_model: str, language: str) -> str:
    """Return a hex SHA-256 digest that uniquely identifies a synthesis request."""
    payload = f"{text}\x00{voice}\x00{tts_model}\x00{language}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TTSCache:
    """Persistent on-disk cache for TTS WAV bytes.

    Parameters
    ----------
    cache_dir:
        Directory where cached .wav files are stored.
        Created automatically if it does not exist.
    """

    def __init__(self, cache_dir: Path | str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.wav"

    def get(
        self,
        text: str,
        voice: str,
        tts_model: str,
        language: str,
    ) -> bytes | None:
        """Return cached WAV bytes, or ``None`` if not cached."""
        p = self._path(_cache_key(text, voice, tts_model, language))
        if p.exists():
            return p.read_bytes()
        return None

    def put(
        self,
        text: str,
        voice: str,
        tts_model: str,
        language: str,
        audio_bytes: bytes,
    ) -> Path:
        """Store *audio_bytes* on disk and return the file path."""
        p = self._path(_cache_key(text, voice, tts_model, language))
        p.write_bytes(audio_bytes)
        return p

    def key(
        self,
        text: str,
        voice: str,
        tts_model: str,
        language: str,
    ) -> str:
        """Return the cache key (hex digest) for the given parameters."""
        return _cache_key(text, voice, tts_model, language)

    def degraded_key(
        self,
        text: str,
        voice: str,
        tts_model: str,
        language: str,
        condition_name: str,
        condition_params: Mapping[str, Any],
        seed: int,
    ) -> str:
        """Key a degraded clip by its TTS source and full condition identity."""
        payload = json.dumps(
            {
                "tts_key": self.key(text, voice, tts_model, language),
                "condition_name": condition_name,
                "condition_params": condition_params,
                "seed": seed,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def degraded_path(
        self,
        text: str,
        voice: str,
        tts_model: str,
        language: str,
        condition_name: str,
        condition_params: Mapping[str, Any],
        seed: int,
    ) -> Path:
        """Return the persistent cache path for one TTS/condition combination."""
        cache_dir = self._dir / "degraded"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self.degraded_key(
            text,
            voice,
            tts_model,
            language,
            condition_name,
            condition_params,
            seed,
        )
        return cache_dir / f"{key}.wav"

    def __contains__(self, args: tuple[str, str, str, str]) -> bool:
        text, voice, tts_model, language = args
        return self._path(_cache_key(text, voice, tts_model, language)).exists()

