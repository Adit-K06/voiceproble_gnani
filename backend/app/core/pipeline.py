"""VoiceProbe core pipeline.

run_case() is the single entry point for one (sentence, condition, tts_cfg, stt_cfg) tuple.

Conditions
----------
  clean        — no degradation at all; audio is passed through as-is after resampling to
                 24 kHz (the TTS output rate). WER baseline.
  phone_8k     — mu-law roundtrip + 8 kHz downsample, NO additive noise.
  phone_snr20  — full telephone_degrade with SNR=20 dB.
  phone_snr10  — full telephone_degrade with SNR=10 dB.
  phone_snr5   — full telephone_degrade with SNR=5 dB.
  phone_snr0   — full telephone_degrade with SNR=0 dB.
  phone_clip   — hard clipping at 40% peak, then the telephone profile.
  phone_loss10 — 10% random 20 ms packet loss, then the telephone profile.

A fixed noise seed (42) is used so results are reproducible.

The degraded WAV is always written to output_dir/audio/<sentence_id>_<condition>.wav
so the user can listen to it.

HONESTY NOTE (reproduced in every output):
  Audio is machine-generated (Gnani Timbre TTS) and noise is simulated.
  Results are RELATIVE comparisons only — not real-world accuracy figures.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from app.core.audio import (
    CONDITION_REGISTRY,
    apply_condition,
    normalize_wav_bytes,
    read_wav_tolerant,
    save_wav,
)
from app.core.budget import BudgetGuard, BudgetExceeded  # noqa: F401 (re-exported)
from app.core.cache import TTSCache
from app.core.entities import score_entity_types
from app.core.scoring import normalize, spacing_only_difference, wer_best
from app.providers.base import BaseProvider, SynthResult, TranscribeResult

logger = logging.getLogger(__name__)

# Noise seed — fixed for reproducibility.
_NOISE_SEED = 42
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_DELAYS_S = (2.0, 5.0)

# Preserve the legacy name -> SNR lookup while the full definitions live in the
# audio condition registry.
CONDITIONS: dict[str, float | None] = {
    name: (
        float(condition.params["snr_db"])
        if condition.params.get("snr_db") is not None
        else None
    )
    for name, condition in CONDITION_REGISTRY.items()
}


@dataclass
class CaseResult:
    """Result of one (sentence, condition, tts_cfg, stt_cfg) run."""

    sentence_id: str
    condition: str
    tts_model: str
    tts_voice: str
    stt_model: str
    stt_format: str
    reference: str
    hypothesis: str
    wer: float
    tts_latency_s: float
    stt_latency_s: float
    cache_hit: bool
    original_wav_path: str
    degraded_wav_path: str
    raw_stt: dict[str, Any] = field(default_factory=dict)
    accepted_references: list[str] = field(default_factory=list)
    spacing_only: bool = False
    expected_entities: list[dict[str, Any]] = field(default_factory=list)
    amount_correct: bool | None = None
    entity_accuracy_by_type: dict[str, bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "sentence_id": self.sentence_id,
            "condition": self.condition,
            "tts_model": self.tts_model,
            "tts_voice": self.tts_voice,
            "stt_model": self.stt_model,
            "stt_format": self.stt_format,
            "reference": self.reference,
            "accepted_references": self.accepted_references,
            "hypothesis": self.hypothesis,
            "wer": round(self.wer, 4),
            "spacing_only": self.spacing_only,
            "expected_entities": self.expected_entities,
            "amount_correct": self.amount_correct,
            "entity_accuracy_by_type": self.entity_accuracy_by_type,
            "tts_latency_s": round(self.tts_latency_s, 3),
            "stt_latency_s": round(self.stt_latency_s, 3),
            "cache_hit": self.cache_hit,
            "original_wav_path": self.original_wav_path,
            "degraded_wav_path": self.degraded_wav_path,
            "_honesty_note": (
                "Audio is machine-generated (Gnani Timbre TTS) and noise is simulated. "
                "Results are RELATIVE comparisons only, not real-world accuracy figures."
            ),
        }


def _retry_status(exc: Exception) -> int | None:
    status = getattr(exc, "http_status", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_timeout(exc: Exception) -> bool:
    if getattr(exc, "timed_out", False):
        return True
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    # Keep requests optional for FakeProvider-only installs and tests.
    return "timeout" in type(exc).__name__.lower()


def _provider_call(
    kind: str,
    budget: BudgetGuard,
    operation,
):
    """Call a provider with budgeted retries for transient failures."""
    started = time.monotonic()
    for attempt in range(3):
        budget.charge(kind)
        try:
            result = operation()
            return replace(result, latency_s=time.monotonic() - started)
        except Exception as exc:
            status = _retry_status(exc)
            retryable = status in _RETRYABLE_STATUSES or _is_timeout(exc)
            if not retryable or attempt == 2:
                logger.warning(
                    "[pipeline] %s failed | attempt=%d error_type=%s http_status=%s",
                    kind.upper(), attempt + 1, type(exc).__name__, status,
                )
                raise
            delay = _RETRY_DELAYS_S[attempt]
            logger.warning(
                "[pipeline] %s transient failure | attempt=%d error_type=%s "
                "http_status=%s retry_in=%.1fs",
                kind.upper(), attempt + 1, type(exc).__name__, status, delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable retry loop")


def _degrade(
    audio: np.ndarray,
    sr: int,
    condition: str,
    seed: int = _NOISE_SEED,
) -> tuple[np.ndarray, int]:
    """Apply the registered condition; retained as the pipeline's old helper."""
    return apply_condition(audio, sr, condition, seed)


def run_case(
    *,
    sentence_id: str,
    text: str,
    language: str,
    reference: str,
    condition: str,
    tts_model: str,
    tts_voice: str,
    stt_model: str | None,
    stt_format: str,
    provider: BaseProvider,
    budget: BudgetGuard,
    cache: TTSCache,
    output_dir: Path,
    accepted_references: list[str] | None = None,
    expected_entities: list[dict[str, Any]] | None = None,
) -> CaseResult:
    """Run one VoiceProbe test case end-to-end."""
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}. "
            f"Valid conditions: {sorted(CONDITIONS)}"
        )

    accepted_references = list(accepted_references or [])
    expected_entities = [dict(entity) for entity in (expected_entities or [])]
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    cached_bytes = cache.get(text, tts_voice, tts_model, language)
    cache_hit = cached_bytes is not None

    if cache_hit:
        logger.info(
            "[pipeline] TTS cache HIT | %s voice=%s model=%s",
            sentence_id, tts_voice, tts_model,
        )
        normalized_bytes = normalize_wav_bytes(cached_bytes)  # type: ignore[arg-type]
        if normalized_bytes != cached_bytes:
            # Old cache entries may contain Gnani's invalid 0xFFFFFFFF header.
            cache.put(text, tts_voice, tts_model, language, normalized_bytes)
        tts_result = SynthResult(
            audio_bytes=normalized_bytes,
            model=tts_model,
            voice=tts_voice,
            latency_s=0.0,
        )
    else:
        logger.info(
            "[pipeline] TTS request | %s voice=%s model=%s",
            sentence_id, tts_voice, tts_model,
        )
        tts_result = _provider_call(
            "tts",
            budget,
            lambda: provider.synthesize(
                text, voice=tts_voice, model=tts_model, language=language
            ),
        )
        normalized_bytes = normalize_wav_bytes(tts_result.audio_bytes)
        tts_result = replace(tts_result, audio_bytes=normalized_bytes)
        cache.put(text, tts_voice, tts_model, language, normalized_bytes)

    original_wav_path = audio_dir / f"{sentence_id}_original.wav"
    # Always rewrite to repair originals created by earlier runs too.
    original_wav_path.write_bytes(tts_result.audio_bytes)
    logger.info("[pipeline] original TTS WAV saved | %s", original_wav_path)

    audio, sr = read_wav_tolerant(tts_result.audio_bytes)
    condition_definition = CONDITION_REGISTRY[condition]
    degraded_cache_path = cache.degraded_path(
        text,
        tts_voice,
        tts_model,
        language,
        condition_definition.name,
        condition_definition.params,
        _NOISE_SEED,
    )
    if not degraded_cache_path.exists():
        degraded_audio, degraded_sr = _degrade(
            audio, sr, condition, seed=_NOISE_SEED
        )
        save_wav(degraded_cache_path, degraded_audio, degraded_sr)

    wav_name = f"{sentence_id}_{condition}.wav"
    degraded_wav_path = audio_dir / wav_name
    degraded_wav_path.write_bytes(degraded_cache_path.read_bytes())
    logger.info("[pipeline] degraded WAV saved | %s", degraded_wav_path)

    logger.info(
        "[pipeline] STT request | %s condition=%s model=%s format=%s",
        sentence_id, condition, stt_model, stt_format,
    )
    stt_result: TranscribeResult = _provider_call(
        "stt",
        budget,
        lambda: provider.transcribe(
            degraded_wav_path,
            language=language,
            model=stt_model,
            format=stt_format,
        ),
    )

    references = [reference, *accepted_references]
    score = wer_best(references, stt_result.text)
    has_exact_normalized_match = any(
        normalize(candidate) == normalize(stt_result.text)
        for candidate in references
    )
    spacing_only = (
        score == 0.0
        and not has_exact_normalized_match
        and any(
            spacing_only_difference(candidate, stt_result.text)
            for candidate in references
        )
    )
    entity_accuracy_by_type = score_entity_types(
        expected_entities, stt_result.text
    )
    amount_ok = entity_accuracy_by_type["amount"]
    logger.info(
        "[pipeline] DONE | %s condition=%s WER=%.3f hypothesis=%r",
        sentence_id, condition, score, stt_result.text,
    )

    return CaseResult(
        sentence_id=sentence_id,
        condition=condition,
        tts_model=tts_model,
        tts_voice=tts_voice,
        stt_model=stt_result.model,
        stt_format=stt_format,
        reference=reference,
        hypothesis=stt_result.text,
        wer=score,
        tts_latency_s=tts_result.latency_s,
        stt_latency_s=stt_result.latency_s,
        cache_hit=cache_hit,
        original_wav_path=str(original_wav_path),
        degraded_wav_path=str(degraded_wav_path),
        raw_stt=stt_result.raw,
        accepted_references=accepted_references,
        spacing_only=spacing_only,
        expected_entities=expected_entities,
        amount_correct=amount_ok,
        entity_accuracy_by_type=entity_accuracy_by_type,
    )
