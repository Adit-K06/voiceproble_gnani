"""Tests for the VoiceProbe pipeline — no real API calls.

All tests use FakeProvider and a temporary directory; they never touch
the network, never read .env, and cost zero API credits.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.core.budget import BudgetGuard, BudgetExceeded
from app.core.cache import TTSCache
from app.core.pipeline import CONDITIONS, _degrade, run_case
from app.providers.fake import FakeProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reference_sentence() -> dict:
    return {
        "sentence_id": "test_001",
        "text": "आपके खाते में दो लाख रुपये जमा हो गए हैं।",
        "language": "hi-IN",
        "reference": "आपके खाते में दो लाख रुपये जमा हो गए हैं।",
    }


def _run(provider, budget, cache, output_dir, condition="clean", sentence=None):
    s = sentence or _reference_sentence()
    return run_case(
        sentence_id=s["sentence_id"],
        text=s["text"],
        language=s["language"],
        reference=s["reference"],
        condition=condition,
        tts_model="fake-tts-v1",
        tts_voice="FakeVoice",
        stt_model=None,
        stt_format="verbatim",
        provider=provider,
        budget=budget,
        cache=cache,
        output_dir=Path(output_dir),
    )


# ---------------------------------------------------------------------------
# Test 1: Pipeline runs end-to-end and returns a CaseResult with a wer field
# ---------------------------------------------------------------------------

def test_pipeline_runs_and_returns_wer():
    provider = FakeProvider()
    with tempfile.TemporaryDirectory() as tmp:
        budget = BudgetGuard(max_calls=10)
        cache = TTSCache(Path(tmp) / "tts_cache")
        result = _run(provider, budget, cache, tmp)

    assert hasattr(result, "wer"), "CaseResult must have a 'wer' attribute"
    assert isinstance(result.wer, float)
    assert 0.0 <= result.wer <= 1.0 or result.wer > 1.0  # WER can exceed 1.0
    assert result.sentence_id == "test_001"
    assert result.condition == "clean"
    assert result.amount_correct is None
    assert result.entity_accuracy_by_type["amount"] is None
    assert result.to_dict()["entity_accuracy_by_type"]["date"] is None


# ---------------------------------------------------------------------------
# Test 2: Cache avoids a second TTS API call for the same sentence
# ---------------------------------------------------------------------------

def test_cache_avoids_second_tts_call():
    provider = FakeProvider()
    with tempfile.TemporaryDirectory() as tmp:
        budget = BudgetGuard(max_calls=20)
        cache = TTSCache(Path(tmp) / "tts_cache")

        # First run — should call TTS once
        _run(provider, budget, cache, tmp)
        assert len(provider.synth_calls) == 1, "First run must call TTS once"

        # Second run with the same sentence/voice/model — cache should serve it
        _run(provider, budget, cache, tmp)
        assert len(provider.synth_calls) == 1, (
            "Second run must NOT call TTS again (cache hit expected)"
        )

        # Budget: 1 TTS (first) + 2 STT (one per run) = 3 total
        assert budget.tts_calls == 1
        assert budget.stt_calls == 2


# ---------------------------------------------------------------------------
# Test 3: BudgetGuard raises BudgetExceeded when the limit is hit
# ---------------------------------------------------------------------------

def test_budget_raises_on_limit():
    budget = BudgetGuard(max_calls=1)
    budget.charge("tts")           # uses up the only credit

    with pytest.raises(BudgetExceeded) as exc_info:
        budget.charge("stt")

    assert exc_info.value.max_calls == 1
    assert exc_info.value.used == 1
    assert "stt" in exc_info.value.kind


def test_budget_guard_stops_pipeline():
    """TTS in cache, budget=1 and already 1 call used: STT charge should exceed budget."""
    provider = FakeProvider()
    with tempfile.TemporaryDirectory() as tmp:
        # Pre-warm the TTS cache so the pipeline won't charge a TTS call.
        cache = TTSCache(Path(tmp) / "tts_cache")
        s = _reference_sentence()
        silent = _make_wav()
        cache.put(s["text"], "FakeVoice", "fake-tts-v1", s["language"], silent)

        budget = BudgetGuard(max_calls=1)
        budget.charge("tts")   # exhaust the single allowed credit

        # STT call inside pipeline should now raise BudgetExceeded.
        with pytest.raises(BudgetExceeded):
            _run(provider, budget, cache, tmp)


def _make_wav() -> bytes:
    """Minimal silent WAV for cache pre-warming."""
    sample_rate = 24000
    num_samples = 100
    pcm = b"\x00\x00" * num_samples
    data_size = len(pcm)
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    ) + pcm


# ---------------------------------------------------------------------------
# Test 4: WER is 0.0 when FakeProvider echoes the reference exactly
# ---------------------------------------------------------------------------

def test_wer_zero_when_transcript_matches_reference():
    """If the STT returns the exact reference text, WER must be 0.0."""
    s = _reference_sentence()
    with tempfile.TemporaryDirectory() as tmp:
        # Pre-compute what the degraded WAV path will be.
        degraded_key = str(Path(tmp) / "audio" / f"{s['sentence_id']}_clean.wav")
        provider = FakeProvider(
            transcript_map={degraded_key: s["reference"]}
        )
        budget = BudgetGuard(max_calls=10)
        cache = TTSCache(Path(tmp) / "tts_cache")
        result = _run(provider, budget, cache, tmp, condition="clean", sentence=s)

    assert result.wer == pytest.approx(0.0), (
        f"WER should be 0.0 when transcript matches reference, got {result.wer}"
    )


# ---------------------------------------------------------------------------
# Test 5: Degraded WAV file is saved to output_dir/audio/
# ---------------------------------------------------------------------------

def test_degraded_wav_is_saved():
    provider = FakeProvider()
    with tempfile.TemporaryDirectory() as tmp:
        budget = BudgetGuard(max_calls=10)
        cache = TTSCache(Path(tmp) / "tts_cache")
        result = _run(provider, budget, cache, tmp, condition="phone_snr20")
        # Check inside the with-block before the temp dir is cleaned up.
        degraded = Path(result.degraded_wav_path)
        original = Path(result.original_wav_path)

        assert degraded.exists(), f"Degraded WAV not found at {degraded}"
        assert degraded.suffix == ".wav"
        assert degraded.stat().st_size > 44, "Degraded WAV should be larger than a bare header"
        assert degraded.name == "test_001_phone_snr20.wav", f"Unexpected name: {degraded.name}"

        assert original.exists(), f"Original TTS WAV not found at {original}"
        assert original.name == "test_001_original.wav", f"Unexpected name: {original.name}"


# ---------------------------------------------------------------------------
# Test 6: TTSCache key is deterministic and cache get/put round-trip works
# ---------------------------------------------------------------------------

def test_tts_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cache = TTSCache(Path(tmp) / "cache")
        payload = b"RIFF\x24\x00\x00\x00WAVEdata\x00\x00\x00\x00"
        cache.put("hello", "Voice", "model-v1", "hi-IN", payload)

        result = cache.get("hello", "Voice", "model-v1", "hi-IN")
        assert result == payload

        # Different text -> miss
        assert cache.get("world", "Voice", "model-v1", "hi-IN") is None


def test_degraded_cache_key_includes_condition_name_params_and_seed():
    with tempfile.TemporaryDirectory() as tmp:
        cache = TTSCache(Path(tmp) / "cache")
        base = ("hello", "Voice", "model-v1", "hi-IN")
        snr5 = cache.degraded_key(
            *base,
            "phone_snr5",
            {"phone_path": True, "snr_db": 5.0},
            42,
        )
        snr0 = cache.degraded_key(
            *base,
            "phone_snr0",
            {"phone_path": True, "snr_db": 0.0},
            42,
        )
        changed_params = cache.degraded_key(
            *base,
            "phone_snr5",
            {"phone_path": True, "snr_db": 4.0},
            42,
        )
        changed_seed = cache.degraded_key(
            *base,
            "phone_snr5",
            {"phone_path": True, "snr_db": 5.0},
            7,
        )

    assert len({snr5, snr0, changed_params, changed_seed}) == 4


# ---------------------------------------------------------------------------
# Test 7: BudgetGuard counters track tts vs stt separately
# ---------------------------------------------------------------------------

def test_budget_counters():
    b = BudgetGuard(max_calls=10)
    b.charge("tts")
    b.charge("tts")
    b.charge("stt")
    assert b.tts_calls == 2
    assert b.stt_calls == 1
    assert b.used == 3
    assert b.remaining == 7



@pytest.mark.parametrize(
    "condition",
    ["phone_snr5", "phone_snr0", "phone_clip", "phone_loss10"],
)
def test_new_phone_conditions_are_registered_reproducible_and_keep_length(condition):
    sample_rate = 16000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.5 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)

    assert condition in CONDITIONS
    first, first_sr = _degrade(audio, sample_rate, condition)
    second, second_sr = _degrade(audio, sample_rate, condition)

    assert first_sr == second_sr == sample_rate
    assert len(first) == len(second) == len(audio)
    assert np.array_equal(first, second)
