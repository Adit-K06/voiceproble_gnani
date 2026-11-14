from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from scipy.io import wavfile

from app.core.audio import normalize_wav_bytes, read_wav_tolerant
from app.core.budget import BudgetGuard
from app.core.cache import TTSCache
from app.providers.base import TranscribeResult
from app.providers.fake import FakeProvider
from scripts import run_demo


class HTTPFailure(RuntimeError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def _sentence(identifier: str = "offline_001") -> dict:
    return {
        "id": identifier,
        "text": "नमस्ते, यह synthetic test है।",
        "language": "hi-IN",
        "reference": "नमस्ते, यह synthetic test है।",
        "expected_entities": [],
    }


class FailTwiceThenSucceed(FakeProvider):
    def transcribe(self, *args, **kwargs):
        self.transcribe_calls.append({"attempt": len(self.transcribe_calls) + 1})
        if len(self.transcribe_calls) <= 2:
            raise HTTPFailure(504)
        return TranscribeResult("नमस्ते, यह synthetic test है।", "fake-stt-v1", 0.001)


def test_504_twice_then_success_retries_and_charges_each_attempt(tmp_path, monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr("app.core.pipeline.time.sleep", delays.append)
    provider = FailTwiceThenSucceed()
    budget = BudgetGuard(max_calls=10)

    result = run_demo.run_units(
        provider=provider,
        budget=budget,
        cache=TTSCache(tmp_path / "cache"),
        output_dir=tmp_path,
        sentences=[_sentence()],
        conditions=["clean"],
    )

    row = next(iter(result.values()))
    assert row["status"] == "ok"
    assert len(provider.transcribe_calls) == 3
    assert budget.tts_calls == 1
    assert budget.stt_calls == 3
    assert delays == [2.0, 5.0]


class TTSFailTwiceThenSucceed(FakeProvider):
    def __init__(self):
        super().__init__()
        self.tts_attempts = 0

    def synthesize(self, *args, **kwargs):
        self.tts_attempts += 1
        if self.tts_attempts <= 2:
            raise HTTPFailure(503)
        return super().synthesize(*args, **kwargs)


def test_tts_retries_are_also_charged(tmp_path, monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr("app.core.pipeline.time.sleep", delays.append)
    provider = TTSFailTwiceThenSucceed()
    budget = BudgetGuard(max_calls=10)

    result = run_demo.run_units(
        provider=provider,
        budget=budget,
        cache=TTSCache(tmp_path / "cache"),
        output_dir=tmp_path,
        sentences=[_sentence()],
        conditions=["clean"],
    )

    assert next(iter(result.values()))["status"] == "ok"
    assert provider.tts_attempts == 3
    assert len(provider.synth_calls) == 1
    assert budget.tts_calls == 3
    assert budget.stt_calls == 1
    assert delays == [2.0, 5.0]


class FailOnlyFirstUnit(FakeProvider):
    def transcribe(self, wav_path, language, model=None, **options):
        self.transcribe_calls.append({"wav_path": str(wav_path)})
        if wav_path.name.endswith("_clean.wav"):
            raise HTTPFailure(504)
        return TranscribeResult("transcript", "fake-stt-v1", 0.001)


def test_permanent_unit_error_is_journaled_and_later_unit_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.pipeline.time.sleep", lambda _seconds: None)
    provider = FailOnlyFirstUnit()
    latest = run_demo.run_units(
        provider=provider,
        budget=BudgetGuard(max_calls=20),
        cache=TTSCache(tmp_path / "cache"),
        output_dir=tmp_path,
        sentences=[_sentence()],
        conditions=["clean", "phone_8k"],
    )

    rows = list(latest.values())
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[0]["http_status"] == 504
    assert len(provider.transcribe_calls) == 4
    journal_lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["status"] for line in journal_lines] == ["error", "ok"]
    summary = run_demo._summary_by_condition(rows)
    assert summary["clean"]["error_count"] == 1
    assert summary["clean"]["mean_wer"] is None
    assert summary["phone_8k"]["error_count"] == 0


def test_resume_skips_successfully_finished_unit(tmp_path):
    sentence = _sentence()
    cache = TTSCache(tmp_path / "cache")
    first_provider = FakeProvider()
    first = run_demo.run_units(
        provider=first_provider,
        budget=BudgetGuard(max_calls=10),
        cache=cache,
        output_dir=tmp_path,
        sentences=[sentence],
        conditions=["clean"],
    )
    second_provider = FakeProvider()
    second = run_demo.run_units(
        provider=second_provider,
        budget=BudgetGuard(max_calls=10),
        cache=cache,
        output_dir=tmp_path,
        sentences=[sentence],
        conditions=["clean"],
    )

    assert len(first) == len(second) == 1
    assert second_provider.synth_calls == []
    assert second_provider.transcribe_calls == []
    assert (tmp_path / "results.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_retry_errors_reruns_error_rows_and_updates_latest_status(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.pipeline.time.sleep", lambda _seconds: None)
    sentence = _sentence()
    cache = TTSCache(tmp_path / "cache")
    failed = run_demo.run_units(
        provider=FailOnlyFirstUnit(),
        budget=BudgetGuard(max_calls=10),
        cache=cache,
        output_dir=tmp_path,
        sentences=[sentence],
        conditions=["clean"],
    )
    assert next(iter(failed.values()))["status"] == "error"

    retried = run_demo.run_units(
        provider=FakeProvider(),
        budget=BudgetGuard(max_calls=10),
        cache=cache,
        output_dir=tmp_path,
        sentences=[sentence],
        conditions=["clean"],
        retry_errors=True,
    )
    assert next(iter(retried.values()))["status"] == "ok"
    journal_rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in journal_rows] == ["error", "ok"]


def test_read_wav_tolerant_and_cache_repair_for_sentinel_data_size(tmp_path):
    wav_path = tmp_path / "good.wav"
    wavfile.write(wav_path, 24000, np.array([0, 1024, -1024, 0] * 25, dtype=np.int16))
    normal = wav_path.read_bytes()
    malformed = bytearray(normal)
    data_chunk = malformed.index(b"data")
    struct.pack_into("<I", malformed, data_chunk + 4, 0xFFFFFFFF)

    audio, sample_rate = read_wav_tolerant(bytes(malformed))
    assert sample_rate == 24000
    assert len(audio) == 100
    fixed = normalize_wav_bytes(bytes(malformed))
    assert struct.unpack_from("<I", fixed, 4)[0] == len(fixed) - 8

    sentence = _sentence()
    cache = TTSCache(tmp_path / "cache")
    cache.put(sentence["text"], run_demo.TTS_VOICE, run_demo.TTS_MODEL, sentence["language"], bytes(malformed))

    class NoTTSProvider(FakeProvider):
        def synthesize(self, *args, **kwargs):
            raise AssertionError("cache hit must not call TTS")

    result = run_demo.run_units(
        provider=NoTTSProvider(),
        budget=BudgetGuard(max_calls=5),
        cache=cache,
        output_dir=tmp_path / "cached-run",
        sentences=[sentence],
        conditions=["clean"],
    )
    assert next(iter(result.values()))["status"] == "ok"
    repaired = cache.get(sentence["text"], run_demo.TTS_VOICE, run_demo.TTS_MODEL, sentence["language"])
    assert repaired is not None
    assert struct.unpack_from("<I", repaired, 40)[0] != 0xFFFFFFFF
    original = (tmp_path / "cached-run" / "audio" / "offline_001_original.wav").read_bytes()
    assert struct.unpack_from("<I", original, 40)[0] != 0xFFFFFFFF


def test_empty_pending_plan_has_zero_calls(tmp_path):
    assert run_demo._estimate_api_calls(TTSCache(tmp_path / "cache"), []) == (0, 0)
