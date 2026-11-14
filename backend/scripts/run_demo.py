"""VoiceProbe demo runner.

Run from backend/ with:
    python -m scripts.run_demo [--max-calls N] [--output-dir PATH]

This script runs 5 synthetic Hindi/Hinglish banking sentences through all
registered phone conditions using the Gnani Prisma STT default model
(prisma-v2.5) in verbatim mode.

The script estimates uncached TTS + STT requests before starting. It uses one
provider attempt per request so the estimate and BudgetGuard count match.
Hard limit: --max-calls (default 60).

HONESTY NOTICE (printed at start and embedded in results.json):
  Audio is machine-generated (Gnani Timbre TTS) and noise is simulated.
  Results are RELATIVE comparisons only, not real-world accuracy figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow `python -m scripts.run_demo` from backend/
# ---------------------------------------------------------------------------
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from dotenv import load_dotenv
load_dotenv(_BACKEND / ".env")

from app.core.budget import BudgetGuard, BudgetExceeded
from app.core.cache import TTSCache
from app.core.pipeline import run_case, CaseResult
from app.providers.gnani import GnaniProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_demo")

# ---------------------------------------------------------------------------
# Honesty banner
# ---------------------------------------------------------------------------
HONESTY_NOTE = (
    "⚠  HONESTY NOTICE: Audio is machine-generated (Gnani Timbre TTS) and "
    "noise is simulated. Results are RELATIVE comparisons only — not "
    "real-world ASR accuracy figures."
)

# ---------------------------------------------------------------------------
# Test suite (synthetic data only — no real Aadhaar / PAN / account numbers)
# ---------------------------------------------------------------------------
SENTENCES = [
    {
        "id": "hi_001",
        "text": "आपके खाते में दो लाख पचास हजार रुपये जमा हो गए हैं।",
        "language": "hi-IN",
        "reference": "आपके खाते में दो लाख पचास हजार रुपये जमा हो गए हैं।",
        "expected_entities": [{"type": "amount", "value": 250000}],
    },
    {
        "id": "hi_002",
        "text": "आपकी ईएमआई तीन हजार पाँच सौ रुपये प्रति माह है।",
        "language": "hi-IN",
        "reference": "आपकी ईएमआई तीन हजार पाँच सौ रुपये प्रति माह है।",
        "expected_entities": [{"type": "amount", "value": 3500}],
    },
    {
        "id": "hi_003",
        "text": "बीस तारीख को आपका बिल भुगतान होगा।",
        "language": "hi-IN",
        "reference": "बीस तारीख को आपका बिल भुगतान होगा।",
        "expected_entities": [],
    },
    {
        "id": "hi_004",
        "text": "एक करोड़ रुपये का ऋण तीस साल के लिए उपलब्ध है।",
        "language": "hi-IN",
        "reference": "एक करोड़ रुपये का ऋण तीस साल के लिए उपलब्ध है।",
        "expected_entities": [{"type": "amount", "value": 10000000}],
    },
    {
        "id": "hi_005",
        "text": "आपका account balance check करने के liye star one dabaye.",
        "language": "hi-IN",
        "reference": "आपका account balance check करने के liye star one dabaye.",
        "expected_entities": [],
        "accepted_references": [
            "आपका अकाउंट बैलेंस चेक करने के लिए स्टार वन दबाएँ।"
        ],
    },
]

CONDITIONS = [
    "clean",
    "phone_8k",
    "phone_snr20",
    "phone_snr10",
    "phone_snr5",
    "phone_snr0",
    "phone_clip",
    "phone_loss10",
]

TTS_MODEL = "timbre-v2.5"
TTS_VOICE = "Nalini"
TTS_LANGUAGE = "hi-IN"

STT_MODEL = None          # server default = prisma-v2.5
STT_FORMAT = "verbatim"


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------
def _amount_accuracy_by_condition(
    results: list[CaseResult],
) -> dict[str, dict[str, int | float]]:
    from collections import defaultdict

    by_condition: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        if result.amount_correct is not None:
            by_condition[result.condition].append(result.amount_correct)

    return {
        condition: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
        for condition, values in by_condition.items()
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _summary_by_condition(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = [result for result in results if result.condition == condition]
        amount_flags = [
            result.amount_correct
            for result in rows
            if result.amount_correct is not None
        ]
        stt_latencies = [result.stt_latency_s for result in rows]
        summary[condition] = {
            "count": len(rows),
            "mean_wer": (
                sum(result.wer for result in rows) / len(rows) if rows else None
            ),
            "amount_correct": sum(amount_flags),
            "amount_scored": len(amount_flags),
            "amount_accuracy": (
                sum(amount_flags) / len(amount_flags) if amount_flags else None
            ),
            "stt_latency_p50_s": _percentile(stt_latencies, 0.50),
            "stt_latency_p95_s": _percentile(stt_latencies, 0.95),
        }
    return summary


def _estimate_api_calls(cache: TTSCache) -> tuple[int, int]:
    """Return unique uncached TTS requests and the STT requests for all cases."""
    tts_missing_by_key: dict[str, bool] = {}
    for sentence in SENTENCES:
        key = cache.key(sentence["text"], TTS_VOICE, TTS_MODEL, TTS_LANGUAGE)
        if key not in tts_missing_by_key:
            tts_missing_by_key[key] = (
                cache.get(sentence["text"], TTS_VOICE, TTS_MODEL, TTS_LANGUAGE)
                is None
            )
    return sum(tts_missing_by_key.values()), len(SENTENCES) * len(CONDITIONS)


def _print_table(results: list[CaseResult]) -> None:
    header = f"{'ID':<10} {'Condition':<14} {'WER':>6} {'Spacing':>8} {'Amt':>5}  {'TTS lat':>8} {'STT lat':>8}  {'Cache':>5}  Hypothesis (first 40 chars)"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        hyp_preview = r.hypothesis[:40].replace("\n", " ")
        cache_flag = "HIT" if r.cache_hit else "miss"
        amount_flag = (
            "n/a"
            if r.amount_correct is None
            else "OK" if r.amount_correct else "MISS"
        )
        print(
            f"{r.sentence_id:<10} {r.condition:<14} {r.wer:>6.3f} "
            f"{str(r.spacing_only):>8} {amount_flag:>5}  "
            f"{r.tts_latency_s:>7.2f}s {r.stt_latency_s:>7.2f}s  "
            f"{cache_flag:>5}  {hyp_preview}"
        )
    print(sep)
    print()
    print("Summary by condition:")
    print(
        f"{'Condition':<16} {'Cases':>5} {'Mean WER':>10} "
        f"{'Amount accuracy':>18} {'STT p50':>10} {'STT p95':>10}"
    )
    for condition, metrics in _summary_by_condition(results).items():
        mean_wer = (
            "n/a" if metrics["mean_wer"] is None else f"{metrics['mean_wer']:.3f}"
        )
        amount_accuracy = (
            "n/a"
            if metrics["amount_accuracy"] is None
            else f"{metrics['amount_correct']}/{metrics['amount_scored']} "
            f"({metrics['amount_accuracy']:.1%})"
        )
        p50 = metrics["stt_latency_p50_s"]
        p95 = metrics["stt_latency_p95_s"]
        p50_display = "n/a" if p50 is None else f"{p50:.3f}s"
        p95_display = "n/a" if p95 is None else f"{p95:.3f}s"
        print(
            f"{condition:<16} {metrics['count']:>5} {mean_wer:>10} "
            f"{amount_accuracy:>18} {p50_display:>10} {p95_display:>10}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceProbe demo runner")
    parser.add_argument(
        "--max-calls", type=int, default=60,
        help="Hard API call budget for this run (default: 60)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_BACKEND / "output",
        help="Root directory for audio files and results.json",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Announce the honesty notice immediately.
    print()
    print("=" * 70)
    print(HONESTY_NOTE)
    print("=" * 70)

    n_sentences = len(SENTENCES)
    n_conditions = len(CONDITIONS)
    cache = TTSCache(cache_dir=output_dir / "tts_cache")
    tts_calls, stt_calls = _estimate_api_calls(cache)
    planned_calls = tts_calls + stt_calls
    print(
        f"Planned API requests: TTS={tts_calls} after cache + "
        f"STT={stt_calls} = {planned_calls}; max_calls={args.max_calls}"
    )
    logger.info(
        "Demo plan: %d sentences × %d conditions | "
        "uncached TTS=%d, STT=%d, total=%d | budget=%d",
        n_sentences, n_conditions, tts_calls, stt_calls, planned_calls,
        args.max_calls,
    )
    if planned_calls > args.max_calls:
        raise SystemExit(
            f"Planned API requests ({planned_calls}) exceed "
            f"max_calls ({args.max_calls}); no provider was initialized."
        )

    budget = BudgetGuard(max_calls=args.max_calls)
    provider = GnaniProvider(max_attempts=1)

    results: list[CaseResult] = []
    t_start = time.monotonic()

    for sentence in SENTENCES:
        for condition in CONDITIONS:
            logger.info(
                "Running: %s / %s", sentence["id"], condition
            )
            try:
                result = run_case(
                    sentence_id=sentence["id"],
                    text=sentence["text"],
                    language=sentence["language"],
                    reference=sentence["reference"],
                    accepted_references=sentence.get("accepted_references"),
                    expected_entities=sentence.get("expected_entities", []),
                    condition=condition,
                    tts_model=TTS_MODEL,
                    tts_voice=TTS_VOICE,
                    stt_model=STT_MODEL,
                    stt_format=STT_FORMAT,
                    provider=provider,
                    budget=budget,
                    cache=cache,
                    output_dir=output_dir,
                )
                results.append(result)
            except BudgetExceeded as e:
                logger.error("BUDGET EXCEEDED — stopping early: %s", e)
                print(f"\n⛔  {e}\n")
                break
        else:
            continue
        break  # outer loop

    elapsed = time.monotonic() - t_start
    logger.info("All cases done in %.1f s. %s", elapsed, budget.summary())

    # Print summary table
    _print_table(results)
    print(HONESTY_NOTE)
    print()

    # Write results.json
    results_path = output_dir / "results.json"
    payload = {
        "_honesty_note": HONESTY_NOTE,
        "run_config": {
            "tts_model": TTS_MODEL,
            "tts_voice": TTS_VOICE,
            "stt_model": STT_MODEL or "prisma-v2.5 (default)",
            "stt_format": STT_FORMAT,
            "conditions": CONDITIONS,
            "max_calls": args.max_calls,
        },
        "budget": {
            "used": budget.used,
            "max": budget.max_calls,
            "tts_calls": budget.tts_calls,
            "stt_calls": budget.stt_calls,
        },
        "api_call_estimate": {
            "tts_after_cache": tts_calls,
            "stt": stt_calls,
            "total": planned_calls,
        },
        "elapsed_s": round(elapsed, 2),
        "summary": {
            "amount_accuracy_by_condition": _amount_accuracy_by_condition(results),
            "by_condition": _summary_by_condition(results),
        },
        "results": [r.to_dict() for r in results],
    }
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Results written to %s", results_path)
    print(f"✅  Results saved to: {results_path}")
    print(f"🎵  Degraded WAVs in:  {output_dir / 'audio'}")


if __name__ == "__main__":
    main()

