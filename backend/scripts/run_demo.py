"""VoiceProbe demo runner.

Run from backend/ with:
    python -m scripts.run_demo [--max-calls N] [--output-dir PATH]

This script runs 5 synthetic Hindi/Hinglish banking sentences through all
registered phone conditions using the Gnani Prisma STT default model
(prisma-v2.5) in verbatim mode.

The script estimates first-attempt TTS + STT requests before starting. Up to
three attempts are made for transient errors, and every attempt counts against
the hard --max-calls budget (default 60).

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
def _as_row(result: CaseResult | dict[str, Any]) -> dict[str, Any]:
    return result.to_dict() if isinstance(result, CaseResult) else result


def _row_status(row: dict[str, Any]) -> str:
    return str(row.get("status", "ok" if "wer" in row else "error"))


def _unit_key(
    sentence_id: str,
    condition: str,
    tts_model: str,
    tts_voice: str,
    stt_model: str | None,
    stt_format: str,
) -> str:
    dimensions = (sentence_id, condition, tts_model, tts_voice, stt_model or "default", stt_format)
    return "|".join(dimensions)


def _amount_accuracy_by_condition(
    results: list[CaseResult | dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    from collections import defaultdict

    by_condition: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        row = _as_row(result)
        if _row_status(row) == "ok" and row.get("amount_correct") is not None:
            by_condition[str(row["condition"])].append(bool(row["amount_correct"]))

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


def _summary_by_condition(
    results: list[CaseResult | dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        condition_rows = [
            _as_row(result)
            for result in results
            if _as_row(result).get("condition") == condition
        ]
        rows = [row for row in condition_rows if _row_status(row) == "ok"]
        amount_flags = [
            bool(row["amount_correct"])
            for row in rows
            if row.get("amount_correct") is not None
        ]
        stt_latencies = [float(row["stt_latency_s"]) for row in rows if row.get("stt_latency_s") is not None]
        summary[condition] = {
            "count": len(condition_rows),
            "ok_count": len(rows),
            "error_count": len(condition_rows) - len(rows),
            "mean_wer": (
                sum(float(row["wer"]) for row in rows) / len(rows) if rows else None
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


def _read_journal(path: Path) -> dict[str, dict[str, Any]]:
    """Load latest row per unit; the append-only journal remains the history."""
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as journal:
        for line_number, line in enumerate(journal, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                key = row.get("unit_key") or _unit_key(
                    str(row.get("sentence_id", "")),
                    str(row.get("condition", "")),
                    str(row.get("tts_model", TTS_MODEL)),
                    str(row.get("tts_voice", TTS_VOICE)),
                    row.get("stt_model", STT_MODEL),
                    str(row.get("stt_format", STT_FORMAT)),
                )
                row["unit_key"] = key
                row.setdefault("status", "ok" if "wer" in row else "error")
                latest[str(key)] = row
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Ignoring malformed results.jsonl line %d", line_number)
    return latest


def _append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as journal:
        journal.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        journal.write("\n")
        journal.flush()
        os.fsync(journal.fileno())


def _error_row(sentence: dict[str, Any], condition: str, exc: Exception) -> dict[str, Any]:
    status = getattr(exc, "http_status", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    error_type = (
        str(getattr(exc, "source_error_type", type(exc).__name__))
        if type(exc).__name__ == "GnaniProviderError"
        else type(exc).__name__
    )
    return {
        "status": "error",
        "sentence_id": str(sentence["id"]),
        "condition": condition,
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
        "stt_model": STT_MODEL or "prisma-v2.5 (default)",
        "stt_format": STT_FORMAT,
        "error_type": error_type,
        "http_status": status,
        "_honesty_note": HONESTY_NOTE,
    }


def run_units(
    *,
    provider,
    budget: BudgetGuard,
    cache: TTSCache,
    output_dir: Path,
    sentences: list[dict[str, Any]] = SENTENCES,
    conditions: list[str] = CONDITIONS,
    retry_errors: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run pending units and persist each terminal outcome immediately."""
    journal_path = output_dir / "results.jsonl"
    latest = _read_journal(journal_path)
    for sentence in sentences:
        for condition in conditions:
            key = _unit_key(
                str(sentence["id"]), condition, TTS_MODEL, TTS_VOICE,
                STT_MODEL, STT_FORMAT,
            )
            previous = latest.get(key)
            if previous and _row_status(previous) == "ok":
                continue
            if previous and _row_status(previous) == "error" and not retry_errors:
                continue
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
            except BudgetExceeded as exc:
                logger.error("Budget exhausted; stopping before an unbudgeted attempt: %s", exc)
                return latest
            except Exception as exc:
                row = _error_row(sentence, condition, exc)
                row["unit_key"] = key
                _append_journal(journal_path, row)
                latest[key] = row
                logger.error(
                    "Unit failed and was journaled | sentence=%s condition=%s "
                    "error_type=%s http_status=%s",
                    sentence["id"], condition, row["error_type"], row["http_status"],
                )
                continue

            row = result.to_dict()
            row["unit_key"] = key
            _append_journal(journal_path, row)
            latest[key] = row
    return latest


def _estimate_api_calls(
    cache: TTSCache,
    pending_units: list[tuple[dict[str, Any], str]] | None = None,
) -> tuple[int, int]:
    """Return first-attempt counts for pending units, after TTS cache hits."""
    tts_missing_by_key: dict[str, bool] = {}
    units = (
        pending_units
        if pending_units is not None
        else [
            (sentence, condition)
            for sentence in SENTENCES
            for condition in CONDITIONS
        ]
    )
    for sentence, _condition in units:
        key = cache.key(sentence["text"], TTS_VOICE, TTS_MODEL, TTS_LANGUAGE)
        if key not in tts_missing_by_key:
            tts_missing_by_key[key] = (
                cache.get(sentence["text"], TTS_VOICE, TTS_MODEL, TTS_LANGUAGE)
                is None
            )
    return sum(tts_missing_by_key.values()), len(units)


def _print_table(results: list[CaseResult | dict[str, Any]]) -> None:
    rows = [_as_row(result) for result in results]
    header = f"{'ID':<10} {'Condition':<14} {'Status':<7} {'WER':>6} {'Spacing':>8} {'Amt':>5}  {'TTS lat':>8} {'STT lat':>8}  {'Details'}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        status = _row_status(r)
        spacing = str(r.get("spacing_only", False)) if status == "ok" else "n/a"
        wer = f"{float(r['wer']):.3f}" if status == "ok" and r.get("wer") is not None else "n/a"
        amount_correct = r.get("amount_correct") if status == "ok" else None
        amount_flag = "n/a" if amount_correct is None else "OK" if amount_correct else "MISS"
        tts_lat = r.get("tts_latency_s")
        stt_lat = r.get("stt_latency_s")
        details = (
            f"{r.get('error_type', 'error')} HTTP {r.get('http_status')}"
            if status == "error"
            else str(r.get("hypothesis", "")).replace("\n", " ")[:40]
        )
        print(
            f"{str(r.get('sentence_id', '')):<10} {str(r.get('condition', '')):<14} "
            f"{status:<7} {wer:>6} {spacing:>8} {amount_flag:>5}  "
            f"{('n/a' if tts_lat is None else f'{float(tts_lat):.2f}s'):>8} "
            f"{('n/a' if stt_lat is None else f'{float(stt_lat):.2f}s'):>8}  {details}"
        )
    print(sep)
    print()
    print("Summary by condition:")
    print(
        f"{'Condition':<16} {'Cases':>5} {'Errors':>6} {'Mean WER':>10} "
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
            f"{condition:<16} {metrics['count']:>5} {metrics['error_count']:>6} {mean_wer:>10} "
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
        help="Root directory for audio files, results.json and results.jsonl",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Retry units whose latest journal entry has status=error",
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
    journal_path = output_dir / "results.jsonl"
    latest = _read_journal(journal_path)
    pending_units: list[tuple[dict[str, Any], str]] = []
    for sentence in SENTENCES:
        for condition in CONDITIONS:
            key = _unit_key(
                sentence["id"], condition, TTS_MODEL, TTS_VOICE,
                STT_MODEL, STT_FORMAT,
            )
            previous = latest.get(key)
            if previous and _row_status(previous) == "ok":
                continue
            if previous and _row_status(previous) == "error" and not args.retry_errors:
                continue
            pending_units.append((sentence, condition))

    tts_calls, stt_calls = _estimate_api_calls(cache, pending_units)
    planned_calls = tts_calls + stt_calls
    print(
        f"Planned API requests: TTS={tts_calls} after cache + "
        f"STT={stt_calls} = {planned_calls} first attempts; retries are budgeted "
        f"separately (up to 3 attempts/request); max_calls={args.max_calls}"
    )
    logger.info(
        "Demo plan: %d sentences × %d conditions | "
        "uncached TTS=%d, STT=%d, total=%d | budget=%d",
        n_sentences, n_conditions, tts_calls, stt_calls, planned_calls,
        args.max_calls,
    )
    budget = BudgetGuard(max_calls=args.max_calls)
    t_start = time.monotonic()
    elapsed = 0.0
    try:
        if planned_calls > args.max_calls:
            raise SystemExit(
                f"Planned first attempts ({planned_calls}) exceed max_calls "
                f"({args.max_calls}); no provider was initialized."
            )
        # Load developer-provided environment only when the real CLI is run.
        from dotenv import load_dotenv
        load_dotenv(_BACKEND / ".env")
        provider = GnaniProvider()
        logger.info(
            "Running %d sentences × %d conditions; %d units are pending",
            n_sentences, n_conditions, len(pending_units),
        )
        run_units(
            provider=provider,
            budget=budget,
            cache=cache,
            output_dir=output_dir,
            retry_errors=args.retry_errors,
        )
    except KeyboardInterrupt:
        print("\nRun interrupted. Saving the completed journal entries.")
    finally:
        elapsed = time.monotonic() - t_start
        logger.info("Run stopped after %.1f s. %s", elapsed, budget.summary())
        # The append-only JSONL is authoritative, including after interruption.
        latest = _read_journal(journal_path)
        rows = list(latest.values())
        _print_table(rows)
        print(HONESTY_NOTE)
        print()
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
                "retry_errors": args.retry_errors,
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
                "first_attempt_total": planned_calls,
                "max_attempts_per_request": 3,
            },
            "elapsed_s": round(elapsed, 2),
            "summary": {
                "amount_accuracy_by_condition": _amount_accuracy_by_condition(rows),
                "by_condition": _summary_by_condition(rows),
            },
            "results": rows,
        }
        results_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Results written to %s", results_path)
        print(f"✅  Results saved to: {results_path}")
        print(f"🧾  Result journal: {journal_path}")
        print(f"🎵  Degraded WAVs in:  {output_dir / 'audio'}")


if __name__ == "__main__":
    main()

