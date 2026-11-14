"""Re-score saved results offline; this script never initializes a provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.scoring import wer_best_with_flags


def main() -> None:
    results_path = _BACKEND / "data" / "sample_results.json"
    suite_path = _BACKEND / "data" / "suite.json"

    results_data = json.loads(results_path.read_text(encoding="utf-8"))
    suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
    sentence_by_id = {
        sentence["id"]: sentence
        for sentence in suite_data.get("sentences", [])
    }

    print(
        f"{'Sentence':<12} {'Condition':<16} {'Old WER':>8} "
        f"{'New WER':>8} {'Spacing only':>13}"
    )
    print("-" * 64)
    for result in results_data.get("results", []):
        sentence = sentence_by_id.get(result["sentence_id"], {})
        accepted = list(result.get("accepted_references", []))
        for reference in sentence.get("accepted_references", []):
            if reference not in accepted:
                accepted.append(reference)

        reference = sentence.get("text", result["reference"])
        references = [reference, *accepted]
        old_wer = result.get("wer")
        new_wer, spacing_only = wer_best_with_flags(
            references, result["hypothesis"]
        )

        old_display = "n/a" if old_wer is None else f"{old_wer:.4f}"
        print(
            f"{result['sentence_id']:<12} {result['condition']:<16} "
            f"{old_display:>8} {new_wer:>8.4f} {str(spacing_only):>13}"
        )


if __name__ == "__main__":
    main()
