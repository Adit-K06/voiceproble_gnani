"""One-attempt Gnani TTS/STT smoke check.

Run from backend/ with:
    python -m scripts.smoke_gnani

This sends at most one TTS request and one STT request. It reads
GNANI_API_KEY from the process environment and does not load or print .env.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.budget import BudgetGuard
from app.providers.gnani import GnaniProvider


def main() -> None:
    budget = BudgetGuard(max_calls=2)
    provider = GnaniProvider(max_attempts=1)

    budget.charge("tts")
    synthesized = provider.synthesize(
        "नमस्ते, यह वॉइसप्रोब का परीक्षण है।",
        voice="Pranav",
        model="timbre-v2.5",
        language="hi-IN",
    )

    with tempfile.TemporaryDirectory(prefix="voiceprobe-smoke-") as temp_dir:
        wav_path = Path(temp_dir) / "smoke.wav"
        wav_path.write_bytes(synthesized.audio_bytes)

        budget.charge("stt")
        transcription = provider.transcribe(wav_path, language="hi-IN")

    print(f"Transcript: {transcription.text}")
    print(f"STT model: {transcription.model}")
    print(f"TTS model: {synthesized.model}")
    print(f"TTS latency: {synthesized.latency_s:.3f}s")
    print(f"STT latency: {transcription.latency_s:.3f}s")
    print(budget.summary())


if __name__ == "__main__":
    main()
