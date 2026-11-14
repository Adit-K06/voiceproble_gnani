"""Provider abstraction for VoiceProbe.

Each provider implements two methods:
  tts(text, voice, model, language) -> bytes  (WAV bytes)
  stt(wav_path, language_code, model, format) -> dict  {"transcript": str, "model": str}

GnaniProvider wraps the real SDK with:
  - Disk cache keyed by hash(text+voice+model) so the same sentence is never billed twice.
  - BudgetGuard that aborts when max_calls is hit.

FakeProvider is used in all tests and never touches the network.
"""

