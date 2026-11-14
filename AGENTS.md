# VoiceProbe: agent rules

PRODUCT: VoiceProbe is a test lab for Indic speech AI on phone-quality audio. A suite has
sentences with known text and entities. For each case and condition: Gnani Timbre (TTS)
speaks it -> we degrade it like a phone call -> Gnani Prisma (STT) transcribes -> we score WER
and entity accuracy. We compare configs (Prisma v2.5 vs v3.0, modes, keywords) and recommend
the best. HONESTY: audio is synthetic and noise simulated, so results are RELATIVE comparisons,
never real-world accuracy. Every report, README and UI page must say so.
Contest: Gnani AI Great Indian AI Internship Challenge; submission closes 18:00 IST,
15 Nov 2026 (verify on the live page).

STACK (decided): Python, FastAPI, SQLAlchemy 2.0. SQLite locally, Postgres in production via
DATABASE_URL. No Docker, no Redis. Background job worker inside the app with job state in
the DB (resumable). Auth: argon2 + JWT, plus API keys for the CLI. Frontend: plain
HTML/CSS/JS in backend/static served by FastAPI. Each user stores their OWN Gnani key,
encrypted (Fernet, SECRET_KEY from env). Production target: free hosting.

GNANI SDK (package gnani-vachana): STT: from gnani.stt import GnaniSTTClient;
transcribe(path, language_code="hi-IN") -> dict with "transcript" and "model" (default
prisma-v2.5); model="prisma-v3.0" is opt-in; REST clips < 30 s. TTS: from gnani.tts import
GnaniTTSClient, AudioConfig; synthesize(text, voice=, model=, audio_config=AudioConfig(
sample_rate=24000, encoding="linear_pcm", container="wav")) -> WAV bytes.
UNVERIFIED: per-call credit cost, rate limits, v3.0 modes/keywords parameter names, whether
credits cover v3.0 and timbre-v2.5. Inspect the installed package before using any option;
report what you could not verify. Never invent API parameters.

RULES
- Never read, print, open or commit .env. Never ask me to paste secrets.
- Ask my approval before: installing packages, deleting files, git push, and any command
  that calls the Gnani API (show the call count). Always keep the max_calls budget guard.
- Tests use FakeProvider and never touch the real API.
- Synthetic data only: no real phone numbers, Aadhaar, PAN, account numbers or recordings.
- One task per session. Before finishing: run tests, tell me exactly how to verify, stop.
- Commit-sized changes. Explain non-obvious choices in docs/DECISIONS.md.
- Keep it simple; few dependencies; no speculative features.