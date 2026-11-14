"""FastAPI application entry point for VoiceProbe."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

VERSION = "0.1.0"
_BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_url: str
    secret_key: str
    env: str

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            database_url=os.environ.get("DATABASE_URL", "sqlite:///./voiceprobe.db"),
            secret_key=os.environ.get("SECRET_KEY", ""),
            env=os.environ.get("ENV", "development"),
        )


settings = Settings.from_environment()
app = FastAPI(title="VoiceProbe", version=VERSION)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": VERSION}


app.mount(
    "/",
    StaticFiles(directory=_BACKEND_DIR / "static", html=True),
    name="static",
)
