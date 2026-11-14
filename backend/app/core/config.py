"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

_DEV_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)


def _cors_origins_from_environment(env: str) -> tuple[str, ...]:
    configured = os.environ.get("CORS_ORIGINS")
    if configured is None:
        return _DEV_CORS_ORIGINS if env != "production" else ()

    origins = tuple(
        origin.strip().rstrip("/")
        for origin in configured.split(",")
        if origin.strip()
    )
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            "*" in origin
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "CORS_ORIGINS must contain exact http(s) origins, "
                "not wildcards or URLs with paths."
            )
    return origins


@dataclass(frozen=True)
class Settings:
    database_url: str
    secret_key: str
    env: str
    jwt_secret: str = ""
    jwt_expires_seconds: int = 43200
    encryption_key: str = ""
    cors_origins: tuple[str, ...] = _DEV_CORS_ORIGINS

    @classmethod
    def from_environment(cls) -> Settings:
        secret_key = os.environ.get("SECRET_KEY", "")
        env = os.environ.get("ENV", "development").strip().lower()
        return cls(
            database_url=os.environ.get("DATABASE_URL", "sqlite:///./voiceprobe.db"),
            secret_key=secret_key,
            env=env,
            jwt_secret=os.environ.get("JWT_SECRET", ""),
            jwt_expires_seconds=int(os.environ.get("JWT_EXPIRES_SECONDS", "43200")),
            encryption_key=os.environ.get("ENCRYPTION_KEY", ""),
            cors_origins=_cors_origins_from_environment(env),
        )


settings = Settings.from_environment()
