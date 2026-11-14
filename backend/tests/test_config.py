"""Configuration policy tests for exact CORS allowlists."""

from __future__ import annotations

import pytest

from app.core.config import Settings


def test_development_cors_defaults_are_localhost_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    origins = Settings.from_environment().cors_origins

    assert origins
    assert "*" not in origins
    assert all(
        origin.startswith("http://localhost:")
        or origin.startswith("http://127.0.0.1:")
        for origin in origins
    )


def test_production_cors_origins_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://voiceprobe.example.com, https://admin.example.com/",
    )

    settings = Settings.from_environment()

    assert settings.cors_origins == (
        "https://voiceprobe.example.com",
        "https://admin.example.com",
    )


def test_wildcard_cors_origin_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("CORS_ORIGINS", "*")

    with pytest.raises(ValueError, match=r"exact http\(s\) origins"):
        Settings.from_environment()
