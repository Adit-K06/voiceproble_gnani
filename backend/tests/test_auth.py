"""Authentication tests use in-memory SQLite and never call external APIs."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import time
from collections.abc import Iterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.auth import auth_rate_limiter
from app.core.config import settings
from app.core.security import hash_api_key, hash_password
from app.db.base import Base, create_database_engine, get_db
from app.db.models import ApiKey, User
from app.main import app


class _ASGITestClient:
    """Synchronous test facade over HTTPX's in-process ASGI transport."""

    def request(self, method: str, path: str, **kwargs: Any) -> Response:
        async def send() -> Response:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs: Any) -> Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Response:
        return self.request("POST", path, **kwargs)

    def options(self, path: str, **kwargs: Any) -> Response:
        return self.request("OPTIONS", path, **kwargs)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[_ASGITestClient]:
    monkeypatch.setenv("JWT_SECRET", "test-only-jwt-secret-0123456789abcdef")
    monkeypatch.setenv("JWT_EXPIRES_SECONDS", "43200")
    engine = create_database_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)

    def override_get_db():
        with factory() as session:
            yield session

    auth_rate_limiter.reset()
    app.dependency_overrides[get_db] = override_get_db
    yield _ASGITestClient()
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(engine)
    engine.dispose()
    auth_rate_limiter.reset()


@contextmanager
def _test_session() -> Iterator[Session]:
    dependency = app.dependency_overrides[get_db]()
    session = next(dependency)
    try:
        yield session
    finally:
        dependency.close()


def test_signup_login_and_get_me(client: _ASGITestClient) -> None:
    signup = client.post(
        "/auth/signup",
        json={"email": "  Adit@Example.com ", "password": "long-enough-test-password"},
    )
    assert signup.status_code == 201
    signup_data = signup.json()
    assert signup_data["user"]["email"] == "adit@example.com"
    assert "password" not in signup_data and "password_hash" not in signup_data
    assert "long-enough-test-password" not in signup.text

    login = client.post(
        "/auth/login",
        json={"email": "ADIT@example.com", "password": "long-enough-test-password"},
    )
    assert login.status_code == 200
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": signup_data["user"]["id"],
        "email": "adit@example.com",
        "is_active": True,
    }


def test_duplicate_email_is_generic_conflict(client: _ASGITestClient) -> None:
    payload = {"email": "person@example.com", "password": "long-enough-password"}
    assert client.post("/auth/signup", json=payload).status_code == 201
    duplicate = client.post("/auth/signup", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "Unable to create account."}


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": "long-enough-password"},
        {"email": "valid@example.com", "password": "short"},
    ],
)
def test_signup_rejects_invalid_email_or_short_password(
    client: _ASGITestClient, payload: dict[str, str]
) -> None:
    response = client.post("/auth/signup", json=payload)
    assert response.status_code == 422
    assert payload["password"] not in response.text


def test_auth_openapi_documents_credentials_without_echoing_invalid_password(
    client: _ASGITestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    signup_schema = schema["paths"]["/auth/signup"]["post"]["requestBody"]
    login_schema = schema["paths"]["/auth/login"]["post"]["requestBody"]
    assert "application/json" in signup_schema["content"]
    assert "application/json" in login_schema["content"]

    response = client.post(
        "/auth/signup",
        json={"email": [], "password": "never-echo-this-password"},
    )
    assert response.status_code == 422
    assert "never-echo-this-password" not in response.text


def test_wrong_password_and_unknown_email_have_same_response(client: _ASGITestClient) -> None:
    client.post(
        "/auth/signup",
        json={"email": "known@example.com", "password": "long-enough-password"},
    )
    wrong_password = client.post(
        "/auth/login",
        json={"email": "known@example.com", "password": "different-password"},
    )
    unknown_email = client.post(
        "/auth/login",
        json={"email": "unknown@example.com", "password": "different-password"},
    )
    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()


def test_expired_token_is_rejected(
    client: _ASGITestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JWT_EXPIRES_SECONDS", "1")
    signup = client.post(
        "/auth/signup",
        json={"email": "expire@example.com", "password": "long-enough-password"},
    )
    token = signup.json()["access_token"]
    time.sleep(1.1)
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials."}


def test_tampered_token_is_rejected(client: _ASGITestClient) -> None:
    signup = client.post(
        "/auth/signup",
        json={"email": "tamper@example.com", "password": "long-enough-password"},
    )
    token = signup.json()["access_token"]
    header, payload, signature = token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    tampered = f"{header}.{payload}.{replacement}{signature[1:]}"

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {tampered}"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials."}


def test_rate_limit_returns_429(client: _ASGITestClient) -> None:
    for _ in range(10):
        assert client.post("/auth/login", json={}).status_code == 422
    limited = client.post("/auth/login", json={})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_cors_allows_only_configured_origins(client: _ASGITestClient) -> None:
    if settings.cors_origins:
        allowed = client.options(
            "/auth/me",
            headers={
                "Origin": settings.cors_origins[0],
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,x-api-key",
            },
        )
        assert allowed.headers["access-control-allow-origin"] == settings.cors_origins[0]

    rejected = client.options(
        "/auth/me",
        headers={
            "Origin": "https://unapproved.example.invalid",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in rejected.headers


def test_api_key_authentication_and_inactive_user_rejection(
    client: _ASGITestClient,
) -> None:
    with _test_session() as session:
        user = User(email="cli@example.com", password_hash=hash_password("unused-password"))
        session.add(user)
        session.flush()
        key = "vp_test_cli_key"
        session.add(
            ApiKey(
                user_id=user.id,
                name="test",
                prefix=key[:11],
                key_hash=hash_api_key(key),
            )
        )
        session.commit()
        user_id = user.id

    login = client.post(
        "/auth/login",
        json={"email": "cli@example.com", "password": "unused-password"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    response = client.get("/auth/me", headers={"X-API-Key": "vp_test_cli_key"})
    assert response.status_code == 200
    assert response.json()["id"] == user_id

    with _test_session() as session:
        user = session.scalar(select(User).where(User.id == user_id))
        assert user is not None
        user.is_active = False
        session.commit()

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401

    response = client.get("/auth/me", headers={"X-API-Key": "vp_test_cli_key"})
    assert response.status_code == 401
