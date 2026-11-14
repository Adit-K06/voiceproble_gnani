"""Per-user secret and API-key tests use synthetic credentials only."""

from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.api.auth import auth_rate_limiter
from app.core.crypto import (
    SecretDecryptionError,
    decrypt_secret,
    validate_production_secrets,
)
from app.core.security import verify_api_key
from app.db.base import Base, create_database_engine, get_db
from app.db.models import ApiKey, User, UserGnaniKey
from app.main import app
from app.providers.gnani import GnaniProvider, GnaniProviderError, get_gnani_provider_for


class _ASGITestClient:
    """Synchronous facade for the in-process ASGI transport."""

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

    def put(self, path: str, **kwargs: Any) -> Response:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Response:
        return self.request("DELETE", path, **kwargs)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[_ASGITestClient]:
    monkeypatch.setenv("JWT_SECRET", "test-only-jwt-secret-0123456789abcdef")
    monkeypatch.setenv("JWT_EXPIRES_SECONDS", "43200")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
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


def _signup(client: _ASGITestClient, email: str) -> dict[str, str]:
    response = client.post(
        "/auth/signup",
        json={"email": email, "password": "synthetic-test-password"},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_gnani_key_is_encrypted_and_get_returns_only_last_four(
    client: _ASGITestClient,
) -> None:
    headers = _signup(client, "gnani-user@example.com")
    api_key = "synthetic-gnani-key-ab12"

    put_response = client.put("/me/gnani-key", headers=headers, json={"key": api_key})
    assert put_response.status_code == 200
    assert put_response.json() == {"set": True, "last4": "ab12"}
    assert api_key not in put_response.text

    with _test_session() as session:
        record = session.scalar(select(UserGnaniKey))
        assert record is not None
        assert record.ciphertext != api_key
        assert api_key not in record.ciphertext
        assert decrypt_secret(record.ciphertext) == api_key

    get_response = client.get("/me/gnani-key", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json() == {"set": True, "last4": "ab12"}
    assert api_key not in get_response.text
    assert "ciphertext" not in get_response.text


def test_wrong_encryption_key_fails_without_revealing_the_key(
    client: _ASGITestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _signup(client, "wrong-key@example.com")
    api_key = "synthetic-gnani-key-wrong-key"
    assert client.put("/me/gnani-key", headers=headers, json={"key": api_key}).status_code == 200
    with _test_session() as session:
        record = session.scalar(select(UserGnaniKey))
        assert record is not None
        ciphertext = record.ciphertext

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(ciphertext)

    response = client.get("/me/gnani-key", headers=headers)
    assert response.status_code == 503
    assert api_key not in response.text


def test_production_startup_rejects_missing_or_placeholder_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "fallback-secret-is-not-enough")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("ENCRYPTION_KEY", "replace-me")
    with pytest.raises(RuntimeError) as exc_info:
        validate_production_secrets()
    message = str(exc_info.value)
    assert "ENCRYPTION_KEY" in message and "JWT_SECRET" in message
    assert "replace-me" not in message
    assert "fallback-secret-is-not-enough" not in message

    monkeypatch.setenv("JWT_SECRET", "a" * 32)
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        validate_production_secrets()

    monkeypatch.setenv("JWT_SECRET", "production-jwt-secret-0123456789abcdef")
    monkeypatch.setenv(
        "ENCRYPTION_KEY",
        base64.urlsafe_b64encode(bytes(32)).decode("ascii"),
    )
    with pytest.raises(RuntimeError, match="ENCRYPTION_KEY"):
        validate_production_secrets()

    monkeypatch.setenv("JWT_SECRET", "production-jwt-secret-0123456789abcdef")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    validate_production_secrets()


def test_api_key_is_returned_once_hashes_and_revokes(client: _ASGITestClient) -> None:
    headers = _signup(client, "cli-user@example.com")
    create_response = client.post(
        "/me/api-keys", headers=headers, json={"name": "local cli"}
    )
    assert create_response.status_code == 201
    created = create_response.json()
    key = created["key"]
    assert key.startswith("vp_")
    assert created["prefix"] == key[:11]

    with _test_session() as session:
        record = session.get(ApiKey, created["id"])
        assert record is not None
        assert record.key_hash != key
        assert verify_api_key(key, record.key_hash)

    listing = client.get("/me/api-keys", headers=headers)
    assert listing.status_code == 200
    assert listing.json() == [
        {
            "id": created["id"],
            "name": "local cli",
            "prefix": key[:11],
            "last_used_at": None,
        }
    ]
    assert key not in listing.text
    assert "key_hash" not in listing.text

    authenticated = client.get("/auth/me", headers={"X-API-Key": key})
    assert authenticated.status_code == 200
    used_listing = client.get("/me/api-keys", headers=headers)
    assert used_listing.json()[0]["last_used_at"] is not None

    revoked = client.delete(f"/me/api-keys/{created['id']}", headers=headers)
    assert revoked.status_code == 204
    rejected = client.get("/auth/me", headers={"X-API-Key": key})
    assert rejected.status_code == 401


def test_user_cannot_list_or_revoke_another_users_api_keys(
    client: _ASGITestClient,
) -> None:
    user_a_headers = _signup(client, "user-a@example.com")
    user_b_headers = _signup(client, "user-b@example.com")
    user_b_key = client.post(
        "/me/api-keys", headers=user_b_headers, json={"name": "user b cli"}
    ).json()

    user_a_listing = client.get("/me/api-keys", headers=user_a_headers)
    assert user_a_listing.status_code == 200
    assert user_a_listing.json() == []

    revoke = client.delete(
        f"/me/api-keys/{user_b_key['id']}", headers=user_a_headers
    )
    assert revoke.status_code == 404
    user_b_listing = client.get("/me/api-keys", headers=user_b_headers)
    assert [item["id"] for item in user_b_listing.json()] == [user_b_key["id"]]


def test_user_provider_factory_uses_decrypted_key_parameter(
    client: _ASGITestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _signup(client, "provider-user@example.com")
    key = "synthetic-provider-key-1234"
    assert client.put("/me/gnani-key", headers=headers, json={"key": key}).status_code == 200

    received_keys: list[str] = []

    class FakeSDKClient:
        def __init__(self, api_key: str | None = None, **_kwargs: Any):
            assert api_key is not None
            received_keys.append(api_key)

    monkeypatch.setattr("gnani.tts.GnaniTTSClient", FakeSDKClient)
    monkeypatch.setattr("gnani.stt.GnaniSTTClient", FakeSDKClient)
    with _test_session() as session:
        user = session.scalar(
            select(User)
            .options(selectinload(User.gnani_key))
            .where(User.email == "provider-user@example.com")
        )
        assert user is not None
        provider = get_gnani_provider_for(user)
        assert isinstance(provider, GnaniProvider)
    assert received_keys == [key, key]


def test_provider_logs_and_errors_do_not_include_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    key = "synthetic-provider-key-never-log"

    class FailingTTSClient:
        def __init__(self, api_key: str | None = None, **_kwargs: Any):
            self._api_key = api_key

        def synthesize(self, *_args: Any, **_kwargs: Any) -> bytes:
            raise RuntimeError(f"upstream echoed credential {key}")

    class FakeSTTClient:
        def __init__(self, api_key: str | None = None, **_kwargs: Any):
            self._api_key = api_key

    monkeypatch.setattr("gnani.tts.GnaniTTSClient", FailingTTSClient)
    monkeypatch.setattr("gnani.stt.GnaniSTTClient", FakeSTTClient)
    caplog.set_level(logging.DEBUG)
    provider = GnaniProvider(api_key=key, max_attempts=1)

    with pytest.raises(GnaniProviderError) as exc_info:
        provider.synthesize("नमस्ते", voice=key, model="timbre-v2.5", language="hi-IN")
    assert key not in str(exc_info.value)
    assert key not in caplog.text
