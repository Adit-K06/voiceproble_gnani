"""Signup, login, and current-user authentication endpoints."""

from __future__ import annotations

import math
import re
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from jwt import InvalidTokenError
from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    API_KEY_PREFIX_LENGTH,
    JWTConfigurationError,
    create_access_token,
    decode_access_token,
    hash_password,
    password_needs_rehash,
    hash_api_key,
    verify_password,
)
from app.db.base import get_db
from app.db.models import ApiKey, User

router = APIRouter(prefix="/auth", tags=["auth"])
_EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class CredentialsPayload(BaseModel):
    """Request schema for signup and login; password values stay redacted."""

    model_config = ConfigDict(extra="forbid")

    email: str
    password: SecretStr


class TokenBucketRateLimiter:
    """Thread-safe, process-local token buckets keyed by client IP."""

    def __init__(self, capacity: int = 10, refill_period_seconds: float = 60.0):
        self.capacity = float(capacity)
        self.refill_rate = capacity / refill_period_seconds
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def consume(self, key: str) -> int | None:
        now = time.monotonic()
        with self._lock:
            tokens, updated_at = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated_at) * self.refill_rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return max(1, math.ceil((1.0 - tokens) / self.refill_rate))
            self._buckets[key] = (tokens - 1.0, now)
            return None

    def reset(self) -> None:
        """Clear buckets; primarily useful for isolated tests and local operations."""
        with self._lock:
            self._buckets.clear()


auth_rate_limiter = TokenBucketRateLimiter(capacity=10, refill_period_seconds=60)


def _client_ip(request: Request) -> str:
    # Do not trust forwarded headers here; only use the direct ASGI peer address.
    return request.client.host if request.client is not None else "unknown"


def enforce_auth_rate_limit(request: Request) -> None:
    retry_after = auth_rate_limiter.consume(_client_ip(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many authentication attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _require_jwt_configuration() -> None:
    settings = Settings.from_environment()
    if not settings.jwt_secret and not settings.secret_key:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured.",
        )


def _read_credentials(payload: CredentialsPayload) -> tuple[str, str]:
    normalized_email = payload.email.strip().lower()
    password = payload.password.get_secret_value()
    if len(normalized_email) > 320 or not _EMAIL_SHAPE.fullmatch(normalized_email):
        raise HTTPException(status_code=422, detail="Invalid email or password.")
    if len(password) < 10:
        raise HTTPException(
            status_code=422,
            detail="Password must be at least 10 characters.",
        )
    return normalized_email, password


def _auth_response(user: User) -> dict[str, object]:
    try:
        token = create_access_token(user.id)
    except JWTConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured.",
        ) from None
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email},
    }


@router.post("/signup", status_code=201, dependencies=[Depends(enforce_auth_rate_limit)])
def signup(
    payload: CredentialsPayload,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    _require_jwt_configuration()
    email, password = _read_credentials(payload)

    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise HTTPException(status_code=409, detail="Unable to create account.")

    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Unable to create account.") from None
    db.refresh(user)
    return _auth_response(user)


@router.post("/login", dependencies=[Depends(enforce_auth_rate_limit)])
def login(
    payload: CredentialsPayload,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    _require_jwt_configuration()
    email, password = _read_credentials(payload)
    user = db.scalar(select(User).where(User.email == email))

    password_hash = user.password_hash if user is not None else DUMMY_PASSWORD_HASH
    valid_password = verify_password(password_hash, password)
    if user is None or not valid_password or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        db.commit()

    return _auth_response(user)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _user_from_api_key(api_key: str, db: Session) -> User | None:
    if len(api_key) > 4096:
        return None

    prefix = api_key[:API_KEY_PREFIX_LENGTH]
    key_hash = hash_api_key(api_key)
    # Identity is unknown before credential resolution; select only the owner id.
    user_id = db.scalar(
        select(ApiKey.user_id).where(
            ApiKey.prefix == prefix,
            ApiKey.key_hash == key_hash,
            ApiKey.revoked_at.is_(None),
        )
    )
    if user_id is None:
        return None

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None

    result = db.execute(
        update(ApiKey)
        .where(
            ApiKey.user_id == user.id,
            ApiKey.prefix == prefix,
            ApiKey.key_hash == key_hash,
            ApiKey.revoked_at.is_(None),
        )
        .values(last_used_at=datetime.now(timezone.utc))
    )
    if result.rowcount != 1:
        db.rollback()
        return None
    db.commit()
    return user


def get_current_user(
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> User:
    """Authenticate the request with a bearer JWT or a stored API key."""
    if authorization is not None:
        scheme, separator, credential = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not credential:
            raise _unauthorized()
        try:
            subject = decode_access_token(credential).get("sub")
        except JWTConfigurationError:
            raise HTTPException(
                status_code=503,
                detail="Authentication is not configured.",
            ) from None
        except InvalidTokenError:
            raise _unauthorized() from None
        try:
            user_id = int(subject)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise _unauthorized() from None
        user = db.get(User, user_id)
        if user is None or not user.is_active:
            raise _unauthorized()
        return user

    if x_api_key is not None:
        user = _user_from_api_key(x_api_key, db)
        if user is not None:
            return user

    raise _unauthorized()


@router.get("/me")
def me(user: User = Depends(get_current_user)) -> dict[str, object]:
    return {"id": user.id, "email": user.email, "is_active": user.is_active}
