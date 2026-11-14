"""Password hashing and signed access tokens."""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from app.core.config import Settings

_password_hasher = PasswordHasher()
# Verify an equivalent Argon2 hash for unknown users to reduce login timing leaks.
DUMMY_PASSWORD_HASH = _password_hasher.hash("not-a-real-user-password")
API_KEY_PREFIX_LENGTH = 11


class JWTConfigurationError(RuntimeError):
    """Raised when token signing is attempted without a configured secret."""


def hash_password(password: str) -> str:
    """Hash a password using Argon2id with the library's current defaults."""
    return _password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Return whether the password matches a stored Argon2 hash."""
    try:
        return _password_hasher.verify(password_hash, password)
    except VerificationError:
        return False


def password_needs_rehash(password_hash: str) -> bool:
    """Check whether a stored hash should be upgraded to current parameters."""
    try:
        return _password_hasher.check_needs_rehash(password_hash)
    except VerificationError:
        return False


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 digest stored for an API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(api_key: str, key_hash: str) -> bool:
    """Compare an API key with its stored SHA-256 digest in constant time."""
    return hmac.compare_digest(hash_api_key(api_key), key_hash)


def create_access_token(subject: str | int) -> str:
    """Create an HS256 access JWT with the configured lifetime (12 hours by default)."""
    settings = Settings.from_environment()
    jwt_secret = settings.jwt_secret or settings.secret_key
    if not jwt_secret:
        raise JWTConfigurationError("JWT_SECRET or SECRET_KEY must be configured")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(seconds=settings.jwt_expires_seconds),
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, object]:
    """Validate an HS256 token and require its subject and expiry claims."""
    settings = Settings.from_environment()
    jwt_secret = settings.jwt_secret or settings.secret_key
    if not jwt_secret:
        raise JWTConfigurationError("JWT_SECRET or SECRET_KEY must be configured")

    return jwt.decode(
        token,
        jwt_secret,
        algorithms=["HS256"],
        options={"require": ["exp", "sub"]},
    )
