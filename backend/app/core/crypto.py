"""Fernet encryption for application secrets stored in the database."""

from __future__ import annotations

import base64

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings

_PLACEHOLDER_VALUES = {
    "changeme",
    "change-me",
    "change_me",
    "dev",
    "development",
    "dev-secret",
    "example",
    "example-secret",
    "placeholder",
    "replace-me",
    "replace_me",
    "secret",
    "test",
    "test-secret",
    "your-secret-key",
    "your-encryption-key",
    "your-jwt-secret",
}
_PLACEHOLDER_PREFIXES = (
    "change-",
    "change_",
    "dev-",
    "dev_",
    "example-",
    "example_",
    "replace-",
    "replace_",
    "test-",
    "test_",
    "your-",
    "your_",
)


class CryptoConfigurationError(RuntimeError):
    """Raised when secret encryption cannot be configured safely."""


class SecretDecryptionError(RuntimeError):
    """Raised when ciphertext cannot be decrypted by the configured key."""


def _is_placeholder(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    return (
        not normalized
        or normalized in _PLACEHOLDER_VALUES
        or "placeholder" in normalized
        or (normalized.startswith("<") and normalized.endswith(">"))
        or normalized.startswith(_PLACEHOLDER_PREFIXES)
    )


def _fernet() -> Fernet:
    encryption_key = Settings.from_environment().encryption_key
    if _is_placeholder(encryption_key):
        raise CryptoConfigurationError("ENCRYPTION_KEY is not configured.")
    try:
        return Fernet(encryption_key.encode("ascii"))
    except (UnicodeEncodeError, ValueError):
        raise CryptoConfigurationError(
            "ENCRYPTION_KEY must be a valid Fernet key."
        ) from None


def encrypt_secret(value: str) -> str:
    """Encrypt a secret and return its ASCII Fernet token."""
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a stored Fernet token without exposing crypto internals."""
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except CryptoConfigurationError:
        raise
    except (InvalidToken, UnicodeDecodeError, ValueError):
        raise SecretDecryptionError(
            "Stored secret could not be decrypted; check ENCRYPTION_KEY."
        ) from None


def validate_production_secrets(settings: Settings | None = None) -> None:
    """Fail app startup when production secrets are absent or placeholders."""
    selected = settings or Settings.from_environment()
    if selected.env.strip().lower() != "production":
        return

    encryption_key = selected.encryption_key
    jwt_secret = selected.jwt_secret
    missing_or_placeholder = [
        name
        for name, value in (
            ("ENCRYPTION_KEY", encryption_key),
            ("JWT_SECRET", jwt_secret),
        )
        if _is_placeholder(value)
    ]
    if missing_or_placeholder:
        names = ", ".join(missing_or_placeholder)
        raise RuntimeError(
            f"Production startup requires non-placeholder secrets: {names}."
        )

    try:
        jwt_secret_bytes = jwt_secret.encode("utf-8")
        if len(jwt_secret_bytes) < 32 or len(set(jwt_secret_bytes)) < 8:
            raise RuntimeError(
                "Production JWT_SECRET must be at least 32 bytes and not a repeated value."
            )
        Fernet(encryption_key.encode("ascii"))
        raw_encryption_key = base64.urlsafe_b64decode(encryption_key.encode("ascii"))
    except (UnicodeEncodeError, ValueError):
        raise RuntimeError(
            "Production ENCRYPTION_KEY must be a valid Fernet key."
        ) from None
    if len(set(raw_encryption_key)) < 8:
        raise RuntimeError(
            "Production ENCRYPTION_KEY must be a randomly generated Fernet key."
        )
