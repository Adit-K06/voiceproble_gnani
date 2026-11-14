"""Authenticated per-user secrets and CLI API key management."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.crypto import (
    CryptoConfigurationError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from app.core.security import API_KEY_PREFIX_LENGTH, hash_api_key
from app.db.base import get_db
from app.db.models import ApiKey, User, UserGnaniKey

router = APIRouter(tags=["me"])


class GnaniKeyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: SecretStr


class ApiKeyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)


def _encryption_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="Secret encryption is not configured.")


def _safe_last4(key: str) -> str:
    # Never reveal the entire secret, even if a caller supplied a very short key.
    return key[-4:] if len(key) > 4 else "****"


@router.put("/me/gnani-key")
def put_gnani_key(
    payload: GnaniKeyPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    key = payload.key.get_secret_value()
    if not key:
        raise HTTPException(status_code=422, detail="Invalid key payload.")
    try:
        ciphertext = encrypt_secret(key)
    except CryptoConfigurationError:
        raise _encryption_unavailable() from None

    record = db.scalar(select(UserGnaniKey).where(UserGnaniKey.user_id == user.id))
    if record is None:
        record = UserGnaniKey(user_id=user.id, ciphertext=ciphertext)
        db.add(record)
    else:
        record.ciphertext = ciphertext
        record.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"set": True, "last4": _safe_last4(key)}


@router.get("/me/gnani-key")
def get_gnani_key(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    record = db.scalar(select(UserGnaniKey).where(UserGnaniKey.user_id == user.id))
    if record is None:
        return {"set": False}
    try:
        key = decrypt_secret(record.ciphertext)
    except (CryptoConfigurationError, SecretDecryptionError):
        raise _encryption_unavailable() from None
    return {"set": True, "last4": _safe_last4(key)}


@router.delete("/me/gnani-key", status_code=204)
def delete_gnani_key(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    record = db.scalar(select(UserGnaniKey).where(UserGnaniKey.user_id == user.id))
    if record is not None:
        db.delete(record)
        db.commit()
    return Response(status_code=204)


@router.post("/me/api-keys", status_code=201)
def create_api_key(
    payload: ApiKeyPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Invalid API key name.")

    key = f"vp_{secrets.token_urlsafe(32)}"
    prefix = key[:API_KEY_PREFIX_LENGTH]
    record = ApiKey(
        user_id=user.id,
        name=name,
        prefix=prefix,
        key_hash=hash_api_key(key),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {
        "id": record.id,
        "name": record.name,
        "key": key,
        "prefix": record.prefix,
        "created_at": record.created_at,
    }


@router.get("/me/api-keys")
def list_api_keys(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    records = db.scalars(
        select(ApiKey)
        .where(ApiKey.user_id == user.id, ApiKey.revoked_at.is_(None))
        .order_by(ApiKey.id)
    ).all()
    return [
        {
            "id": record.id,
            "name": record.name,
            "prefix": record.prefix,
            "last_used_at": record.last_used_at,
        }
        for record in records
    ]


@router.delete("/me/api-keys/{key_id}", status_code=204)
def revoke_api_key(
    key_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    record = db.scalar(
        select(ApiKey).where(
            ApiKey.id == key_id,
            ApiKey.user_id == user.id,
            ApiKey.revoked_at.is_(None),
        )
    )
    if record is None:
        raise HTTPException(status_code=404, detail="API key not found.")
    record.revoked_at = datetime.now(timezone.utc)
    db.commit()
    return Response(status_code=204)
