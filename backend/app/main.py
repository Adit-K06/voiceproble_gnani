"""FastAPI application entry point for VoiceProbe."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from app.api.me import router as me_router
from app.core.config import settings
from app.api.auth import router as auth_router
from app.core.crypto import validate_production_secrets

VERSION = "0.1.0"
_BACKEND_DIR = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    validate_production_secrets()
    yield


app = FastAPI(title="VoiceProbe", version=VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
app.include_router(auth_router)
app.include_router(me_router)


@app.exception_handler(RequestValidationError)
async def sanitize_auth_validation_errors(
    request: Request,
    exc: RequestValidationError,
):
    if request.url.path in {
        "/auth/signup",
        "/auth/login",
        "/me/gnani-key",
        "/me/api-keys",
    }:
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid credentials payload."},
        )
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": VERSION}


app.mount(
    "/",
    StaticFiles(directory=_BACKEND_DIR / "static", html=True),
    name="static",
)
