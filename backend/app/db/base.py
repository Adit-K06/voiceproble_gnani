"""SQLAlchemy engine, declarative base, and request-scoped sessions."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, make_url
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all ORM models and Alembic metadata."""


def normalize_database_url(database_url: str) -> str:
    """Use Psycopg 3 explicitly for common hosted PostgreSQL URL prefixes."""
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url[len("postgres://"):]
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url[len("postgresql://"):]
    return database_url


def create_database_engine(database_url: str) -> Engine:
    """Create an engine with SQLite thread and foreign-key settings applied."""
    normalized_url = normalize_database_url(database_url)
    parsed_url: URL = make_url(normalized_url)
    connect_args: dict[str, object] = {}
    engine_options: dict[str, object] = {}

    if parsed_url.get_backend_name() == "sqlite":
        connect_args["check_same_thread"] = False
        if parsed_url.database in (None, "", ":memory:"):
            engine_options["poolclass"] = StaticPool

    engine = create_engine(
        normalized_url,
        connect_args=connect_args,
        **engine_options,
    )

    if parsed_url.get_backend_name() == "sqlite":
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine = create_database_engine(settings.database_url)
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Yield a session for a FastAPI request and always close it afterward."""
    with SessionLocal() as session:
        yield session
