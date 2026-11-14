"""Database layer tests use isolated in-memory SQLite databases only."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base, create_database_engine, normalize_database_url
from app.db.models import Case, Suite, User, UserGnaniKey


@pytest.fixture
def db_session():
    engine = create_database_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    with factory() as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgres://user:pass@db.example/voiceprobe",
            "postgresql+psycopg://user:pass@db.example/voiceprobe",
        ),
        (
            "postgresql://user:pass@db.example/voiceprobe",
            "postgresql+psycopg://user:pass@db.example/voiceprobe",
        ),
        (
            "postgresql+psycopg://user:pass@db.example/voiceprobe",
            "postgresql+psycopg://user:pass@db.example/voiceprobe",
        ),
        ("sqlite:///voiceprobe.db", "sqlite:///voiceprobe.db"),
    ],
)
def test_database_url_normalization(url: str, expected: str) -> None:
    assert normalize_database_url(url) == expected


def test_sqlite_foreign_keys_enabled() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("PRAGMA foreign_keys")) == 1
    finally:
        engine.dispose()


def test_user_email_is_lowercase_and_created_at_is_set(db_session: Session) -> None:
    user = User(email="  Adit@Example.COM ", password_hash="hashed")
    db_session.add(user)
    db_session.commit()

    assert user.email == "adit@example.com"
    assert user.created_at is not None


def test_user_email_database_constraint_enforces_lowercase(db_session: Session) -> None:
    with pytest.raises(IntegrityError):
        db_session.execute(
            text(
                "INSERT INTO users (email, password_hash, is_active) "
                "VALUES ('Mixed@Example.com', 'hashed', 1)"
            )
        )
        db_session.commit()
    db_session.rollback()


def test_foreign_key_columns_have_indexes() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        expected = {
            "api_keys": ("user_id",),
            "user_gnani_keys": ("user_id",),
            "suites": ("user_id",),
            "cases": ("suite_id",),
        }
        for table_name, (column_name,) in expected.items():
            indexed_columns = {
                column
                for index in inspector.get_indexes(table_name)
                for column in index["column_names"]
            }
            assert column_name in indexed_columns
    finally:
        engine.dispose()


def test_unique_user_gnani_key_and_suite_case_cascade(db_session: Session) -> None:
    user = User(email="person@example.com", password_hash="hashed")
    user.gnani_key = UserGnaniKey(ciphertext="encrypted")
    suite = Suite(name="demo", user=user)
    suite.cases.append(Case(text="नमस्ते", language="hi-IN"))
    db_session.add(user)
    db_session.commit()

    db_session.add(UserGnaniKey(user_id=user.id, ciphertext="another"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    suite_id = suite.id
    db_session.execute(delete(Suite).where(Suite.id == suite_id))
    db_session.commit()
    assert db_session.scalar(select(Case).where(Case.suite_id == suite_id)) is None


def test_initial_alembic_migration_runs_on_memory_connection() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    engine = create_database_engine("sqlite:///:memory:")
    try:
        with engine.connect() as connection:
            config = Config(str(backend_dir / "alembic.ini"))
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
            tables = set(inspect(connection).get_table_names())
            inspector = inspect(connection)
            api_key_indexes = inspector.get_indexes("api_keys")
            assert any(index["name"] == "ix_api_keys_prefix" for index in api_key_indexes)
            expected_foreign_keys = {
                "api_keys": ("user_id", "users", "CASCADE"),
                "user_gnani_keys": ("user_id", "users", "CASCADE"),
                "suites": ("user_id", "users", "CASCADE"),
                "cases": ("suite_id", "suites", "CASCADE"),
            }
            for table_name, (
                column_name,
                referred_table,
                ondelete,
            ) in expected_foreign_keys.items():
                foreign_keys = inspector.get_foreign_keys(table_name)
                assert any(
                    foreign_key["constrained_columns"] == [column_name]
                    and foreign_key["referred_table"] == referred_table
                    and foreign_key["options"].get("ondelete") == ondelete
                    for foreign_key in foreign_keys
                )
                indexed_columns = {
                    column
                    for index in inspector.get_indexes(table_name)
                    for column in index["column_names"]
                }
                assert column_name in indexed_columns
        assert {"users", "api_keys", "user_gnani_keys", "suites", "cases", "alembic_version"} <= tables
    finally:
        engine.dispose()
