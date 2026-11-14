"""Alembic environment wired to VoiceProbe settings and ORM metadata."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from app.core.config import settings
from app.db.base import Base, create_database_engine, normalize_database_url
from app.db import models  # noqa: F401 - register model metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

configured_url = config.get_main_option("sqlalchemy.url")
database_url = normalize_database_url(configured_url or settings.database_url)
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    provided_connection: Connection | None = config.attributes.get("connection")
    if provided_connection is not None:
        _run_migrations_on_connection(provided_connection)
        return

    connectable = create_database_engine(database_url)
    try:
        with connectable.connect() as connection:
            _run_migrations_on_connection(connection)
    finally:
        connectable.dispose()


def _run_migrations_on_connection(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
