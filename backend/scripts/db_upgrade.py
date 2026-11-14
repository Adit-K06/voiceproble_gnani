"""Apply all Alembic migrations to the configured database."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from alembic import command
from alembic.config import Config


def main() -> None:
    config = Config(str(_BACKEND / "alembic.ini"))
    command.upgrade(config, "head")
    print("Database upgraded to Alembic head.")


if __name__ == "__main__":
    main()
