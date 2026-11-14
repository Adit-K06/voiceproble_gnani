"""Index API-key prefixes for credential lookup."""

from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0002_api_key_prefix"
down_revision: str | None = "0001_initial"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_api_keys_prefix", "api_keys", ["prefix"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_api_keys_prefix", table_name="api_keys")
