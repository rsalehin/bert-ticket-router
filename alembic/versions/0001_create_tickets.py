"""Create tickets table.

Revision ID: 0001_create_tickets
Revises:
Create Date: 2026-04-25 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_create_tickets"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tickets",
        sa.Column("id", sa.String(length=26), primary_key=True, nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("intent", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("top_k_json", sa.String(), nullable=False),
        sa.Column("department", sa.String(), nullable=False),
        sa.Column("priority", sa.String(), nullable=False),
        sa.Column("sla_hours", sa.Integer(), nullable=False),
        sa.Column("tags_json", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_tickets_created_at", "tickets", ["created_at"])
    op.create_index("ix_tickets_department", "tickets", ["department"])
    op.create_index("ix_tickets_intent", "tickets", ["intent"])


def downgrade() -> None:
    op.drop_index("ix_tickets_intent", table_name="tickets")
    op.drop_index("ix_tickets_department", table_name="tickets")
    op.drop_index("ix_tickets_created_at", table_name="tickets")
    op.drop_table("tickets")
