"""Persist structured tutor answers for chat history."""

import sqlalchemy as sa
from alembic import op

revision = "0006_chat_history_answers"
down_revision = "0005_ingestion_revision"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_messages",
        sa.Column("structured_answer", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_messages", "structured_answer")
