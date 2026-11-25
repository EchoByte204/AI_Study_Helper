"""Track document extraction versions so updates can reindex old PPTs."""

import sqlalchemy as sa
from alembic import op

revision = "0005_ingestion_revision"
down_revision = "0004_retry_empty_figures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("ingestion_version", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("documents", "ingestion_version")
