"""Track one-time figure indexing for documents already in the library."""

import sqlalchemy as sa
from alembic import op

revision = "0003_visual_reindex"
down_revision = "0002_figures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("visual_indexed", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("documents", "visual_indexed")
