"""Retry visual extraction for documents that have no indexed figures."""

from alembic import op

revision = "0004_retry_empty_figures"
down_revision = "0003_visual_reindex"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE documents SET visual_indexed = FALSE WHERE figure_count = 0")


def downgrade() -> None:
    # Figure indexing is data state; a downgrade must not prevent a later retry.
    pass
