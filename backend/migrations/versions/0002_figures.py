"""Index extracted PDF and PowerPoint figures."""

import sqlalchemy as sa
from alembic import op
from app.config import get_settings
from pgvector.sqlalchemy import Vector

revision = "0002_figures"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    dimensions = get_settings().embedding_dimensions
    op.add_column("documents", sa.Column("figure_count", sa.Integer(), nullable=False, server_default="0"))
    op.create_table(
        "figures",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("figure_index", sa.Integer(), nullable=False),
        sa.Column("media_type", sa.String(length=100), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dimensions), nullable=False),
    )
    op.create_index("ix_figures_document_id", "figures", ["document_id"])
    op.create_index("ix_figures_document_page", "figures", ["document_id", "page_number"])
    op.create_index(
        "ix_figures_embedding_hnsw",
        "figures",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_figures_embedding_hnsw", table_name="figures")
    op.drop_index("ix_figures_document_page", table_name="figures")
    op.drop_index("ix_figures_document_id", table_name="figures")
    op.drop_table("figures")
    op.drop_column("documents", "figure_count")
