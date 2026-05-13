"""add_required_user_options_to_model_template

Adds nullable JSON column `required_user_options` to modeltemplatedb. Stores
the JSON-schema top-level `required` array for the template's user options.
NULL means "legacy template predating this column", in which case the
configured-model validator falls back to its old heuristic (infer required
from missing literal "default" keys in the properties dict).

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "modeltemplatedb",
        sa.Column("required_user_options", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("modeltemplatedb", "required_user_options")
