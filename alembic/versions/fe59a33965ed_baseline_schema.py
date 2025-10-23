"""baseline_schema

This is a baseline migration that marks the starting point for Alembic migrations.
It is a no-op migration that doesn't change the schema - all tables are already
managed by the custom migration system in database.py.

Future schema changes should be made through Alembic migrations after this baseline.

Revision ID: fe59a33965ed
Revises:
Create Date: 2025-10-23 11:56:30.686702

"""
from typing import Sequence, Union



# revision identifiers, used by Alembic.
revision: str = 'fe59a33965ed'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
