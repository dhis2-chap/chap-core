"""fix_hpo_search_space_empty_arrays

Note: This migration is a hack to fix an issue after the generic_migration_script was run on the hpo_search_space and made it into a list.
o

Fixes hpo_search_space field in modeltemplatedb table that was incorrectly
initialized to empty array [] instead of null by the generic migration.

The generic migration defaulted all JSON columns to [] (empty arrays), but
hpo_search_space expects a dictionary or null.

Revision ID: ff2b1bbb8418
Revises: fe59a33965ed
Create Date: 2025-11-12 10:19:41.034518

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "ff2b1bbb8418"
down_revision: Union[str, Sequence[str], None] = "fe59a33965ed"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Fix hpo_search_space: convert empty arrays to null
    # hpo_search_space is Optional[dict] with default=None, so null is appropriate
    # Note: Cast to text for comparison since PostgreSQL doesn't support json = json
    op.execute(
        "UPDATE modeltemplatedb SET hpo_search_space = NULL WHERE hpo_search_space::text = '[]'"
    )

    # Also fix any dict fields that might have been set to [] instead of {}
    # user_options has default_factory=dict, so {} is more appropriate than []
    op.execute(
        "UPDATE modeltemplatedb SET user_options = '{}'::json WHERE user_options::text = '[]'"
    )

    # user_option_values also has default_factory=dict
    op.execute(
        "UPDATE configuredmodeldb SET user_option_values = '{}'::json WHERE user_option_values::text = '[]'"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Reversing this is not really meaningful since [] was incorrect anyway
    # But for completeness, we can reverse the changes
    op.execute(
        "UPDATE modeltemplatedb SET hpo_search_space = '[]'::json WHERE hpo_search_space IS NULL"
    )
    op.execute(
        "UPDATE modeltemplatedb SET user_options = '[]'::json WHERE user_options::text = '{}'"
    )
    op.execute(
        "UPDATE configuredmodeldb SET user_option_values = '[]'::json WHERE user_option_values::text = '{}'"
    )
