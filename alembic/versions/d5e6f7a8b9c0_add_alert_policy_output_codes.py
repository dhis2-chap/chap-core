"""Store output codes on alert policies.

Existing policies get "0" for no alert and "1", "2", ... in level order.

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
"""

import sqlalchemy as sa

from alembic import op
from chap_core.database.migration_helpers import has_column

revision = "d5e6f7a8b9c0"
down_revision = "c4d5e6f7a8b9"
branch_labels = None
depends_on = None

policies = sa.table("alertpolicy", sa.column("id", sa.Integer()), sa.column("levels", sa.JSON()))


def upgrade() -> None:
    if not has_column("alertpolicy", "no_alert_output_code"):
        op.add_column("alertpolicy", sa.Column("no_alert_output_code", sa.String(), nullable=True))
    # Startup's generic migration may already have added the column.
    op.execute(
        "UPDATE alertpolicy SET no_alert_output_code = '0' "
        "WHERE no_alert_output_code IS NULL OR no_alert_output_code = ''"
    )
    op.alter_column("alertpolicy", "no_alert_output_code", existing_type=sa.String(), nullable=False)

    connection = op.get_bind()
    for policy_id, levels in connection.execute(sa.select(policies.c.id, policies.c.levels)):
        if levels and any("output_code" not in level for level in levels):
            for index, level in enumerate(levels, start=1):
                level.setdefault("output_code", str(index))
            connection.execute(policies.update().where(policies.c.id == policy_id).values(levels=levels))


def downgrade() -> None:
    connection = op.get_bind()
    for policy_id, levels in connection.execute(sa.select(policies.c.id, policies.c.levels)):
        if levels:
            for level in levels:
                level.pop("output_code", None)
            connection.execute(policies.update().where(policies.c.id == policy_id).values(levels=levels))
    op.drop_column("alertpolicy", "no_alert_output_code")
