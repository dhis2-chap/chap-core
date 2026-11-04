"""Rename forecast alerts to signals and make the dissemination gate explicit.

Revision ID: a1c3e5f70926
Revises: b64628109792
"""

import sqlalchemy as sa
from alembic import op

from chap_core.database.migration_helpers import foreign_key_name, has_column, has_table

revision = "a1c3e5f70926"
down_revision = "b64628109792"
branch_labels = None
depends_on = None


def rename_column(table: str, old: str, new: str) -> None:
    # Startup may already have added the new column with placeholder values.
    if has_column(table, new):
        op.drop_column(table, new)
    op.alter_column(table, old, new_column_name=new)


def rename_levels(table_name: str, old: str, new: str) -> None:
    table = sa.table(table_name, sa.column("id", sa.Integer()), sa.column("levels", sa.JSON()))
    conn = op.get_bind()
    for policy_id, levels in conn.execute(sa.select(table.c.id, table.c.levels)).all():
        if levels is None:
            continue
        for level in levels:
            if old in level:
                level[new] = level.pop(old)
        conn.execute(table.update().where(table.c.id == policy_id).values(levels=levels))


def rename_signal_indexes(
    old: str, new: str, old_policy: str, new_policy: str, old_status: str, new_status: str
) -> None:
    op.drop_index(f"ix_{old}_{old_policy}", table_name=new)
    op.drop_index(f"ix_{old}_{old_status}", table_name=new)
    op.create_index(f"ix_{new}_{new_policy}", new, [new_policy])
    op.create_index(f"ix_{new}_{new_status}", new, [new_status])
    op.drop_constraint(f"uq_{old}_prediction_org_unit_period", new, type_="unique")
    op.create_unique_constraint(
        f"uq_{new}_prediction_org_unit_period", new, ["prediction_id", "org_unit", "time_period"]
    )


def upgrade() -> None:
    if not has_table("alertpolicy"):
        return

    # create_all runs before Alembic. Remove only empty tables it created under
    # the new names, then rename the originals so IDs and references survive.
    conn = op.get_bind()
    for table_name in ("signal", "thresholdpolicy"):
        if has_table(table_name):
            table = sa.table(table_name, sa.column("id"))
            if conn.execute(sa.select(table.c.id).limit(1)).first() is not None:
                raise RuntimeError(f"Cannot rename alerts: {table_name} already contains data")
    for table_name in ("prediction", "predictionsetup"):
        constraint = foreign_key_name(table_name, "thresholdpolicy")
        if constraint is not None:
            op.drop_constraint(constraint, table_name, type_="foreignkey")
    if has_table("signal"):
        op.drop_table("signal")
    if has_table("thresholdpolicy"):
        op.drop_table("thresholdpolicy")

    op.rename_table("alertpolicy", "thresholdpolicy")
    op.rename_table("alert", "signal")
    for table_name in ("prediction", "predictionsetup", "signal"):
        rename_column(table_name, "alert_policy_id", "threshold_policy_id")
    rename_column("thresholdpolicy", "no_alert_output_code", "no_signal_output_code")
    rename_column("signal", "approved", "release_status")
    rename_column("signal", "approved_by", "reviewed_by")
    rename_column("signal", "approved_at", "reviewed_at")
    rename_levels("thresholdpolicy", "exceedance_threshold", "alarm_threshold")
    op.execute(
        "UPDATE signal SET release_status = CASE release_status "
        "WHEN 'approved' THEN 'released' WHEN 'declined' THEN 'withheld' ELSE release_status END"
    )
    rename_signal_indexes("alert", "signal", "alert_policy_id", "threshold_policy_id", "approved", "release_status")


def downgrade() -> None:
    rename_levels("thresholdpolicy", "alarm_threshold", "exceedance_threshold")
    op.execute(
        "UPDATE signal SET release_status = CASE release_status "
        "WHEN 'released' THEN 'approved' WHEN 'withheld' THEN 'declined' ELSE release_status END"
    )
    op.rename_table("signal", "alert")
    op.rename_table("thresholdpolicy", "alertpolicy")
    for table_name in ("prediction", "predictionsetup", "alert"):
        rename_column(table_name, "threshold_policy_id", "alert_policy_id")
    rename_column("alertpolicy", "no_signal_output_code", "no_alert_output_code")
    rename_column("alert", "release_status", "approved")
    rename_column("alert", "reviewed_by", "approved_by")
    rename_column("alert", "reviewed_at", "approved_at")
    rename_signal_indexes("signal", "alert", "threshold_policy_id", "alert_policy_id", "release_status", "approved")
