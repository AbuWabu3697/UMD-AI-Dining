"""Initial schema: halls, dining_periods, menu_items.

Revision ID: 001
Revises:
Create Date: 2026-08-22

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "halls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column("schedule_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_halls_id", "halls", ["id"], unique=False)

    op.create_table(
        "dining_periods",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("hall_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column(
            "meal",
            sa.Enum("breakfast", "lunch", "dinner", "late_night", name="mealperiod"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["hall_id"], ["halls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hall_id", "date", "meal", name="uq_hall_date_meal"),
    )
    op.create_index("ix_dining_periods_id", "dining_periods", ["id"], unique=False)
    op.create_index("ix_dining_periods_hall_id", "dining_periods", ["hall_id"], unique=False)
    op.create_index("ix_dining_periods_date", "dining_periods", ["date"], unique=False)

    op.create_table(
        "menu_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("period_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=120), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_vegan", sa.Boolean(), nullable=False),
        sa.Column("is_vegetarian", sa.Boolean(), nullable=False),
        sa.Column("is_gluten_free", sa.Boolean(), nullable=False),
        sa.Column("is_halal", sa.Boolean(), nullable=False),
        sa.Column("contains_nuts", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["period_id"], ["dining_periods.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_menu_items_id", "menu_items", ["id"], unique=False)
    op.create_index("ix_menu_items_period_id", "menu_items", ["period_id"], unique=False)
    op.create_index("ix_menu_items_name", "menu_items", ["name"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_menu_items_name", table_name="menu_items")
    op.drop_index("ix_menu_items_period_id", table_name="menu_items")
    op.drop_index("ix_menu_items_id", table_name="menu_items")
    op.drop_table("menu_items")

    op.drop_index("ix_dining_periods_date", table_name="dining_periods")
    op.drop_index("ix_dining_periods_hall_id", table_name="dining_periods")
    op.drop_index("ix_dining_periods_id", table_name="dining_periods")
    op.drop_table("dining_periods")

    op.execute("DROP TYPE IF EXISTS mealperiod")

    op.drop_index("ix_halls_id", table_name="halls")
    op.drop_table("halls")
