"""SQLAlchemy ORM models for UMD Dining.

Schema:
  Hall          — a dining location (251 North, The Diner, etc.)
  DiningPeriod  — a meal window (breakfast/lunch/dinner/late_night) on a specific date at a hall
  MenuItem      — a single dish served in a dining period, with dietary flags preserved
"""

from datetime import date as Date

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship

import enum


class MealPeriod(str, enum.Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    late_night = "late_night"


class Base(DeclarativeBase):
    pass


class Hall(Base):
    """A UMD campus dining hall."""

    __tablename__ = "halls"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False, unique=True)
    location = Column(String(255), nullable=True)
    # JSON-encoded weekday hours, e.g. '{"monday": "7am-10pm", ...}'
    schedule_json = Column(Text, nullable=True)

    periods = relationship("DiningPeriod", back_populates="hall", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Hall id={self.id} name={self.name!r}>"


class DiningPeriod(Base):
    """A meal window (breakfast/lunch/dinner/late_night) on a specific date at a hall."""

    __tablename__ = "dining_periods"

    id = Column(Integer, primary_key=True, index=True)
    hall_id = Column(Integer, ForeignKey("halls.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    meal = Column(Enum(MealPeriod), nullable=False)

    __table_args__ = (
        UniqueConstraint("hall_id", "date", "meal", name="uq_hall_date_meal"),
    )

    hall = relationship("Hall", back_populates="periods")
    items = relationship("MenuItem", back_populates="period", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<DiningPeriod hall_id={self.hall_id} date={self.date} meal={self.meal}>"


class MenuItem(Base):
    """A single dish offered during a dining period."""

    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    period_id = Column(Integer, ForeignKey("dining_periods.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    category = Column(String(120), nullable=True)
    description = Column(Text, nullable=True)

    # Dietary flags — preserved from the UMD dining API labels
    is_vegan = Column(Boolean, default=False, nullable=False)
    is_vegetarian = Column(Boolean, default=False, nullable=False)
    is_gluten_free = Column(Boolean, default=False, nullable=False)
    is_halal = Column(Boolean, default=False, nullable=False)
    contains_nuts = Column(Boolean, default=False, nullable=False)

    period = relationship("DiningPeriod", back_populates="items")

    def __repr__(self) -> str:
        return f"<MenuItem id={self.id} name={self.name!r}>"
