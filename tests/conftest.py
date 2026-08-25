"""Pytest fixtures for the UMD Dining API test suite.

Uses an in-memory SQLite database so tests run without a live Postgres instance.
SQLAlchemy models are compatible with SQLite for all operations tested here.

Fixtures:
  engine      — SQLite engine with all tables created
  db_session  — transactional session, rolled back after each test
  client      — TestClient with get_db overridden to use db_session
  seeded_db   — db_session pre-populated with 2 halls, 2 periods, 4 menu items
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import get_db
from app.main import app
from app.models import Base, DiningPeriod, Hall, MealPeriod, MenuItem

_SQLITE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def engine():
    """Create tables once for the test session."""
    eng = create_engine(_SQLITE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture()
def db_session(engine):
    """Yield a session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    """TestClient with get_db overridden to use the test session."""

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def seeded_db(db_session):
    """Seed minimal fixture data: 2 halls, 2 periods, 4 menu items."""
    hall1 = Hall(name="251 North", location="251 North Campus Dr")
    hall2 = Hall(name="The Diner", location="Ellicott Community Center")
    db_session.add_all([hall1, hall2])
    db_session.flush()

    today = date.today()
    period1 = DiningPeriod(hall_id=hall1.id, date=today, meal=MealPeriod.lunch)
    period2 = DiningPeriod(hall_id=hall2.id, date=today, meal=MealPeriod.dinner)
    db_session.add_all([period1, period2])
    db_session.flush()

    items = [
        MenuItem(
            period_id=period1.id,
            name="Grilled Chicken",
            category="Entrees",
            is_vegan=False,
            is_vegetarian=False,
            is_gluten_free=True,
            is_halal=True,
            contains_nuts=False,
        ),
        MenuItem(
            period_id=period1.id,
            name="Roasted Broccoli",
            category="Vegetables",
            is_vegan=True,
            is_vegetarian=True,
            is_gluten_free=True,
            is_halal=True,
            contains_nuts=False,
        ),
        MenuItem(
            period_id=period2.id,
            name="Pasta Marinara",
            category="Entrees",
            is_vegan=True,
            is_vegetarian=True,
            is_gluten_free=False,
            is_halal=True,
            contains_nuts=False,
        ),
        MenuItem(
            period_id=period2.id,
            name="Peanut Butter Cookies",
            category="Desserts",
            is_vegan=False,
            is_vegetarian=True,
            is_gluten_free=False,
            is_halal=False,
            contains_nuts=True,
        ),
    ]
    db_session.add_all(items)
    db_session.flush()

    return {
        "halls": [hall1, hall2],
        "periods": [period1, period2],
        "items": items,
        "today": today,
    }
