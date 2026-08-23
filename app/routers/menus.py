"""Router: /menus

Endpoints:
  GET /menus — menu items filtered by hall_id, date, and meal period

Query params:
  hall_id  (int, optional)       — filter to one dining hall
  date     (YYYY-MM-DD, optional) — filter to a specific date (defaults to today)
  meal     (str, optional)        — one of breakfast / lunch / dinner / late_night
  limit    (int, default 100)     — max items returned
  offset   (int, default 0)       — pagination offset
"""

from datetime import date as Date, date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

router = APIRouter()


@router.get("/", response_model=list[schemas.MenuItemResponse])
def list_menu_items(
    hall_id: Optional[int] = Query(None, description="Filter to a specific hall"),
    menu_date: Optional[Date] = Query(None, alias="date", description="Date in YYYY-MM-DD format"),
    meal: Optional[str] = Query(None, description="breakfast | lunch | dinner | late_night"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[models.MenuItem]:
    """Return menu items, optionally filtered by hall, date, and meal period."""
    if meal is not None:
        try:
            meal_enum = models.MealPeriod(meal)
        except ValueError:
            valid = [m.value for m in models.MealPeriod]
            raise HTTPException(
                status_code=422,
                detail=f"Invalid meal '{meal}'. Valid values: {valid}",
            )
    else:
        meal_enum = None

    target_date = menu_date or date.today()

    query = (
        db.query(models.MenuItem)
        .join(models.DiningPeriod, models.MenuItem.period_id == models.DiningPeriod.id)
        .filter(models.DiningPeriod.date == target_date)
    )

    if hall_id is not None:
        query = query.filter(models.DiningPeriod.hall_id == hall_id)

    if meal_enum is not None:
        query = query.filter(models.DiningPeriod.meal == meal_enum)

    return query.order_by(models.MenuItem.category, models.MenuItem.name).offset(offset).limit(limit).all()
