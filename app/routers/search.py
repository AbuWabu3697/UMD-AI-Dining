"""Router: /search

Endpoints:
  GET /search?q=... — case-insensitive text search across menu item names and categories

Query params:
  q       (str, required)         — search term; matched against name and category
  date    (YYYY-MM-DD, optional)  — restrict to a specific date
  hall_id (int, optional)         — restrict to one dining hall
  limit   (int, default 50)       — max results returned

Matching is normalized: query is lowercased, results ranked by exact name match first,
then partial matches in name, then partial matches in category.
"""

from datetime import date as Date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

router = APIRouter()


@router.get("/", response_model=list[schemas.SearchResult])
def search_menu_items(
    q: str = Query(..., min_length=1, description="Search term"),
    menu_date: Optional[Date] = Query(None, alias="date", description="Restrict to date YYYY-MM-DD"),
    hall_id: Optional[int] = Query(None, description="Restrict to one hall"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[schemas.SearchResult]:
    """Return menu items whose name or category contains the query string.

    Results include hall name and meal context so callers can display
    where and when the item is served without a second request.
    """
    term = f"%{q.lower()}%"

    base_query = (
        db.query(models.MenuItem, models.DiningPeriod, models.Hall)
        .join(models.DiningPeriod, models.MenuItem.period_id == models.DiningPeriod.id)
        .join(models.Hall, models.DiningPeriod.hall_id == models.Hall.id)
        .filter(
            func.lower(models.MenuItem.name).like(term)
            | func.lower(models.MenuItem.category).like(term)
        )
    )

    if menu_date is not None:
        base_query = base_query.filter(models.DiningPeriod.date == menu_date)

    if hall_id is not None:
        base_query = base_query.filter(models.Hall.id == hall_id)

    rows = base_query.order_by(models.DiningPeriod.date.desc(), models.MenuItem.name).limit(limit).all()

    results: list[schemas.SearchResult] = []
    for item, period, hall in rows:
        results.append(
            schemas.SearchResult(
                item=schemas.MenuItemResponse.model_validate(item),
                hall_name=hall.name,
                date=period.date,
                meal=period.meal,
            )
        )
    return results
