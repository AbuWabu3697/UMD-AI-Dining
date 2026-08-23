"""Router: /halls

Endpoints:
  GET /halls          — list all dining halls
  GET /halls/{hall_id} — single hall detail
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

router = APIRouter()


@router.get("/", response_model=list[schemas.HallResponse])
def list_halls(db: Session = Depends(get_db)) -> list[models.Hall]:
    """Return all dining halls ordered by name."""
    return db.query(models.Hall).order_by(models.Hall.name).all()


@router.get("/{hall_id}", response_model=schemas.HallResponse)
def get_hall(hall_id: int, db: Session = Depends(get_db)) -> models.Hall:
    """Return a single dining hall by ID. Returns 404 if not found."""
    hall = db.query(models.Hall).filter(models.Hall.id == hall_id).first()
    if hall is None:
        raise HTTPException(status_code=404, detail=f"Hall {hall_id} not found")
    return hall
