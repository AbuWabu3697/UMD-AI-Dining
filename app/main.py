"""FastAPI application entry point.

Routers:
  /halls   — dining hall list and detail
  /menus   — menu items filtered by hall, date, and meal period
  /search  — normalized text search across menu items

Run with:
  uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.routers import halls, menus, search

app = FastAPI(
    title="UMD Dining API",
    description=(
        "REST API over UMD campus dining data. "
        "Scraped from dining.umd.edu, normalized, and stored in Postgres."
    ),
    version="1.0.0",
)

app.include_router(halls.router, prefix="/halls", tags=["halls"])
app.include_router(menus.router, prefix="/menus", tags=["menus"])
app.include_router(search.router, prefix="/search", tags=["search"])


@app.get("/health", tags=["meta"])
def health_check() -> dict:
    """Liveness check — returns 200 when the app is running."""
    return {"status": "ok"}
