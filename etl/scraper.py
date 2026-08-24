"""ETL scraper: fetches dining data from the UMD dining services API.

UMD Dining exposes a public JSON API at:
  https://dining.umd.edu/locations/

This module fetches hall metadata and per-date menu data,
then normalizes it into dicts matching the ORM schema.

Normalized output shapes:

  HallData = {
      "name": str,
      "location": str | None,
      "schedule_json": str | None,   # JSON-encoded dict of weekday -> hours
  }

  MenuItemData = {
      "hall_name": str,
      "date": str,                   # YYYY-MM-DD
      "meal": str,                   # breakfast | lunch | dinner | late_night
      "name": str,
      "category": str | None,
      "description": str | None,
      "is_vegan": bool,
      "is_vegetarian": bool,
      "is_gluten_free": bool,
      "is_halal": bool,
      "contains_nuts": bool,
  }
"""

import json
import logging
from datetime import date, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

# UMD Dining public API base
_BASE = "https://dining.umd.edu"

# Meal period labels as returned by the UMD API, mapped to normalized names
_MEAL_MAP: dict[str, str] = {
    "Breakfast": "breakfast",
    "Brunch": "breakfast",
    "Lunch": "lunch",
    "Dinner": "dinner",
    "Late Night": "late_night",
    "Late-Night": "late_night",
}

# Dietary flag label strings used in the UMD API response
_DIETARY_FLAGS = {
    "vegan": "is_vegan",
    "vegetarian": "is_vegetarian",
    "gluten free": "is_gluten_free",
    "gluten-free": "is_gluten_free",
    "halal": "is_halal",
    "contains nuts": "contains_nuts",
    "tree nuts": "contains_nuts",
}

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json", "User-Agent": "UMD-AI-Dining-ETL/1.0"})


# ---------------------------------------------------------------------------
# Known UMD dining halls — used as fallback when the API listing is unavailable
# ---------------------------------------------------------------------------

_FALLBACK_HALLS: list[dict[str, Any]] = [
    {"name": "251 North", "location": "251 North Campus Dr", "schedule_json": None},
    {"name": "South Campus Dining", "location": "Cambridge Community Center", "schedule_json": None},
    {"name": "The Diner", "location": "Ellicott Community Center", "schedule_json": None},
    {"name": "Yahentamitsi Dining Hall", "location": "Yahentamitsi Building", "schedule_json": None},
]


def fetch_halls() -> list[dict[str, Any]]:
    """Return a list of HallData dicts.

    Attempts to hit the UMD dining locations endpoint; falls back to a
    static list when the API is unreachable (e.g. network-restricted CI).
    """
    try:
        resp = _SESSION.get(f"{_BASE}/locations/", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        halls = []
        for loc in data.get("locations", []):
            schedule = loc.get("hours")
            halls.append(
                {
                    "name": loc.get("name", "Unknown"),
                    "location": loc.get("address"),
                    "schedule_json": json.dumps(schedule) if schedule else None,
                }
            )
        if halls:
            return halls
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not reach UMD dining API (%s); using deterministic fallback", exc)

    return _FALLBACK_HALLS


def _parse_dietary(labels: list[str]) -> dict[str, bool]:
    """Map a list of dietary label strings to flag fields."""
    flags: dict[str, bool] = {
        "is_vegan": False,
        "is_vegetarian": False,
        "is_gluten_free": False,
        "is_halal": False,
        "contains_nuts": False,
    }
    for label in labels:
        key = _DIETARY_FLAGS.get(label.lower().strip())
        if key:
            flags[key] = True
    return flags


def fetch_menu_for_date(hall_name: str, target_date: date) -> list[dict[str, Any]]:
    """Fetch and normalize menu items for one hall on one date.

    Returns a list of MenuItemData dicts. Returns an empty list on
    network errors or when no menu is published for that date.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    # UMD dining API path convention (subject to change with site updates)
    slug = hall_name.lower().replace(" ", "-").replace("'", "")
    url = f"{_BASE}/menus/{slug}/{date_str}/"

    items: list[dict[str, Any]] = []
    try:
        resp = _SESSION.get(url, timeout=10)
        if resp.status_code == 404:
            return items
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Menu fetch failed for %s on %s: %s", hall_name, date_str, exc)
        return items

    for meal_block in data.get("menu", []):
        raw_meal = meal_block.get("meal_period", "")
        meal = _MEAL_MAP.get(raw_meal)
        if meal is None:
            logger.debug("Skipping unknown meal period %r for %s", raw_meal, hall_name)
            continue

        for category_block in meal_block.get("categories", []):
            category = category_block.get("name")
            for dish in category_block.get("items", []):
                dietary_labels = [d.get("label", "") for d in dish.get("dietary", [])]
                flags = _parse_dietary(dietary_labels)
                items.append(
                    {
                        "hall_name": hall_name,
                        "date": date_str,
                        "meal": meal,
                        "name": dish.get("name", "Unknown item"),
                        "category": category,
                        "description": dish.get("description"),
                        **flags,
                    }
                )

    return items


def scrape(days_ahead: int = 7) -> tuple[list[dict], list[dict]]:
    """Scrape all halls for today through today + days_ahead.

    Returns (halls, menu_items) where each is a list of normalized dicts.
    """
    halls = fetch_halls()
    today = date.today()
    dates = [today + timedelta(days=d) for d in range(days_ahead)]

    all_items: list[dict] = []
    for hall in halls:
        for target_date in dates:
            items = fetch_menu_for_date(hall["name"], target_date)
            all_items.extend(items)
            logger.info("Fetched %d items for %s on %s", len(items), hall["name"], target_date)

    return halls, all_items
