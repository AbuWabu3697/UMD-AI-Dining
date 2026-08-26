# UMD-AI-Dining

REST API over UMD campus dining data. Scrapes dining.umd.edu, normalizes menu items, stores them in Postgres, and exposes them through a FastAPI service.

---

## Pipeline

```
dining.umd.edu  →  etl/scraper.py  →  etl/loader.py  →  Postgres  →  FastAPI
```

1. **Scrape** — `etl/scraper.py` fetches hall metadata and per-date menus from the UMD dining JSON API. When the API is unreachable (e.g. network-restricted CI), it falls back to a deterministic list of known halls.
2. **Normalize** — meal periods are mapped to `breakfast | lunch | dinner | late_night`; dietary labels (`vegan`, `halal`, `gluten free`, etc.) are extracted into explicit boolean columns.
3. **Load** — `etl/loader.py` upserts halls and dining periods, clears existing items for each `(hall, date, meal)` combination, then inserts fresh rows. Each run is idempotent.
4. **Serve** — FastAPI reads from Postgres and exposes three routers documented below.

---

## Endpoints

All endpoints return JSON. Interactive docs at `/docs` after `uvicorn app.main:app --reload`.

### `GET /halls`

List all dining halls ordered by name.

**Response:**
```json
[
  {
    "id": 1,
    "name": "251 North",
    "location": "251 North Campus Dr",
    "schedule_json": null
  }
]
```

### `GET /halls/{hall_id}`

Single hall detail. Returns 404 if the ID does not exist.

### `GET /menus`

Menu items for a given date, with optional filters.

| Parameter | Type    | Default | Description |
|-----------|---------|---------|-------------|
| `date`    | string  | today   | YYYY-MM-DD |
| `hall_id` | integer | —       | Filter to one hall |
| `meal`    | string  | —       | `breakfast` / `lunch` / `dinner` / `late_night` |
| `limit`   | integer | 100     | Max items (1–500) |
| `offset`  | integer | 0       | Pagination offset |

**Example:**
```
GET /menus?date=2026-08-25&hall_id=1&meal=lunch
```

### `GET /search`

Case-insensitive text search across menu item names and categories. Results include hall and meal context so you can display where and when the item is served without a second request.

| Parameter | Type    | Required | Description |
|-----------|---------|----------|-------------|
| `q`       | string  | yes      | Search term |
| `date`    | string  | no       | Restrict to a date |
| `hall_id` | integer | no       | Restrict to a hall |
| `limit`   | integer | no       | Max results (1–200, default 50) |

**Example response:**
```json
[
  {
    "item": {
      "id": 12,
      "name": "Grilled Chicken",
      "category": "Entrees",
      "is_vegan": false,
      "is_gluten_free": true,
      "is_halal": true,
      ...
    },
    "hall_name": "251 North",
    "date": "2026-08-25",
    "meal": "lunch"
  }
]
```

---

## Data model

```
Hall
  id, name, location, schedule_json

DiningPeriod  (FK → Hall)
  id, hall_id, date, meal [breakfast|lunch|dinner|late_night]

MenuItem  (FK → DiningPeriod)
  id, period_id, name, category, description
  is_vegan, is_vegetarian, is_gluten_free, is_halal, contains_nuts
```

---

## Running locally

### With Docker (recommended)

```bash
docker compose up --build
```

This starts Postgres, runs Alembic migrations, runs the ETL scraper (7 days ahead), then starts the API on port 8000.

### Without Docker

**Prerequisites:** Python 3.12, a running Postgres instance.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# fill in values as needed
export DATABASE_URL=postgresql://user:pass@localhost:5432/umd_dining

alembic upgrade head
python -m etl.run --days 7
uvicorn app.main:app --reload
```

---

## Tests

Tests use an in-memory SQLite database — no Postgres required.

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Coverage: `/halls` list and detail, `/menus` date and meal filtering, `/search` text matching and hall scoping.

---

## Environment variables

| Variable       | Required | Default                                           | Description |
|----------------|----------|---------------------------------------------------|-------------|
| `DATABASE_URL` | no       | `postgresql://dining:dining@localhost:5432/umd_dining` | Postgres DSN |
| `APP_ENV`      | no       | `development`                                     | `development` or `production` |
| `LOG_LEVEL`    | no       | `info`                                            | uvicorn log level |

---

## CI

GitHub Actions runs `pytest` on every push and pull request to `main`. The workflow spins up a Postgres service container, applies migrations, then runs the test suite against it.

See `.github/workflows/test.yml`.
