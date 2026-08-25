"""Tests for GET /search (normalized text matching)."""


def test_search_by_name(client, seeded_db):
    """Matching on item name (case-insensitive) returns the correct item."""
    today = seeded_db["today"].strftime("%Y-%m-%d")
    resp = client.get("/search/", params={"q": "chicken", "date": today})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["item"]["name"] == "Grilled Chicken"


def test_search_by_category(client, seeded_db):
    """Matching on category string returns all items in that category."""
    today = seeded_db["today"].strftime("%Y-%m-%d")
    resp = client.get("/search/", params={"q": "entrees", "date": today})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    names = {r["item"]["name"] for r in data}
    assert "Grilled Chicken" in names
    assert "Pasta Marinara" in names


def test_search_case_insensitive(client, seeded_db):
    """Search is case-insensitive."""
    today = seeded_db["today"].strftime("%Y-%m-%d")
    resp = client.get("/search/", params={"q": "PASTA", "date": today})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["item"]["name"] == "Pasta Marinara"


def test_search_no_match(client, seeded_db):
    """A query that matches nothing returns an empty list."""
    today = seeded_db["today"].strftime("%Y-%m-%d")
    resp = client.get("/search/", params={"q": "xyznotareal dish", "date": today})
    assert resp.status_code == 200
    assert resp.json() == []


def test_search_result_includes_context(client, seeded_db):
    """Each search result includes hall_name, date, and meal context."""
    today = seeded_db["today"].strftime("%Y-%m-%d")
    resp = client.get("/search/", params={"q": "broccoli", "date": today})
    assert resp.status_code == 200
    result = resp.json()[0]
    assert "hall_name" in result
    assert "date" in result
    assert "meal" in result
    assert result["hall_name"] == "251 North"
    assert result["meal"] == "lunch"


def test_search_filter_by_hall(client, seeded_db):
    """Combining q with hall_id restricts results to that hall."""
    hall_id = seeded_db["halls"][0].id  # 251 North
    today = seeded_db["today"].strftime("%Y-%m-%d")
    # "entrees" exists in both halls; filtering to hall1 returns only Grilled Chicken
    resp = client.get("/search/", params={"q": "entrees", "hall_id": hall_id, "date": today})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["item"]["name"] == "Grilled Chicken"


def test_search_missing_q_returns_422(client):
    """Omitting the required q parameter returns HTTP 422."""
    resp = client.get("/search/")
    assert resp.status_code == 422
