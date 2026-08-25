"""Tests for GET /halls and GET /halls/{id}."""

import pytest


def test_list_halls_empty(client):
    """Returns an empty list when no halls are in the database."""
    resp = client.get("/halls/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_halls_returns_all(client, seeded_db):
    """Returns all seeded halls ordered by name."""
    resp = client.get("/halls/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Ordered by name — "251 North" < "The Diner"
    assert data[0]["name"] == "251 North"
    assert data[1]["name"] == "The Diner"


def test_list_halls_response_shape(client, seeded_db):
    """Each hall object has id, name, location, and schedule_json fields."""
    resp = client.get("/halls/")
    hall = resp.json()[0]
    assert "id" in hall
    assert "name" in hall
    assert "location" in hall
    assert "schedule_json" in hall


def test_get_hall_by_id(client, seeded_db):
    """Returns the correct hall for a valid ID."""
    hall_id = seeded_db["halls"][0].id
    resp = client.get(f"/halls/{hall_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == hall_id
    assert data["name"] == "251 North"
    assert data["location"] == "251 North Campus Dr"


def test_get_hall_not_found(client, seeded_db):
    """Returns 404 for an ID that doesn't exist."""
    resp = client.get("/halls/99999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_health_check(client):
    """GET /health returns 200 with status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
