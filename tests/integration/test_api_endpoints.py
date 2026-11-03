"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from apps.api.main import app


@pytest.fixture(scope="module")
def client():
    """Context manager triggers FastAPI lifespan startup & teardown."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "visual-memory-ai-api"


def test_overview_stats_endpoint(client):
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert "total_sightings" in data
    assert "unique_objects_count" in data


def test_cameras_endpoint(client):
    res = client.get("/api/cameras")
    assert res.status_code == 200
    cams = res.json()
    assert len(cams) >= 1
    assert cams[0]["camera_id"] == "camera_01"


def test_query_endpoint(client):
    res = client.post("/api/query", json={"question": "What objects have you seen?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data


def test_timeline_endpoint(client):
    res = client.get("/api/timeline?limit=10")
    assert res.status_code == 200
    events = res.json()
    assert isinstance(events, list)


def test_observability_endpoint(client):
    res = client.get("/api/observability")
    assert res.status_code == 200
    obs = res.json()
    assert "cpu_usage_pct" in obs
    assert "workers" in obs
    assert len(obs["workers"]) >= 5
