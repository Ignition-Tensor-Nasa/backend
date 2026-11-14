"""Run from backend/: python -m pytest tests -q (needs pytest and httpx).

Rows below are synthetic test fixtures, not NASA measurements.
"""

from fastapi.testclient import TestClient
import polars as pl
import pytest

from app.main import app
from app.services import catalogue

URL = "/api/v1/cases/search"
ROWS = [
    # case_id, investigation, fuel, o2_pct, pressure_kpa
    ("T-003", "INV-A", "ethylene", 30.0, 101.3),
    ("T-001", "INV-A", "methane", 21.0, 101.3),
    ("T-002", "INV-A", "methane", 40.0, 50.0),
    ("T-004", "INV-B", "methane", 21.0, None),
    ("T-005", "INV-B", None, None, 101.3),
    ("T-006", "INV-B", "ethylene", float("nan"), 200.0),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "catalogue.parquet"
    pl.DataFrame(
        [(cid, f"Fixture {cid}", inv, fuel, o2, kpa, "not returned")
         for cid, inv, fuel, o2, kpa in ROWS],
        schema=["case_id", "title", "investigation", "fuel", "o2_pct", "pressure_kpa", "note"],
        orient="row",
    ).write_parquet(path)
    monkeypatch.setattr(catalogue, "CATALOGUE_PATH", path)
    catalogue.load_catalogue.cache_clear()
    yield TestClient(app)
    catalogue.load_catalogue.cache_clear()


def ids(client, filters):
    response = client.post(URL, json={"filters": filters})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == len(body["items"])
    return [item["case_id"] for item in body["items"]]


@pytest.mark.parametrize("kwargs", [{}, {"json": {}}, {"json": {"filters": {}}}])
def test_no_filters_returns_everything_ordered_with_nulls(client, kwargs):
    body = client.post(URL, **kwargs).json()
    assert [item["case_id"] for item in body["items"]] == [f"T-00{n}" for n in range(1, 7)]
    assert body["total"] == 6
    assert body["items"][4] == {
        "case_id": "T-005", "title": "Fixture T-005", "investigation": "INV-B",
        "fuel": None, "o2_pct": None, "pressure_kpa": 101.3,
    }
    assert body["items"][5]["o2_pct"] is None  # NaN in the catalogue is unknown


@pytest.mark.parametrize("filters, expected", [
    ({"fuel": ["methane", "ethylene"]}, ["T-001", "T-002", "T-003", "T-004", "T-006"]),
    ({"investigation": ["INV-A"], "fuel": ["methane"]}, ["T-001", "T-002"]),
    ({"o2_pct": {"min": 30}}, ["T-002", "T-003"]),  # open range, inclusive, null/NaN fail
    ({"o2_pct": {"max": 21}}, ["T-001", "T-004"]),
    ({"o2_pct": {"min": 21, "max": 30}}, ["T-001", "T-003", "T-004"]),
    ({"pressure_kpa": {"min": 101.3, "max": 101.3}}, ["T-001", "T-003", "T-005"]),
    ({"fuel": ["methane"], "o2_pct": {"max": 21}, "pressure_kpa": {"min": 100}}, ["T-001"]),
    ({"pressure_kpa": {"min": 5000}}, []),  # outside observed bounds is valid
    ({"investigation": ["INV-B"], "o2_pct": {"min": 30}}, []),
])
def test_filters(client, filters, expected):
    assert ids(client, filters) == expected


@pytest.mark.parametrize("payload", [
    {"filters": {"gravity": ["0g"]}},
    {"filters": {}, "page": 2},
    {"filters": {"fuel": ["kerosene"]}},
    {"filters": {"fuel": ["methane", "Methane"]}},
    {"filters": {"fuel": []}},
    {"filters": {"fuel": "methane"}},
    {"filters": {"o2_pct": {}}},
    {"filters": {"o2_pct": {"min": 30, "max": 21}}},
    {"filters": {"o2_pct": {"max": 101}}},
    {"filters": {"o2_pct": {"min": -1}}},
    {"filters": {"o2_pct": {"min": 21, "step": 1}}},
    {"filters": {"pressure_kpa": {"min": -0.1}}},
])
def test_invalid_filters_are_rejected(client, payload):
    assert client.post(URL, json=payload).status_code == 422


@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e999"])
def test_nonfinite_bounds_are_rejected(client, number):
    response = client.post(
        URL, content='{"filters": {"pressure_kpa": {"min": %s}}}' % number,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_unknown_category_detail_is_a_string(client):
    detail = client.post(URL, json={"filters": {"fuel": ["kerosene"]}}).json()["detail"]
    assert "kerosene" in detail


def test_missing_catalogue_is_503(client, tmp_path, monkeypatch):
    monkeypatch.setattr(catalogue, "CATALOGUE_PATH", tmp_path / "absent.parquet")
    catalogue.load_catalogue.cache_clear()
    response = client.post(URL, json={})
    assert response.status_code == 503
    assert "B01" in response.json()["detail"]
