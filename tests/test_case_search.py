"""Run from backend/: python3 -m pytest tests -q (needs pytest and httpx)."""

import polars as pl
import pytest

from app.services import catalogue
from tests.conftest import clear_caches

URL = "/api/v1/cases/search"


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


def test_filter_options_come_from_the_catalogue(client):
    assert client.get("/api/v1/filters").json() == {
        "investigation": ["INV-A", "INV-B"],
        "fuel": ["ethylene", "methane"],
        "o2_pct": {"min": 21.0, "max": 40.0},  # null and NaN are not bounds
        "pressure_kpa": {"min": 50.0, "max": 200.0},
    }


def test_filter_bounds_are_null_when_nothing_is_known(client, tmp_path):
    pl.read_parquet(tmp_path / "catalogue.parquet").with_columns(
        pl.lit(None, dtype=pl.Float64).alias("o2_pct")
    ).write_parquet(tmp_path / "catalogue.parquet")
    clear_caches()
    assert client.get("/api/v1/filters").json()["o2_pct"] is None


def test_missing_catalogue_is_503(client, tmp_path, monkeypatch):
    monkeypatch.setattr(catalogue, "CURATED_DIR", tmp_path / "absent")
    clear_caches()
    response = client.post(URL, json={})
    assert response.status_code == 503
    assert "B01" in response.json()["detail"]
    assert client.get("/api/v1/filters").status_code == 503
