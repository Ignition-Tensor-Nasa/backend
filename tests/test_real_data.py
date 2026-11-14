"""Smoke test on the real curated data; skipped until `python3 -m data_prep.prepare` has run."""

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.services import catalogue
from tests.conftest import clear_caches

pytestmark = pytest.mark.skipif(
    not (catalogue.CURATED_DIR / "catalogue.parquet").is_file(), reason="no curated data")


def test_every_case_opens_its_measurements():
    clear_caches()
    client = TestClient(app)
    items = client.post("/api/v1/cases/search").json()["items"]
    assert items
    options = client.get("/api/v1/filters").json()
    for item in items:
        assert item["fuel"] is None or item["fuel"] in options["fuel"]
        detail = client.get(f"/api/v1/cases/{item['case_id']}").json()
        assert detail["datasets"] and detail["source_label"] and detail["note"]
        for dataset in detail["datasets"]:
            keys = [column["key"] for column in dataset["columns"]]
            assert dataset["default_x"] in keys and set(dataset["default_y"]) <= set(keys)
            data = client.get(f"/api/v1/datasets/{dataset['dataset_id']}/data").json()
            assert data["dataset"] == dataset and data["note"]
            assert data["rows"] and all(len(row) == len(keys) for row in data["rows"])
            assert all(value is None or isinstance(value, (int, float))
                       for row in data["rows"][:50] for value in row)
            x = keys.index(dataset["default_x"])
            times = [row[x] for row in data["rows"]]
            assert times == sorted(times)
            response = client.get(f"/api/v1/datasets/{dataset['dataset_id']}/csv")
            assert response.status_code == 200
            lines = response.text.splitlines()
            assert lines[0].split(",") == keys
            assert len(lines) > 1 and all(value == "" or float(value) == float(value) for value in lines[1].split(","))
