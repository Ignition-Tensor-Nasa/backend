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
        assert item["fuel"] in options["fuel"]
        detail = client.get(f"/api/v1/cases/{item['case_id']}").json()
        assert detail["datasets"] and detail["source_label"] and detail["note"]
        for dataset in detail["datasets"]:
            keys = [column["key"] for column in dataset["columns"]]
            assert dataset["default_x"] in keys and set(dataset["default_y"]) <= set(keys)
