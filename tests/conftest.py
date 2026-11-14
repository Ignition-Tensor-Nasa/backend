"""Synthetic fixtures (not NASA measurements) written to a temporary curated folder."""

import json

from fastapi.testclient import TestClient
import polars as pl
import pytest

from app.main import app
from app.services import catalogue

ROWS = [
    # case_id, investigation, fuel, o2_pct, pressure_kpa
    ("T-003", "INV-A", "ethylene", 30.0, 101.3),
    ("T-001", "INV-A", "methane", 21.0, 101.3),
    ("T-002", "INV-A", "methane", 40.0, 50.0),
    ("T-004", "INV-B", "methane", 21.0, None),
    ("T-005", "INV-B", None, None, 101.3),
    ("T-006", "INV-B", "ethylene", float("nan"), 200.0),
]
Z_ST = {"T-001": 0.1, "T-002": 0.25, "T-003": 0.3}  # unknown for the others
CSV = "time_s,signal\n0.0,1.5\n0.5,\n"


def dataset(dataset_id, case_id, file):
    return {
        "dataset_id": dataset_id, "case_id": case_id, "label": "Fixture series", "file": file,
        "columns": [
            {"key": "time_s", "label": "Elapsed time", "data_type": "number", "unit": "s"},
            {"key": "signal", "label": "Signal", "data_type": "number", "unit": None},
        ],
        "default_x": "time_s", "default_y": ["signal"],
    }


def clear_caches():
    catalogue.load_catalogue.cache_clear()
    catalogue.load_datasets.cache_clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    pl.DataFrame(
        [(cid, f"Fixture {cid}", inv, fuel, o2, kpa, Z_ST.get(cid), "Fixture source", None,
          "Fixture note") for cid, inv, fuel, o2, kpa in ROWS],
        schema=["case_id", "title", "investigation", "fuel", "o2_pct", "pressure_kpa", "z_st",
                "source_label", "source_url", "note"],
        orient="row",
    ).write_parquet(tmp_path / "catalogue.parquet")
    (tmp_path / "csv").mkdir()
    (tmp_path / "csv" / "T-001-a.csv").write_text(CSV, "utf-8")
    (tmp_path / "secret.csv").write_text("outside the csv folder", "utf-8")
    (tmp_path / "datasets.json").write_text(json.dumps({"datasets": [
        dataset("T-001-a", "T-001", "T-001-a.csv"),
        dataset("T-001-gone", "T-001", "T-001-gone.csv"),
        dataset("T-002-escape", "T-002", "../secret.csv"),
    ]}), "utf-8")
    monkeypatch.setattr(catalogue, "CURATED_DIR", tmp_path)
    clear_caches()
    yield TestClient(app)
    clear_caches()
