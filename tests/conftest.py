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
CSV = "time_s,signal\n0.0,1.5\n0.5,\n1.0,\n2.0,3.0\n"  # export of the fixture series below


def dataset(dataset_id, case_id, file):
    return {
        "dataset_id": dataset_id, "case_id": case_id, "label": "Fixture series", "file": file,
        "note": "Fixture note about sampling.",
        "columns": [
            {"key": "time_s", "label": "Elapsed time", "data_type": "number", "unit": "s"},
            {"key": "signal", "label": "Signal", "data_type": "number", "unit": None},
        ],
        "default_x": "time_s", "default_y": ["signal"],
    }


def clear_caches():
    catalogue.load_catalogue.cache_clear()
    catalogue.load_datasets.cache_clear()
    catalogue.load_events.cache_clear()
    catalogue.load_neighbors.cache_clear()
    catalogue.load_context.cache_clear()
    catalogue.dataset_data_json.cache_clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    pl.DataFrame(
        [(cid, f"Fixture {cid}", inv, fuel, o2, kpa, Z_ST.get(cid), "Fixture source", None,
          "Fixture note", "Fixture explanation." if cid == "T-001" else None,
          "curated" if cid == "T-001" else None) for cid, inv, fuel, o2, kpa in ROWS],
        schema=["case_id", "title", "investigation", "fuel", "o2_pct", "pressure_kpa", "z_st",
                "source_label", "source_url", "note", "explanation", "explanation_origin"],
        orient="row",
    ).write_parquet(tmp_path / "catalogue.parquet")
    (tmp_path / "parquet").mkdir()
    # Stored out of time order, with a missing sample, a NaN and an extra column.
    pl.DataFrame({"time_s": [2.0, 0.0, 0.5, 1.0], "signal": [3.0, 1.5, None, float("nan")],
                  "not_described": [1, 2, 3, 4]}).write_parquet(tmp_path / "parquet" / "T-001-a.parquet")
    pl.DataFrame({"time_s": [0.0], "signal": [42.0]}).write_parquet(tmp_path / "secret.parquet")
    (tmp_path / "datasets.json").write_text(json.dumps({"datasets": [
        dataset("T-001-a", "T-001", "T-001-a.parquet"),
        dataset("T-001-gone", "T-001", "T-001-gone.parquet"),
        dataset("T-002-escape", "T-002", "../secret.parquet"),
    ]}), "utf-8")
    (tmp_path / "events.json").write_text(json.dumps({"method": "fixture", "events": {"T-001": [
        {"label": "Ignition", "time_s": 0.5, "dataset_id": "T-001-a", "source": "fixture.txt"}]}}), "utf-8")
    (tmp_path / "context.json").write_text(json.dumps({"disclaimer": "grouping only", "investigations": {
        "INV-A": {"tags": ["suppression"], "rationale": "fixture", "evidence": [
            {"document": "fixture.pdf", "page": 2, "quote": "fixture quote"}]}}}), "utf-8")
    (tmp_path / "neighbors.json").write_text(json.dumps({"metric": "fixture metric", "neighbors": {
        "T-001": [{"case_id": "T-003", "distance": 0.5}, {"case_id": "T-002", "distance": 2.0},
                  {"case_id": "T-gone", "distance": 3.0}]}}), "utf-8")
    monkeypatch.setattr(catalogue, "CURATED_DIR", tmp_path)
    clear_caches()
    yield TestClient(app)
    clear_caches()
