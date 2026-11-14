"""Tests for the bespoke SAFFIRE-III (PSI-100) special-case routes and AI summary."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
import pytest

from app.ai import provider
from app.main import app
from app.services import catalogue, special_cases


def _sample_fixture_payload() -> dict:
    sample_stats = {
        "duration_s": 689.9,
        "igniter_window": [10.0, 17.0],
        "peak_sample_tc": "Sample Temp 5 (TC11)",
        "peak_sample_temp_c": 1088.0,
        "peak_sample_time_s": 257.0,
        "peak_after_ignition_s": 247.0,
        "peak_plume_tc": "Plume Temp 3 (TC3)",
        "peak_plume_temp_c": 940.0,
        "peak_radiometer": "Radiometer 4",
        "peak_heat_flux_mw_cm2": 169.0,
        "idle_heat_flux_mw_cm2": -18.0,
        "o2_before_pct": 19.84,
        "o2_min_pct": 19.40,
        "co2_before_pct": 0.40,
        "co2_max_pct": 0.61,
        "pressure_before_kpa": 99.4,
        "pressure_rise_kpa": 0.51,
        "air_flow_cm_s": 25.0,
    }
    return {
        "bespoke_notice": "Bespoke SAFFIRE-III view only.",
        "items": [
            {
                "case_id": "SAFFIRE-III-S1",
                "sample_number": 1,
                "title": "Saffire-III Sample 1",
                "subtitle": "Large burn",
                "investigation": "Saffire-III (PSI-100)",
                "capsule": "Cygnus OA-7",
                "fuel": "Saffire-III Sample 1",
                "o2_pct": 19.84,
                "pressure_kpa": 99.4,
                "recording_duration_s": 689.9,
                "peak_temp_c": 1088.0,
                "peak_heat_flux_mw_cm2": 169.0,
                "highlight_badge": "BESPOKE",
            },
            {
                "case_id": "SAFFIRE-III-S2",
                "sample_number": 2,
                "title": "Saffire-III Sample 2",
                "subtitle": "Companion",
                "investigation": "Saffire-III (PSI-100)",
                "capsule": "Cygnus OA-7",
                "fuel": "Saffire-III Sample 2",
                "o2_pct": 19.9,
                "pressure_kpa": 99.4,
                "recording_duration_s": 160.0,
                "peak_temp_c": 20.0,
                "peak_heat_flux_mw_cm2": -3.7,
                "highlight_badge": "BESPOKE",
            },
        ],
        "samples": {
            "1": {
                "sample_number": 1,
                "case_id": "SAFFIRE-III-S1",
                "tab_label": "Sample 1",
                "duration_s": 689.9,
                "point_limit": 2500,
                "igniter_window": [10.0, 17.0],
                "presets": {"Whole run": [0.0, 689.9]},
                "tiles": [{"value": "1088 °C", "label": "peak", "detail": "TC11"}],
                "stats": sample_stats,
                "burn_panels": [],
                "arrival": {
                    "thermocouples": ["TC7", "TC8", "TC9", "TC10", "TC11", "TC12"],
                    "thresholds_c": [100, 300, 600],
                    "default_threshold_c": 300,
                    "arrivals_by_threshold": {"300": [100.0, 110.0, None, None, 150.0, None]},
                    "peaks_c": [800.0, 820.0, 200.0, 220.0, 1088.0, 180.0],
                    "max_t": 689.9,
                    "note": "Order only",
                },
                "playback": {
                    "thermocouples": ["TC7", "TC8", "TC9", "TC10", "TC11", "TC12"],
                    "step_s": 3.0,
                    "y_max_c": 1175.0,
                    "frames": [{"t_s": 0.0, "values_c": [20.0] * 6}],
                },
            },
            "2": {
                "sample_number": 2,
                "case_id": "SAFFIRE-III-S2",
                "tab_label": "Sample 2",
                "duration_s": 160.0,
                "point_limit": 2500,
                "igniter_window": [60.0, 68.0],
                "presets": {"Whole run": [0.0, 160.0]},
                "tiles": [{"value": "20.0 °C", "label": "peak", "detail": "TC12"}],
                "stats": {**sample_stats, "duration_s": 160.0, "igniter_window": [60.0, 68.0], "peak_sample_temp_c": 20.0, "peak_plume_temp_c": 48.4, "peak_heat_flux_mw_cm2": -3.7},
                "burn_panels": [],
                "arrival": {
                    "thermocouples": ["TC7", "TC8", "TC9", "TC10", "TC11", "TC12"],
                    "thresholds_c": [15, 17, 100],
                    "default_threshold_c": 15,
                    "arrivals_by_threshold": {"15": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                    "peaks_c": [17.5, 17.9, 17.9, 17.1, 17.1, 20.0],
                    "max_t": 160.0,
                    "note": "Order only",
                },
                "playback": {
                    "thermocouples": ["TC7", "TC8", "TC9", "TC10", "TC11", "TC12"],
                    "step_s": 3.0,
                    "y_max_c": 22.0,
                    "frames": [{"t_s": 0.0, "values_c": [17.0] * 6}],
                },
            },
        },
        "spacecraft": {
            "events": {
                "igniter1": [5953.8, 5960.8],
                "igniter2": [5758.8, 5766.0],
                "smoke_wire": [5632.3],
                "window": [5480.8, 6899.8],
                "scatter_baseline": 0.025,
                "scatter_peak": 1.805,
                "scatter_peak_t": 6295.4,
                "first_rise": 5655.3,
                "smoke_lag_s": 23.0,
            },
            "presets": {"Experiment": [5480.8, 6899.8]},
            "t_max": 6899.8,
            "panels": [],
        },
        "methodology_notes": ["Note 1"],
        "sources": ["Source 1"],
    }


def test_special_cases_routes_and_ai_summary(tmp_path, monkeypatch):
    (tmp_path / "saffire3.json").write_text(json.dumps(_sample_fixture_payload()), "utf-8")
    monkeypatch.setattr(catalogue, "CURATED_DIR", tmp_path)
    monkeypatch.setattr(provider, "AI_CONFIG_PATH", tmp_path / ".env")
    special_cases._read_saffire3_file.cache_clear()

    client = TestClient(app)

    listing = client.get("/api/v1/special-cases")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 2
    assert [i["case_id"] for i in body["items"]] == ["SAFFIRE-III-S1", "SAFFIRE-III-S2"]

    detail = client.get("/api/v1/special-cases/SAFFIRE-III-S2")
    assert detail.status_code == 200
    assert detail.json()["active_sample_number"] == 2

    missing = client.get("/api/v1/special-cases/UNKNOWN")
    assert missing.status_code == 404

    monkeypatch.setenv("PSI_AI_MODE", "demo")
    summary_res = client.post(
        "/api/v1/special-cases/SAFFIRE-III-S1/ai-summary",
        json={"focus": "spacecraft_cabin", "language": "pl", "custom_prompt": "Jak reagował detektor dymu?"},
    )
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["mode"] == "demo"
    assert summary["case_id"] == "SAFFIRE-III-S1"
    assert "FIZ 2" in summary["summary"]["spacecraft_telemetry_insight"]

    monkeypatch.setenv("PSI_AI_MODE", "live")
    monkeypatch.delenv("PSI_AI_PROVIDER", raising=False)
    unconfigured = client.post("/api/v1/special-cases/SAFFIRE-III-S1/ai-summary", json={})
    assert unconfigured.status_code == 503


def test_special_cases_real_curated_file_when_present():
    special_cases._read_saffire3_file.cache_clear()
    curated_file = catalogue.CURATED_DIR / "saffire3.json"
    if not curated_file.is_file():
        pytest.skip("curated saffire3.json not present")
    client = TestClient(app)
    res = client.get("/api/v1/special-cases/SAFFIRE-III-S1")
    assert res.status_code == 200
    data = res.json()
    assert len(data["samples"]["1"]["burn_panels"]) == 7
    assert len(data["samples"]["2"]["burn_panels"]) == 7
    assert len(data["spacecraft"]["panels"]) == 5
    assert data["samples"]["1"]["stats"]["peak_sample_temp_c"] == pytest.approx(1088.0, abs=1.0)
