from app.services import projects
from tests.conftest import clear_caches


def test_project_list_and_detail(client):
    listed = client.get("/api/v1/projects").json()
    assert [p["project_id"] for p in listed] == ["proj-a", "proj-b"]
    assert listed[0]["case_count"] == 3 and listed[0]["fuels"] == ["ethylene", "methane"]
    detail = client.get("/api/v1/projects/proj-a").json()
    assert detail["filters"]["fuel"] == ["ethylene", "methane"]
    assert detail["filters"]["o2_pct"] == {"min": 21.0, "max": 40.0}
    assert [v["kind"] for v in detail["views"]] == ["a"] or detail["views"][0]["channels"][0]["key"] == "signal"
    assert client.get("/api/v1/projects/nope").status_code == 404
    # A project without datasets has no views, and unknown metadata stays null.
    assert client.get("/api/v1/projects/proj-b").json()["views"] == []


def test_project_search_is_scoped_to_its_investigations(client):
    body = client.post("/api/v1/projects/proj-a/cases/search", json={}).json()
    assert [i["case_id"] for i in body["items"]] == ["T-001", "T-002", "T-003"]
    # A filter naming another project's investigation cannot widen the scope.
    body = client.post("/api/v1/projects/proj-a/cases/search",
                       json={"filters": {"investigation": ["INV-B"]}}).json()
    assert [i["case_id"] for i in body["items"]] == ["T-001", "T-002", "T-003"]
    narrowed = client.post("/api/v1/projects/proj-a/cases/search",
                           json={"filters": {"fuel": ["methane"]}}).json()
    assert narrowed["total"] == 2
    assert client.post("/api/v1/projects/proj-a/cases/search",
                       json={"filters": {"fuel": ["kerosene"]}}).status_code == 422
    assert client.post("/api/v1/projects/nope/cases/search", json={}).status_code == 404


def overlay_body(**extra):
    return {"kind": "a", "channel": "signal"} | extra


def test_overlay_returns_labelled_series_from_prepared_data(client):
    response = client.post("/api/v1/projects/proj-a/overlay", json=overlay_body())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["channel"] == {"key": "signal", "label": "Signal", "unit": None}
    assert body["x"] == {"label": "Elapsed time", "unit": "s", "align": "start"}
    assert body["matched"] == 1 and [s["case_id"] for s in body["series"]] == ["T-001"]
    series = body["series"][0]
    assert series["points"] == [[0.0, 1.5], [0.5, None], [1.0, None], [2.0, 3.0]]
    assert series["n_source"] == 4 and series["sampled"] is False and series["o2_pct"] == 21.0
    assert "min/max envelope" in body["note"] and "not aligned" in body["note"]


def test_overlay_sampling_is_declared_and_keeps_peaks(client):
    import polars as pl

    frame = pl.DataFrame({"x": list(range(1000)), "y": [0.0] * 1000}).with_columns(
        pl.when(pl.col("x") == 437).then(9.0).otherwise(pl.col("y")).alias("y"))
    thinned = projects._envelope(frame, 50)
    assert thinned.height <= 60 and thinned["x"][0] == 0 and thinned["x"][-1] == 999
    assert 9.0 in thinned["y"].to_list()
    body = client.post("/api/v1/projects/proj-a/overlay", json=overlay_body(max_points=10)).json()
    assert body["series"][0]["sampled"] is False and body["series"][0]["n_source"] == 4  # 4 <= 10


def test_overlay_aligns_to_logged_ignition_and_reports_runs_left_out(client, monkeypatch):
    monkeypatch.setattr(projects, "IGNITION", "Ignition")
    body = client.post("/api/v1/projects/proj-a/overlay", json=overlay_body(align="ignition")).json()
    assert body["x"]["label"] == "Time since ignition"
    assert body["series"][0]["points"][0] == [-0.5, 1.5]  # fixture ignition at 0.5 s
    assert "0 matching run(s) have no logged ignition" in body["note"]
    monkeypatch.setattr(projects, "IGNITION", "Something else")
    body = client.post("/api/v1/projects/proj-a/overlay", json=overlay_body(align="ignition")).json()
    assert body["series"] == [] and "1 matching run(s) have no logged ignition" in body["note"]


def test_overlay_errors(client):
    url = "/api/v1/projects/proj-a/overlay"
    assert client.post(url, json=overlay_body(channel="nope")).status_code == 422
    assert client.post(url, json=overlay_body(kind="nope")).status_code == 422
    assert client.post(url, json=overlay_body(align="middle")).status_code == 422
    assert client.post(url, json=overlay_body(max_points=2)).status_code == 422
    assert client.post(url, json=overlay_body(extra=1)).status_code == 422
    assert client.post("/api/v1/projects/nope/overlay", json=overlay_body()).status_code == 404
    clear_caches()
