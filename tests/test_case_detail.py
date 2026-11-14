def test_case_detail_describes_its_datasets(client):
    body = client.get("/api/v1/cases/T-001").json()
    assert body["case_id"] == "T-001" and body["o2_pct"] == 21.0
    assert (body["source_label"], body["source_url"], body["note"]) == (
        "Fixture source", None, "Fixture note")
    assert [d["dataset_id"] for d in body["datasets"]] == ["T-001-a", "T-001-gone"]
    assert body["datasets"][0] == {
        "dataset_id": "T-001-a", "label": "Fixture series",
        "columns": [
            {"key": "time_s", "label": "Elapsed time", "data_type": "number", "unit": "s"},
            {"key": "signal", "label": "Signal", "data_type": "number", "unit": None},
        ],
        "default_x": "time_s", "default_y": ["signal"],
    }


def test_case_without_datasets_and_with_null_metadata(client):
    body = client.get("/api/v1/cases/T-005").json()
    assert body["datasets"] == [] and body["fuel"] is None and body["o2_pct"] is None
    assert client.get("/api/v1/cases/T-006").json()["o2_pct"] is None  # NaN is unknown


def test_unknown_case_is_404(client):
    response = client.get("/api/v1/cases/T-999")
    assert response.status_code == 404 and "T-999" in response.json()["detail"]

