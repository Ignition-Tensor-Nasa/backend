from tests.conftest import CSV


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


def test_csv_download(client):
    response = client.get("/api/v1/datasets/T-001-a/csv", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200 and response.text == CSV
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == 'attachment; filename="T-001-a.csv"'
    assert "Content-Disposition" in response.headers["access-control-expose-headers"]


def test_unknown_dataset_and_path_tricks_are_404(client):
    for dataset_id in ("nope", "T-001-a.csv", "..%2Fsecret", "..%2F..%2Fcatalogue.parquet", "%2Fetc%2Fpasswd"):
        assert client.get(f"/api/v1/datasets/{dataset_id}/csv").status_code == 404, dataset_id


def test_listed_file_that_is_missing_or_outside_the_folder_is_not_served(client):
    for dataset_id in ("T-001-gone", "T-002-escape"):
        response = client.get(f"/api/v1/datasets/{dataset_id}/csv")
        assert response.status_code == 503 and "secret" not in response.text
