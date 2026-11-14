# Plot Data API

A minimal Python REST API skeleton for serving data to an interactive plotting
frontend. Polars is included as a dependency for future data processing; no data
endpoints are implemented yet.

## Project structure

```text
app/
  main.py          # Creates the app and registers routers
  ai/              # Placeholder for AI-assisted data analysis
    filter_suggestions.py  # Future data filtering suggestions
  routes/          # REST endpoint definitions
    hello.py       # Hello-world endpoint
  services/        # Future data loading and Polars processing logic
```

Add new endpoint routers under `app/routes/` and register them in `app/main.py`.
Keep data loading and processing in `app/services/`, which routes can call to
prepare their responses.
The `app/ai/` package is reserved for future AI-powered filter suggestions; it
does not implement any AI functionality yet.

## Run locally

Requires Python 3.10 or newer.

Choose one of the following setups from the project directory.

### Using a virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Using globally installed libraries

If FastAPI, Uvicorn and Polars are already installed for your Python interpreter
in versions compatible with `requirements.txt`, no installation is needed.

### Start the server

With either setup, run:

```sh
python3 -m uvicorn app.main:app --reload
```

Optional settings are environment variables listed in `.env.example`
(`cp .env.example .env`, edit, then `set -a; . ./.env; set +a` before starting).
The server starts without any of them and without an LLM key. CORS allows the
local frontend origins in `CORS_ORIGINS`.

Visit <http://127.0.0.1:8000/> or run:

```sh
curl http://127.0.0.1:8000/
```

The response is `{"message":"Hello, world!"}`. Interactive API documentation is
available at <http://127.0.0.1:8000/docs>.

## Prepare the demo data (B01)

The API serves a small curated slice of NASA PSI data. Build it once (and again
after editing the manifest); raw files are only read:

```sh
python3 -m data_prep.prepare --data-dir ../data_nasa
```

`--data-dir` is the folder containing `ACME-CFI-G_PSI-159/`. The script reads
the hand-reviewed test list in `data_prep/manifest.json` and writes
`data/curated/` (git-ignored, replaced on every run):

- `catalogue.parquet`: one row per case (verified run) with filter values,
  source reference and note.
- `datasets.json`: descriptors (columns, units, default axes) of each CSV.
- `csv/<dataset_id>.csv`: browser-ready series, full rate, no preamble.

What is and is not verified:

- A case is linked to its recording by the `Test Point:` field inside the raw
  file's preamble, checked against the ID in the experimental summary table.
  Tests with a missing, ambiguous or corrupted (NUL bytes) recording are skipped.
- Only ACME CFI-G normal flames are curated, so `o2_pct` always means the
  chamber O₂ mole fraction before the test. `pressure_kpa` is the table's
  approximate pressure, bar × 100.
- `time_s` is seconds since the first sensor sample (both datasets of a case
  share that zero). Samples keep the source order; source timestamps can
  jitter by a few milliseconds (one known 2 ms backward step in 21174F1). Sensor channels keep their raw recorded values with a null
  unit: the source files do not state units. Setpoints are volts, as named in
  the source header.
- `fuel` is the base hydrocarbon; the table's label (e.g. "30% butane") stays
  in the title and the fuel mole fraction in the note.

## Filter options (B02)

`GET /api/v1/filters` returns the categories and known numeric bounds present
in the curated catalogue (`null` bounds when no case has that value):

```json
{"investigation": ["ACME CFI-G"], "fuel": ["butane", "ethane", "propane"],
 "o2_pct": {"min": 17.969, "max": 39.655},
 "pressure_kpa": {"min": 50.665, "max": 303.99}}
```

## Case search (B03)

`POST /api/v1/cases/search` filters the curated case catalogue and returns
`{ "items": CaseSummary[], "total": number }`, ordered by case ID.

```sh
curl -X POST http://127.0.0.1:8000/api/v1/cases/search \
  -H 'Content-Type: application/json' \
  -d '{"filters": {"fuel": ["methane"], "o2_pct": {"min": 21}}}'
```

The catalogue is read once from `data/curated/catalogue.parquet` (folder
overridable with `PSI_CURATED_DIR`). Until it exists the endpoint answers 503.
Invalid keys, categories or ranges answer 422.

Tests (need `pytest` and `httpx`): `python3 -m pytest tests -q`

## Case detail (B04)

`GET /api/v1/cases/{case_id}` returns the case summary plus `source_label`,
`source_url` (null: no verified link), a curated `note` and the descriptors of
its datasets: `dataset_id`, `label`, `columns` (`key`, `label`, `data_type`,
`unit`), `default_x`, `default_y`. Unknown cases answer 404.

```sh
curl http://127.0.0.1:8000/api/v1/cases/CFIG-21291C1
```
