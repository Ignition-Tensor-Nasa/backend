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

## Case search (B03)

`POST /api/v1/cases/search` filters the curated case catalogue and returns
`{ "items": CaseSummary[], "total": number }`, ordered by case ID.

```sh
curl -X POST http://127.0.0.1:8000/api/v1/cases/search \
  -H 'Content-Type: application/json' \
  -d '{"filters": {"fuel": ["methane"], "o2_pct": {"min": 21}}}'
```

The catalogue is read once from `data/catalogue.parquet` (override with
`PSI_CATALOGUE_PATH`) and must have the columns `case_id`, `title`,
`investigation`, `fuel`, `o2_pct`, `pressure_kpa`. It is produced by the B01
data preparation; until it exists the endpoint answers 503. Invalid keys,
categories or ranges answer 422.

Tests (need `pytest` and `httpx`): `python3 -m pytest tests -q`
