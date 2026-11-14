"""Curated case catalogue (produced by B01) and exact filter search."""

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Annotated

import polars as pl
import polars.selectors as cs
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Output folder of `python3 -m data_prep.prepare` (B01).
CURATED_DIR = Path(
    os.environ.get("PSI_CURATED_DIR")
    or Path(__file__).resolve().parents[2] / "data" / "curated"
)
CATEGORY_KEYS = ("investigation", "fuel")
RANGE_KEYS = ("o2_pct", "pressure_kpa")
SUMMARY_COLUMNS = ("case_id", "title", *CATEGORY_KEYS, *RANGE_KEYS)

O2Pct = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
PressureKpa = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Categories = Annotated[list[str], Field(min_length=1)]


class CatalogueUnavailable(RuntimeError):
    pass


class _Range(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.min is None and self.max is None:
            raise ValueError("range needs min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min must not exceed max")
        return self


class O2Range(_Range):
    min: O2Pct | None = None
    max: O2Pct | None = None


class PressureRange(_Range):
    min: PressureKpa | None = None
    max: PressureKpa | None = None


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investigation: Categories | None = None
    fuel: Categories | None = None
    o2_pct: O2Range | None = None
    pressure_kpa: PressureRange | None = None


@lru_cache(maxsize=1)
def load_catalogue() -> pl.DataFrame:
    path = CURATED_DIR / "catalogue.parquet"
    if not path.is_file():
        raise CatalogueUnavailable(
            "Curated case catalogue is missing; run the B01 data preparation "
            "(python3 -m data_prep.prepare)."
        )
    df = pl.read_parquet(path)
    if missing := set(SUMMARY_COLUMNS) - set(df.columns):
        raise CatalogueUnavailable(f"Case catalogue lacks columns: {sorted(missing)}")
    # NaN would pass ">= min" in Polars and break JSON; unknown stays null.
    return df.with_columns(cs.float().fill_nan(None))


@lru_cache(maxsize=1)
def load_datasets() -> dict[str, dict]:
    """Dataset descriptors by ID: the allowlist of servable CSV files."""
    path = CURATED_DIR / "datasets.json"
    if not path.is_file():
        raise CatalogueUnavailable(
            "Curated dataset descriptors are missing; run the B01 data preparation "
            "(python3 -m data_prep.prepare)."
        )
    return {item["dataset_id"]: item for item in json.loads(path.read_text("utf-8"))["datasets"]}


def case_detail(case_id: str) -> dict | None:
    rows = load_catalogue().filter(pl.col("case_id") == case_id).to_dicts()
    if not rows:
        return None
    datasets = [item for item in load_datasets().values() if item["case_id"] == case_id]
    return {**rows[0], "datasets": datasets}


def dataset_csv_path(dataset_id: str) -> Path | None:
    """File of an allowlisted dataset; None for unknown IDs."""
    dataset = load_datasets().get(dataset_id)
    if dataset is None:
        return None
    folder = (CURATED_DIR / "csv").resolve()
    path = (folder / dataset["file"]).resolve()
    # The name comes from datasets.json, never the request; still refuse to leave the folder.
    if path.parent != folder or not path.is_file():
        raise CatalogueUnavailable(f"CSV of dataset {dataset_id} is missing; rerun the B01 data preparation.")
    return path


def filter_options() -> dict:
    """Actual categories and known numeric bounds of the curated catalogue."""
    df = load_catalogue()
    options: dict = {key: sorted(df[key].drop_nulls().unique()) for key in CATEGORY_KEYS}
    for key in RANGE_KEYS:
        known = df[key].drop_nulls()
        options[key] = {"min": known.min(), "max": known.max()} if len(known) else None
    return options


def check_categories(filters: Filters) -> None:
    """ValueError when a category value does not exist in the catalogue."""
    df = load_catalogue()
    for key in CATEGORY_KEYS:
        if unknown := set(getattr(filters, key) or ()) - set(df[key].drop_nulls()):
            raise ValueError(f"Unknown {key} value(s): {sorted(unknown)}")


def search_cases(filters: Filters) -> list[dict]:
    """All matching case summaries ordered by case ID. ValueError on unknown categories."""
    check_categories(filters)
    df = load_catalogue()
    for key in CATEGORY_KEYS:
        wanted = getattr(filters, key)
        if wanted is not None:
            df = df.filter(pl.col(key).is_in(wanted))
    for key in RANGE_KEYS:
        bounds = getattr(filters, key)
        if bounds is None:
            continue
        # A null comparison is dropped by filter, so null metadata fails the constraint.
        if bounds.min is not None:
            df = df.filter(pl.col(key) >= bounds.min)
        if bounds.max is not None:
            df = df.filter(pl.col(key) <= bounds.max)
    return df.sort("case_id").select(SUMMARY_COLUMNS).to_dicts()
