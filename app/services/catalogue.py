"""Curated case catalogue (produced by B01) and exact filter search."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Annotated

import polars as pl
import polars.selectors as cs
from pydantic import BaseModel, ConfigDict, Field, model_validator

CATALOGUE_PATH = Path(
    os.environ.get("PSI_CATALOGUE_PATH")
    or Path(__file__).resolve().parents[2] / "data" / "catalogue.parquet"
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
    if not CATALOGUE_PATH.is_file():
        raise CatalogueUnavailable(
            f"Case catalogue {CATALOGUE_PATH.name} is missing; run the B01 data preparation."
        )
    df = pl.read_parquet(CATALOGUE_PATH)
    if missing := set(SUMMARY_COLUMNS) - set(df.columns):
        raise CatalogueUnavailable(f"Case catalogue lacks columns: {sorted(missing)}")
    # NaN would pass ">= min" in Polars and break JSON; unknown stays null.
    return df.with_columns(cs.float().fill_nan(None))


def search_cases(filters: Filters) -> list[dict]:
    """All matching case summaries ordered by case ID. ValueError on unknown categories."""
    df = load_catalogue()
    for key in CATEGORY_KEYS:
        wanted = getattr(filters, key)
        if wanted is None:
            continue
        if unknown := set(wanted) - set(df[key].drop_nulls()):
            raise ValueError(f"Unknown {key} value(s): {sorted(unknown)}")
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
