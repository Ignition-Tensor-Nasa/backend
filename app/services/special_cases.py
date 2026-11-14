"""Curated loader for the bespoke SAFFIRE-III (PSI-100) special cases."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from app.services import catalogue
from app.services.catalogue import CatalogueUnavailable

SAFFIRE_CASE_ALIASES = {
    "SAFFIRE-III": "SAFFIRE-III-S1",
    "SAFFIRE-III-S1": "SAFFIRE-III-S1",
    "SAFFIRE-III-S2": "SAFFIRE-III-S2",
}


@lru_cache(maxsize=4)
def _read_saffire3_file(resolved_path: str) -> dict:
    path = Path(resolved_path)
    if not path.is_file():
        raise CatalogueUnavailable(
            "Curated SAFFIRE-III special cases are missing; run data preparation "
            "(python3 -m data_prep.prepare --data-dir ../data)."
        )
    return json.loads(path.read_text("utf-8"))


def load_saffire3() -> dict:
    """Load curated saffire3.json relative to the active catalogue.CURATED_DIR."""
    path = (catalogue.CURATED_DIR / "saffire3.json").resolve()
    return _read_saffire3_file(str(path))


def list_special_cases() -> dict:
    data = load_saffire3()
    items = data.get("items", [])
    return {
        "bespoke_notice": data.get("bespoke_notice", ""),
        "items": items,
        "total": len(items),
    }


def resolve_special_case(case_id: str) -> tuple[str, int] | None:
    normalized = case_id.strip().upper()
    canonical = SAFFIRE_CASE_ALIASES.get(normalized)
    if canonical is None:
        return None
    sample_number = 2 if canonical == "SAFFIRE-III-S2" else 1
    return canonical, sample_number


def special_case_detail(case_id: str) -> dict | None:
    resolved = resolve_special_case(case_id)
    if resolved is None:
        return None
    canonical, sample_number = resolved
    data = load_saffire3()
    return {
        "case_id": canonical,
        "active_sample_number": sample_number,
        "bespoke_notice": data.get("bespoke_notice", ""),
        "items": data.get("items", []),
        "samples": data.get("samples", {}),
        "spacecraft": data.get("spacecraft", {}),
        "methodology_notes": data.get("methodology_notes", []),
        "sources": data.get("sources", []),
    }
