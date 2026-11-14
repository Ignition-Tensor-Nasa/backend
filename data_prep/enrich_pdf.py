"""B08: offline PDF metadata enrichment with page evidence, for human review.

    python3 -m data_prep.enrich_pdf FLEX-1213 "../data_nasa/CFI_PSI-39/pdf/FLEX-1213_Summary.pdf"

Reads one PDF page by page (pdftotext), asks the configured provider for parameters and events
stated explicitly for ONE test, and writes data_prep/enrichment/<test>.proposed.json.
The proposal never touches the catalogue: a person reviews it and copies accepted
values into the manifest. Needs provider credentials and poppler's pdftotext.

Guards applied in code, not left to the model:
- every proposed value must come with a page number and a verbatim quote that
  really occurs on that page and contains the value; otherwise it is rejected;
- anything not stated stays null;
- values that differ from the curated catalogue are flagged, never applied.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

import polars as pl

from app.ai.provider import AIUnavailable, call_json

HERE = Path(__file__).parent
FIELDS = {  # name -> what to look for
    "fuel": "fuel name",
    "pressure": "chamber or ambient pressure",
    "o2_fraction": "oxygen mole fraction or percentage of the atmosphere",
    "n2_fraction": "nitrogen fraction of the atmosphere",
    "he_fraction": "helium fraction of the atmosphere",
    "co2_fraction": "carbon dioxide fraction of the atmosphere",
    "initial_droplet_diameter": "initial droplet diameter",
    "extinction_droplet_diameter": "droplet diameter at extinction",
    "burning_rate_constant": "average burning-rate constant",
    "burn_time": "burn duration",
}
# Curated column <- (field, unit as written, factor to the curated unit); for conflict flags only.
COMPARABLE = {"o2_pct": [("o2_fraction", None, 100.0), ("o2_fraction", "%", 1.0)],
              "pressure_kpa": [("pressure", "mmhg", 101.325 / 760), ("pressure", "atm", 101.325),
                               ("pressure", "bar", 100.0), ("pressure", "kpa", 1.0),
                               ("pressure", "psi", 6.894757293168)]}

SYSTEM = """You extract experiment metadata for exactly one test from the text of a NASA document.

You receive the test ID and the document's pages, each introduced by a line "=== page N ===".

For every requested field, report it only if the document states it explicitly for this \
specific test (for example in that test's own table row, summary sheet or log entry). A \
general statement about the whole campaign, a range covering many tests, a typical value, or \
a value belonging to another test is NOT a value for this test: return null for it. Never \
calculate, convert, round or infer a value.

For each reported field give: value (exactly as written, digits and decimal point only for \
numbers), unit (exactly as written, or null if none is written), page (the page number), and \
quote (a short verbatim excerpt from that page that contains the value). If a field is not \
stated for this test, set value, unit, page and quote to null.

Also list events explicitly logged for this test with a time (ignition, extinction, flow \
changes): label, time_text exactly as written, page and a verbatim quote. Use an empty list \
if there are none."""


def _nullable(kind: str) -> dict:
    return {"anyOf": [{"type": kind}, {"type": "null"}]}


SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(FIELDS)},
                           "value": _nullable("string"), "unit": _nullable("string"),
                           "page": _nullable("integer"), "quote": _nullable("string")},
            "required": ["name", "value", "unit", "page", "quote"], "additionalProperties": False}},
        "events": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "time_text": {"type": "string"},
                           "page": {"type": "integer"}, "quote": {"type": "string"}},
            "required": ["label", "time_text", "page", "quote"], "additionalProperties": False}},
    },
    "required": ["fields", "events"], "additionalProperties": False,
}


def pdf_pages(path: Path) -> list[str]:
    """Text of each page, in order (pdftotext separates pages with form feeds, with pypdf fallback)."""
    try:
        output = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                                capture_output=True, text=True).stdout
    except FileNotFoundError as error:
        try:
            from pypdf import PdfReader
        except ImportError as imp_err:
            raise SystemExit("pdftotext (poppler-utils) or pypdf is required.") from (imp_err or error)
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Cannot read {path.name}: {error.stderr.strip()}") from error
    pages = output.split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _evidence_problem(pages: list[str], page, quote, needle) -> str | None:
    if not isinstance(page, int) or not 1 <= page <= len(pages):
        return "page number outside the document"
    if not quote or _squash(quote) not in _squash(pages[page - 1]):
        return "quote not found on the cited page"
    if _squash(str(needle)) not in _squash(quote):
        return "quote does not contain the value"
    return None


def checked(answer: dict, pages: list[str]) -> dict:
    """Keep only items whose evidence is really on the cited page; report the rest."""
    fields = {name: None for name in FIELDS}
    events, rejected = [], []
    for item in answer.get("fields", []):
        if item.get("name") not in FIELDS or item.get("value") is None:
            continue
        problem = _evidence_problem(pages, item.get("page"), item.get("quote"), item["value"])
        if problem:
            rejected.append({**item, "reason": problem})
        elif fields[item["name"]] is None:
            fields[item["name"]] = {key: item[key] for key in ("value", "unit", "page", "quote")}
        else:
            rejected.append({**item, "reason": "second value for the same field"})
    for item in answer.get("events", []):
        problem = _evidence_problem(pages, item.get("page"), item.get("quote"), item.get("time_text"))
        (rejected if problem else events).append({**item, "reason": problem} if problem else item)
    return {"fields": fields, "events": events, "rejected": rejected}


def conflicts(fields: dict, curated: dict | None) -> list[str]:
    """Differences from curated filter values. Flags only; nothing is overwritten."""
    found = []
    for column, options in COMPARABLE.items():
        for name, unit, factor in options:
            item = fields.get(name)
            if not item or (curated or {}).get(column) is None:
                continue
            written = (item["unit"] or "").casefold().replace(" ", "") or None
            if name == "o2_fraction" and written is None and float(item["value"]) > 1:
                written = "%"  # a bare "21" is a percentage, a bare "0.21" a fraction
            if written != unit:
                continue
            value = float(item["value"]) * factor
            if abs(value - curated[column]) > 0.02 * max(abs(curated[column]), 1e-9):
                found.append(f"{column}: document says {item['value']} {item['unit'] or ''}".strip()
                             + f" (= {value:.3f}), curated value is {curated[column]}")
    return found


def enrich(test_id: str, pdf: Path, curated: dict | None) -> dict:
    pages = pdf_pages(pdf)
    if not any(page.strip() for page in pages):
        raise SystemExit(f"{pdf.name} has no extractable text (scanned?).")
    document = "\n".join(f"=== page {number} ===\n{text}" for number, text in enumerate(pages, 1))
    answer = call_json(SYSTEM, json.dumps({
        "test_id": test_id, "requested_fields": FIELDS, "document": document,
    }, ensure_ascii=False), SCHEMA, effort="medium", timeout=180.0)
    result = checked(answer, pages)
    return {"status": "proposed - needs human review; not applied to the catalogue",
            "test_id": test_id, "document": pdf.name, "pages": len(pages), **result,
            "conflicts_with_curated": conflicts(result["fields"], curated)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("test_id", help="the one test the values must belong to, e.g. FLEX-1213")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--curated-dir", type=Path, default=HERE.parent / "data" / "curated")
    args = parser.parse_args()
    catalogue = args.curated_dir / "catalogue.parquet"
    rows = (pl.read_parquet(catalogue).filter(pl.col("case_id").str.ends_with(args.test_id)).to_dicts()
            if catalogue.is_file() else [])
    try:
        proposal = enrich(args.test_id, args.pdf, rows[0] if len(rows) == 1 else None)
    except AIUnavailable as error:
        raise SystemExit(f"No proposal written: {error}") from None
    target = HERE / "enrichment" / f"{args.test_id}.proposed.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(proposal, ensure_ascii=False, indent=1) + "\n", "utf-8")
    kept = sum(1 for value in proposal["fields"].values() if value)
    print(f"{kept} fields, {len(proposal['events'])} events, {len(proposal['rejected'])} rejected, "
          f"{len(proposal['conflicts_with_curated'])} conflicts -> {target}")
