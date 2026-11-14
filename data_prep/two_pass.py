"""B09: two-pass preparation of the filter values, checked against the curated slice.

Pass 1 (propose) reads every row of the two summary tables and proposes a
vocabulary: each raw fuel label with a suggested canonical filter value. A person
reviews it and saves it as data_prep/vocabulary.json.

Pass 2 (classify) revisits every test selected in the manifest, classifies it
against the reviewed vocabulary, writes data/curated/catalogue_two_pass.parquet
and reports whether it reproduces the curated catalogue's filter values.

    python3 -m data_prep.two_pass propose  --data-dir ../data_nasa
    python3 -m data_prep.two_pass classify --data-dir ../data_nasa

Scope: the selected ACME CFI-G and CFI (FLEX) inputs only, not the whole corpus.
Numeric rules are shared with data_prep.prepare; categories go through the vocabulary.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import polars as pl

from data_prep import flex, prepare

HERE = Path(__file__).parent
KEYS = ["case_id", "investigation", "fuel", "o2_pct", "pressure_kpa", "z_st"]


def propose_fuel(label: str) -> str | None:
    """Suggested canonical fuel: a plain name, with a leading dilution like '30% ' removed.

    Formulas and blends (anything with digits, brackets or slashes) get no suggestion;
    they need a reviewer and evidence.
    """
    name = re.sub(r"^\d+%\s*", "", label.strip()).lower()
    return name if re.fullmatch(r"[a-z][a-z -]*", name) else None


def propose(data_dir: Path) -> dict:
    sources = {
        prepare.INVESTIGATION: Counter(
            (row.get("Fuel") or "").strip()
            for rows in prepare.cfig_table(data_dir).values() for row in rows),
        flex.INVESTIGATION: Counter(
            str(row["Fuel"]).strip()
            for row in flex.sheet_rows(data_dir / flex.FAMILY / "xlsx" / flex.SHEET).values()),
    }
    return {
        "reviewed": False,
        "investigations": {
            name: {"fuel": {label: {"canonical": propose_fuel(label), "rows": count}
                            for label, count in sorted(labels.items()) if label}}
            for name, labels in sources.items()},
    }


def classify_fuel(vocabulary: dict, investigation: str, label: str | None) -> str | None:
    """Canonical fuel from the reviewed vocabulary; unknown labels stay None."""
    entry = vocabulary["investigations"].get(investigation, {}).get("fuel", {}).get((label or "").strip())
    return entry["canonical"] if entry else None


def classify(data_dir: Path, vocabulary: dict) -> pl.DataFrame:
    if not vocabulary.get("reviewed"):
        raise SystemExit("vocabulary.json is not marked as reviewed.")
    manifest = json.loads((HERE / "manifest.json").read_text("utf-8"))
    rows = []
    table = prepare.cfig_table(data_dir)
    for entry in manifest["cases"]:
        found = table.get(entry["test_id"], [])
        if len(found) != 1:
            continue
        o2_pct, kpa, z_st = prepare.cfig_conditions(found[0])
        rows.append((f"CFIG-{entry['test_id']}", prepare.INVESTIGATION, classify_fuel(
            vocabulary, prepare.INVESTIGATION, found[0].get("Fuel")), o2_pct, kpa, z_st))
    sheet = flex.sheet_rows(data_dir / flex.FAMILY / "xlsx" / flex.SHEET)
    for entry in manifest.get("flex_cases", []):
        row = sheet.get(entry["test_id"])
        if row is None:
            continue
        o2_pct, kpa = flex.flex_conditions(row, entry["summary_pdf"])
        rows.append((entry["test_id"], flex.INVESTIGATION, classify_fuel(
            vocabulary, flex.INVESTIGATION, str(row["Fuel"])), o2_pct, kpa, None))
    schema = dict.fromkeys(KEYS[:3], pl.String) | dict.fromkeys(KEYS[3:], pl.Float64)
    return pl.DataFrame(rows, schema=schema, orient="row").sort("case_id")


def differences(two_pass: pl.DataFrame, curated: pl.DataFrame) -> list[str]:
    """Human-readable differences in filter values; empty when the slice is reproduced."""
    ours = {row["case_id"]: row for row in two_pass.to_dicts()}
    theirs = {row["case_id"]: row for row in curated.select(KEYS).to_dicts()}
    problems = [f"{case_id}: only in {'two-pass' if case_id in ours else 'curated'} output"
                for case_id in sorted(ours.keys() ^ theirs.keys())]
    for case_id in sorted(ours.keys() & theirs.keys()):
        problems += [f"{case_id}.{key}: two-pass {ours[case_id][key]!r} != curated {theirs[case_id][key]!r}"
                     for key in KEYS if ours[case_id][key] != theirs[case_id][key]]
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["propose", "classify"])
    parser.add_argument("--data-dir", type=Path, default=prepare.BACKEND.parent / "data_nasa")
    parser.add_argument("--out-dir", type=Path, default=prepare.BACKEND / "data" / "curated")
    args = parser.parse_args()
    if args.command == "propose":
        target = HERE / "vocabulary.proposed.json"
        target.write_text(json.dumps(propose(args.data_dir.resolve()), ensure_ascii=False, indent=1) + "\n", "utf-8")
        print(f"Proposal written to {target}. Review it, set \"reviewed\": true and save it as vocabulary.json.")
    else:
        result = classify(args.data_dir.resolve(), json.loads((HERE / "vocabulary.json").read_text("utf-8")))
        result.write_parquet(args.out_dir / "catalogue_two_pass.parquet")
        curated_path = args.out_dir / "catalogue.parquet"
        if not curated_path.is_file():
            raise SystemExit(f"{result.height} cases classified; no curated catalogue to compare with.")
        curated = pl.read_parquet(curated_path)
        # Only the investigations this script classifies are compared (not Flame Design etc.).
        curated = curated.filter(pl.col("investigation").is_in(result["investigation"].unique().to_list()))
        problems = differences(result, curated)
        print("\n".join(problems))
        print(f"{result.height} cases classified; reproduces the curated slice: {'no' if problems else 'yes'}")
        raise SystemExit(1 if problems else 0)
