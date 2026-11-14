"""B01: build the curated demo catalogue and browser-ready CSVs from raw NASA PSI files.

Run from backend/:  python3 -m data_prep.prepare --data-dir ../data_nasa

Reads only the ACME CFI-G and CFI (FLEX) tests listed in data_prep/manifest.json
and never modifies raw data. Output (data/curated/) is replaced on every run.
"""

import argparse
import json
from pathlib import Path
import re
import shutil

import polars as pl

BACKEND = Path(__file__).resolve().parents[1]
FAMILY = "ACME-CFI-G_PSI-159"
TABLE = "PSI-159_Experimental table_CFI-G_Normal Flames.csv"
INVESTIGATION = "ACME CFI-G"
BAR_TO_KPA = 100.0

# Raw column -> label. Units are null: the source files do not state them
# (signals look like raw volts, which is not verified), so nothing is converted.
CHANNELS = {
    "MFCA1StaticMixerFlow": "MFC A1 static mixer flow (raw)",
    "MFCA2BypassFlow": "MFC A2 bypass flow (raw)",
    "MFCA3FuelFlow": "MFC A3 fuel flow (raw)",
    "IonCurrent": "Ion current (raw)",
    "PTA1": "Pressure transducer PTA1 (raw)",
    "PTA2": "Pressure transducer PTA2 (raw)",
    "PTA3": "Pressure transducer PTA3 (raw)",
    "PMT1": "Photomultiplier PMT1 (raw)",
    "PMT2": "Photomultiplier PMT2 (raw)",
    "PMT3": "Photomultiplier PMT3 (raw)",
    "TCA": "Thermocouple TCA (raw)",
    "TCB": "Thermocouple TCB (raw)",
    "TCC": "Thermocouple TCC (raw)",
    "TCD": "Thermocouple TCD (raw)",
    "TCBurner": "Burner thermocouple (raw)",
    "Radiometer3+60LoGain": "Radiometer +60, low gain (raw)",
    "Radiometer3+30LoGain": "Radiometer +30, low gain (raw)",
    "Radiometer3-10LoGain": "Radiometer -10, low gain (raw)",
    "RadiometerOffset": "Offset radiometer (raw)",
}
SETPOINTS = {f"MFCA{n}Setpoint(V)": f"MFC A{n} setpoint" for n in (1, 2, 3)}
TP_EVENTS = ("Ignition and retraction", "Extinction detected", "MFCs off")
ENDINGS = {"SE": "self-extinction (SE)", "FT": "flow termination (FT)"}


class Skip(Exception):
    """This case cannot be prepared from verified source files."""


def clock_seconds(text: str) -> float:
    hours, minutes, seconds, *millis = (int(part) for part in text.strip().split(":"))
    return hours * 3600 + minutes * 60 + seconds + (millis[0] / 1000 if millis else 0)


def elapsed_seconds(times: list[str], start: float | None = None) -> list[float]:
    """Seconds since `start` (default: the first HH:MM:SS:mmm sample), across midnight."""
    start = clock_seconds(times[0]) if start is None else start
    day, previous, out = 0.0, None, []
    for text in times:
        now = clock_seconds(text)
        if previous is not None and now < previous - 43200:
            day += 86400
        previous = now
        out.append(round(now + day - start, 3))
    return out


def read_series(path: Path, test_id: str, wanted: dict,
                start: float | None = None) -> tuple[pl.DataFrame, float]:
    """Read one raw series; return it with time_s, and the clock time of t=0."""
    if not path.is_file() or path.stat().st_size == 0:
        raise Skip(f"missing or empty {path.name}")
    content = path.read_bytes()
    if b"\x00" in content:
        raise Skip(f"{path.name}: corrupted recording (NUL bytes)")
    preamble = content.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    found = re.search(r"Test Point:\s*([^,]+)", preamble)
    if not found or found.group(1).strip() != test_id:
        raise Skip(f"{path.name}: preamble does not name test point {test_id}")
    raw = pl.read_csv(content, skip_rows=1, schema_overrides={"Time": pl.String})
    missing = [name for name in ("Time", *wanted) if name not in raw.columns]
    if missing:
        raise Skip(f"{path.name}: missing columns {missing}")
    if raw.height == 0:
        raise Skip(f"{path.name}: no samples")
    if not all(raw[name].dtype.is_numeric() for name in wanted):
        raise Skip(f"{path.name}: non-numeric values in selected channels")
    times = raw["Time"].to_list()
    start = clock_seconds(times[0]) if start is None else start
    frame = raw.select(pl.Series("time_s", elapsed_seconds(times, start)), *wanted)
    return frame, start


def descriptor(dataset_id, case_id, label, columns, default_y, time_label="Elapsed time"):
    """Dataset descriptor; `columns` maps key -> (label, unit) and time_s comes first."""
    return {
        "dataset_id": dataset_id, "case_id": case_id, "label": label,
        "file": f"{dataset_id}.csv",
        "columns": [{"key": "time_s", "label": time_label, "data_type": "number", "unit": "s"}]
        + [{"key": key, "label": text, "data_type": "number", "unit": unit}
           for key, (text, unit) in columns.items()],
        "default_x": "time_s", "default_y": default_y,
    }


def text(row: dict, key: str) -> str | None:
    value = (row.get(key) or "").strip()
    return value if value and value != "-" else None


def number(row: dict, key: str) -> float | None:
    try:
        return float(text(row, key))
    except (TypeError, ValueError):
        return None


def tp_events(path: Path, start_clock: float) -> str | None:
    """Sentence quoting test-point script log entries relative to the first data sample."""
    if not path.is_file():
        return None
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name, _, stamp = line.partition(":")
        clock = re.search(r"\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2})\s*$", stamp)
        if name.strip() in TP_EVENTS and clock:
            offset = (clock_seconds(clock.group(1)) - start_clock + 43200) % 86400 - 43200
            events.append((offset, name.strip()))
    parts = "; ".join(f"“{name}” at t≈{offset:.0f} s" for offset, name in sorted(events))
    return f"Test-point script log entries (1 s resolution): {parts}." if events else None


def build_note(row: dict, highlight: str | None, events: str | None) -> str:
    facts = [f"Flame type: {text(row, 'Flame Type')}"]
    if text(row, "Xf"):
        facts.append(f"fuel mole fraction in the burner stream Xf={text(row, 'Xf')}")
    if text(row, "mHC (mg/s)"):
        facts.append(f"fuel flow {text(row, 'mHC (mg/s)')} mg/s")
    facts.append(f"hot ignition: {text(row, 'Hot ignition') or 'not recorded'}")
    burn = number(row, "burn time (sec)")
    if burn:
        facts.append(f"burn time {burn:g} s")
    ending = text(row, "Self-ext or flow term")
    if ending:
        facts.append(f"ended by {ENDINGS.get(ending, ending)}")
    if text(row, "Cool flame?"):
        facts.append(f"cool flame: {text(row, 'Cool flame?')}")
    sentences = [highlight, "Source table: " + "; ".join(facts) + "."]
    if text(row, "Comments"):
        sentences.append(f"Source comment: “{text(row, 'Comments')}”.")
    sentences += [
        events,
        "O₂ is the chamber mole fraction before the test; pressure is the table's "
        "approximate value converted from bar. Sensor channels are the full raw "
        "recording (documented rate 125 Hz); their units are not stated in the source "
        "files, so they are shown unconverted.",
    ]
    return " ".join(part for part in sentences if part)


def cfig_cases(data_dir: Path, entries: list[dict]):
    """Yield (case row, [(descriptor, frame), ...]) for ACME CFI-G normal-flame tests."""
    family = data_dir / FAMILY
    table_path = family / "csv" / TABLE
    if not table_path.is_file():
        raise SystemExit(f"Raw data not found: {table_path} (pass --data-dir)")
    table = pl.read_csv(table_path, infer_schema=False)
    rows: dict[str, list[dict]] = {}
    for row in table.rename({name: name.strip() for name in table.columns}).to_dicts():
        rows.setdefault(re.sub(r"\s+", "", row.get("Test") or ""), []).append(row)
    for entry in entries:
        test_id = entry["test_id"]
        case_id = f"CFIG-{test_id}"
        try:
            if len(rows.get(test_id, [])) != 1:
                raise Skip("not exactly one row in the experimental table")
            row = rows[test_id][0]
            prefix = f"PSI-159_Raw Data_{test_id}_"
            # A second recording (e.g. *_data2.csv) makes the link ambiguous.
            if len(list((family / "csv").glob(prefix + "data*.csv"))) != 1:
                raise Skip("not exactly one *_data.csv recording")
            data, start = read_series(family / "csv" / (prefix + "data.csv"), test_id, CHANNELS)
            exports = [(descriptor(
                f"{case_id}-sensors", case_id, "Sensor recording (raw channels)",
                {key: (label, None) for key, label in CHANNELS.items()},
                ["Radiometer3+30LoGain", "PMT1", "MFCA3FuelFlow"]), data)]
            try:
                setpoints, _ = read_series(
                    family / "csv" / (prefix + "mfcSetpoints.csv"), test_id, SETPOINTS, start)
                exports.append((descriptor(
                    f"{case_id}-mfc-setpoints", case_id, "MFC setpoints (commanded)",
                    {key: (label, "V") for key, label in SETPOINTS.items()},
                    list(SETPOINTS)), setpoints))
            except Skip as reason:
                print(f"  {case_id}: no setpoint dataset ({reason})")
        except Skip as reason:
            print(f"SKIPPED {case_id}: {reason}")
            continue
        fuel_label = text(row, "Fuel")
        o2 = number(row, "XO2 before test")
        bar = number(row, "approx p (bar)")
        o2_pct = None if o2 is None else round(o2 * 100, 3)
        kpa = None if bar is None else round(bar * BAR_TO_KPA, 3)
        conditions = ", ".join(part for part in (
            fuel_label, None if o2_pct is None else f"{o2_pct:.1f}% O₂",
            None if kpa is None else f"{kpa:.0f} kPa") if part)
        yield {
            "case_id": case_id,
            "title": f"CFI-G {test_id}: {conditions}",
            "investigation": INVESTIGATION,
            # "30% butane" in the table becomes the base fuel; dilution stays in the note.
            "fuel": fuel_label and re.sub(r"^\d+%\s*", "", fuel_label),
            "o2_pct": o2_pct,
            "pressure_kpa": kpa,
            "source_label": f"NASA PSI-159 ACME CFI-G, test point {test_id}: "
                            f"{TABLE}; {prefix}data.csv",
            "source_url": None,
            "note": build_note(row, entry.get("highlight"), tp_events(
                family / "txt" / (prefix + "TP.txt"), start)),
        }, exports


def prepare(data_dir: Path, out_dir: Path) -> None:
    from data_prep.flex import flex_cases  # needs openpyxl; imported late to avoid a cycle

    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text("utf-8"))
    csv_dir = out_dir / "csv"
    shutil.rmtree(csv_dir, ignore_errors=True)
    csv_dir.mkdir(parents=True)
    cases, datasets = [], []
    for case, exports in (*cfig_cases(data_dir, manifest["cases"]),
                          *flex_cases(data_dir, manifest.get("flex_cases", []))):
        for meta, frame in exports:
            assert frame.columns == [column["key"] for column in meta["columns"]]
            frame.write_csv(csv_dir / meta["file"])
            datasets.append(meta)
        cases.append(case)
    if not cases:
        raise SystemExit("No case could be prepared.")
    schema = {key: pl.String for key in cases[0]} | {
        "o2_pct": pl.Float64, "pressure_kpa": pl.Float64}
    catalogue = pl.DataFrame(cases, schema=schema).sort("case_id")
    catalogue.write_parquet(out_dir / "catalogue.parquet")
    (out_dir / "datasets.json").write_text(
        json.dumps({"datasets": datasets}, ensure_ascii=False, indent=1), "utf-8")
    size = sum(path.stat().st_size for path in csv_dir.iterdir()) / 1e6
    print(f"{catalogue.height} cases, {len(datasets)} datasets, {size:.1f} MB of CSV -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=BACKEND.parent / "data_nasa",
                        help="folder holding ACME-CFI-G_PSI-159/ and CFI_PSI-39/")
    parser.add_argument("--out-dir", type=Path, default=BACKEND / "data" / "curated")
    args = parser.parse_args()
    prepare(args.data_dir.resolve(), args.out_dir.resolve())
