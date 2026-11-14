"""B01: build the curated demo catalogue and measurement Parquet files from raw NASA PSI files.

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
# Raw column -> (label, unit). Pressure and flow are converted with the formulas of NASA's
# "ACME non-imaging data file description" (see convert_acme); the other channels keep the
# raw recorded values and have no stated unit.
CHANNEL_INFO = {
    "MFCA1StaticMixerFlow": ("MFC A1 static mixer (oxidizer) flow", "SLPM"),
    "MFCA2BypassFlow": ("MFC A2 bypass (inert) flow", "SLPM"),
    "MFCA3FuelFlow": ("MFC A3 fuel flow", "SLPM"),
    "IonCurrent": ("Ion current signal (raw voltage)", "V"),
    "PTA1": ("Outer coflow pressure (PTA1)", "kPa"),
    "PTA2": ("Inner coflow / burner pressure (PTA2)", "kPa"),
    "PTA3": ("Chamber pressure (PTA3)", "kPa"),
    "PMT1": ("Photomultiplier PMT1 current (310 nm band)", "µA"),
    "PMT2": ("Photomultiplier PMT2 current (broadband, neutral density)", "µA"),
    "PMT3": ("Photomultiplier PMT3 current (430 nm band)", "µA"),
    "TCA": ("Thermocouple TCA (raw)", None),
    "TCB": ("Thermocouple TCB (raw)", None),
    "TCC": ("Thermocouple TCC (raw)", None),
    "TCD": ("Thermocouple TCD (raw)", None),
    "TCBurner": ("Burner thermocouple (raw)", None),
    "Radiometer3+60LoGain": ("Radiometer +60, low gain (raw)", None),
    "Radiometer3+30LoGain": ("Radiometer +30, low gain (raw)", None),
    "Radiometer3-10LoGain": ("Radiometer -10, low gain (raw)", None),
    "RadiometerOffset": ("Offset radiometer (raw)", None),
}
CHANNELS = {key: label for key, (label, _) in CHANNEL_INFO.items()}
PSI_TO_KPA = 6.894757293168
FLOWS = {"MFCA1StaticMixerFlow": "MFCA1", "MFCA2BypassFlow": "MFCA2", "MFCA3FuelFlow": "MFCA3"}


def acme_scales(path: Path) -> dict[str, float]:
    """MFC full-scale flows (SLPM) from the first line of a raw ACME recording."""
    first = path.read_bytes().split(b"\n", 1)[0].decode("utf-8", errors="replace")
    return {name: float(value) for name, value in re.findall(r"(MFCA\d) Full Scale: ([\d.]+)", first)}


def convert_acme(frame: pl.DataFrame, scales: dict[str, float]) -> pl.DataFrame:
    """Pressure to kPa and flows to SLPM, as in NASA's ACME non-imaging data file description:
    psia = (V - 0.5) * 14.5038 * 10 / 4;  SLPM = V / 5 * full scale. Missing scales skip a flow."""
    exprs = [((pl.col(key) - 0.5) * 14.5038 * 10 / 4 * PSI_TO_KPA).alias(key)
             for key in ("PTA1", "PTA2", "PTA3") if key in frame.columns]
    exprs += [(pl.col(key) / 5 * scales[mfc]).alias(key)
              for key, mfc in FLOWS.items() if key in frame.columns and mfc in scales]
    return frame.with_columns(exprs)


SETPOINTS = {f"MFCA{n}Setpoint(V)": f"MFC A{n} setpoint" for n in (1, 2, 3)}
SENSOR_NOTE = (
    "Full recording, every sample kept (documented rate 125 Hz); no sampling or interpolation. "
    "Pressures (kPa), MFC flows (SLPM at the device's full scale) and PMT currents (µA) use the "
    "conversions in NASA's ACME non-imaging data file description; ion current is the signal "
    "voltage. Thermocouple and radiometer channels keep the raw recorded values because the "
    "source files state no unit. time_s counts from the first sample."
)
SETPOINT_NOTE = (
    "Commanded setpoint log as recorded; entries with the same timestamp are in the source. "
    "time_s shares the zero of the sensor recording, so the first entries can be slightly negative."
)
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


def descriptor(dataset_id, case_id, label, columns, default_y, note,
               time_label="Elapsed time"):
    """Dataset descriptor; `columns` maps key -> (label, unit) and time_s comes first.

    `note` states sampling and quality facts a chart reader needs.
    """
    return {
        "dataset_id": dataset_id, "case_id": case_id, "label": label,
        "file": f"{dataset_id}.parquet", "note": note,
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


EVENT_METHOD = (
    "Entries copied from the test-point script log (*_TP.txt), which records wall-clock times "
    "at 1 s resolution. time_s = log time minus the clock time of the first sensor sample, so "
    "it shares the zero of the case's sensor dataset and is accurate to about 1 s. "
    "'Extinction detected' is the test script's own log entry, not a detection made here. "
    "Cases without a log, and entries the log leaves blank, have no event; nothing is inferred."
)


def tp_events(path: Path, start_clock: float) -> list[tuple[float, str]]:
    """(seconds since the first data sample, log entry name), ordered by time."""
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name, _, stamp = line.partition(":")
        clock = re.search(r"\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2})\s*$", stamp)
        if name.strip() in TP_EVENTS and clock:
            offset = (clock_seconds(clock.group(1)) - start_clock + 43200) % 86400 - 43200
            events.append((offset, name.strip()))
    return sorted(events)


def events_sentence(events: list[tuple[float, str]]) -> str | None:
    parts = "; ".join(f"“{name}” at t≈{offset:.0f} s" for offset, name in events)
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


EXPLANATION_ORIGIN = "template"  # fixed sentence template over source fields; no AI text


def explain_cfig(row: dict, o2_pct: float | None, kpa: float | None) -> str:
    """B15: one or two plain sentences built only from the experimental table row."""
    where = " and ".join(part for part in (
        None if o2_pct is None else f"{o2_pct:.1f}% oxygen",
        None if kpa is None else f"about {kpa:.0f} kPa") if part)
    opening = f"Gas flame test with {text(row, 'Fuel')}" + (f" in {where}." if where else ".")
    ignition = (text(row, "Hot ignition") or "").lower()
    burn = number(row, "burn time (sec)")
    ending = {"SE": "went out by itself", "FT": "was ended by stopping the fuel flow"}.get(
        text(row, "Self-ext or flow term"))
    if ignition == "yes":
        outcome = "It ignited" + (f", burned for {burn:g} s" if burn else "") + (
            f" and {ending}." if ending else ".")
    elif ignition == "no":
        outcome = "The source table records no hot ignition."
    else:
        outcome = "The source table does not say clearly whether it ignited."
    cool = text(row, "Cool flame?")
    if cool and cool.lower().startswith("yes"):
        outcome += " The table marks a cool flame in this run."
    return f"{opening} {outcome}"


def cfig_table(data_dir: Path) -> dict[str, list[dict]]:
    """Rows of the normal-flame experimental table by test ID (spaces removed)."""
    table_path = data_dir / FAMILY / "csv" / TABLE
    if not table_path.is_file():
        raise SystemExit(f"Raw data not found: {table_path} (pass --data-dir)")
    table = pl.read_csv(table_path, infer_schema=False)
    rows: dict[str, list[dict]] = {}
    for row in table.rename({name: name.strip() for name in table.columns}).to_dicts():
        rows.setdefault(re.sub(r"\s+", "", row.get("Test") or ""), []).append(row)
    return rows


def cfig_conditions(row: dict) -> tuple[float | None, float | None, float | None]:
    """(o2_pct, pressure_kpa, z_st) of one table row; unknown stays None."""
    o2 = number(row, "XO2 before test")
    bar = number(row, "approx p (bar)")
    return (None if o2 is None else round(o2 * 100, 3),
            None if bar is None else round(bar * BAR_TO_KPA, 3),
            number(row, "Zst"))  # as tabulated by the source


def cfig_cases(data_dir: Path, entries: list[dict]):
    """Yield (case row, [(descriptor, frame), ...]) for ACME CFI-G normal-flame tests."""
    family = data_dir / FAMILY
    rows = cfig_table(data_dir)
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
            data_path = family / "csv" / (prefix + "data.csv")
            data, start = read_series(data_path, test_id, CHANNELS)
            data = convert_acme(data, acme_scales(data_path))
            exports = [(descriptor(
                f"{case_id}-sensors", case_id, "Sensor recording (converted where documented)",
                dict(CHANNEL_INFO),
                ["Radiometer3+30LoGain", "PMT1", "MFCA3FuelFlow"], SENSOR_NOTE), data)]
            try:
                setpoints, _ = read_series(
                    family / "csv" / (prefix + "mfcSetpoints.csv"), test_id, SETPOINTS, start)
                exports.append((descriptor(
                    f"{case_id}-mfc-setpoints", case_id, "MFC setpoints (commanded)",
                    {key: (label, "V") for key, label in SETPOINTS.items()},
                    list(SETPOINTS), SETPOINT_NOTE), setpoints))
            except Skip as reason:
                print(f"  {case_id}: no setpoint dataset ({reason})")
        except Skip as reason:
            print(f"SKIPPED {case_id}: {reason}")
            continue
        events = tp_events(family / "txt" / (prefix + "TP.txt"), start)
        fuel_label = text(row, "Fuel")
        o2_pct, kpa, z_st = cfig_conditions(row)
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
            "z_st": z_st,
            "source_label": f"NASA PSI-159 ACME CFI-G, test point {test_id}: "
                            f"{TABLE}; {prefix}data.csv",
            "source_url": None,
            "note": build_note(row, entry.get("highlight"), events_sentence(events)),
            "explanation": explain_cfig(row, o2_pct, kpa),
            "explanation_origin": EXPLANATION_ORIGIN,
            # B13: popped by prepare() into events.json.
            "_events": [{"label": name, "time_s": round(offset, 3),
                         "dataset_id": f"{case_id}-sensors", "source": prefix + "TP.txt"}
                        for offset, name in events],
        }, exports


def prepare(data_dir: Path, out_dir: Path) -> None:
    from data_prep.flame_design import flame_design_cases
    from data_prep.saffire import saffire_cases
    from data_prep.flex import flex_cases  # needs openpyxl; imported late to avoid a cycle
    from data_prep.similarity import METRIC, nearest_neighbours

    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text("utf-8"))
    more = json.loads((Path(__file__).parent / "manifest_projects.json").read_text("utf-8"))
    shutil.rmtree(out_dir / "csv", ignore_errors=True)  # output of earlier versions
    parquet_dir = out_dir / "parquet"
    shutil.rmtree(parquet_dir, ignore_errors=True)
    parquet_dir.mkdir(parents=True)
    cases, datasets, events = [], [], {}
    for case, exports in (*cfig_cases(data_dir, manifest["cases"]),
                          *flex_cases(data_dir, manifest.get("flex_cases", [])),
                          *flame_design_cases(data_dir, more.get("flame_design", [])),
                          *saffire_cases(data_dir, more.get("saffire", []))):
        for meta, frame in exports:
            assert frame.columns == [column["key"] for column in meta["columns"]]
            frame.write_parquet(parquet_dir / meta["file"])
            datasets.append(meta)
        events[case["case_id"]] = case.pop("_events", [])
        cases.append(case)
    if not cases:
        raise SystemExit("No case could be prepared.")
    schema = {key: pl.String for key in cases[0]} | {
        "o2_pct": pl.Float64, "pressure_kpa": pl.Float64, "z_st": pl.Float64}
    catalogue = pl.DataFrame(cases, schema=schema).sort("case_id")
    catalogue.write_parquet(out_dir / "catalogue.parquet")
    (out_dir / "datasets.json").write_text(
        json.dumps({"datasets": datasets}, ensure_ascii=False, indent=1), "utf-8")
    (out_dir / "events.json").write_text(json.dumps(
        {"method": EVENT_METHOD, "events": events}, ensure_ascii=False, indent=1), "utf-8")
    (out_dir / "neighbors.json").write_text(json.dumps(
        {"metric": METRIC, "neighbors": nearest_neighbours(catalogue)}, indent=1), "utf-8")
    registry = json.loads((Path(__file__).parent / "projects.json").read_text("utf-8"))
    present = set(catalogue["investigation"])
    shown = [p for p in registry["projects"] if present & set(p["investigations"])]
    for project in registry["projects"]:
        if project not in shown:
            print(f"Project {project['project_id']} has no prepared cases and is not published.")
    (out_dir / "projects.json").write_text(
        json.dumps({"projects": shown}, ensure_ascii=False, indent=1), "utf-8")
    context = json.loads((Path(__file__).parent / "research_context.json").read_text("utf-8"))
    for name, item in context["investigations"].items():
        allowed = set(item["tags"]) <= {"prevention", "detection", "suppression", "unknown"}
        if not allowed or (item["tags"] != ["unknown"] and not item["evidence"]):
            raise SystemExit(f"research_context.json: {name} needs valid tags with evidence")
    (out_dir / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=1), "utf-8")
    size = sum(path.stat().st_size for path in parquet_dir.iterdir()) / 1e6
    print(f"{catalogue.height} cases, {len(datasets)} datasets, {size:.1f} MB of Parquet -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=BACKEND.parent / "data_nasa",
                        help="folder holding ACME-CFI-G_PSI-159/ and CFI_PSI-39/")
    parser.add_argument("--out-dir", type=Path, default=BACKEND / "data" / "curated")
    args = parser.parse_args()
    prepare(args.data_dir.resolve(), args.out_dir.resolve())
