"""Adapter: ACME Flame Design (PSI-10) normal-flame tests as cases and datasets.

Same recording format as ACME CFI-G, plus a table with fuel fraction, adiabatic flame
temperature (Tad) and mixture fraction (Zst). A case is linked to its recording by the
`Test Point:` field in the raw file's preamble, checked against the table's test ID.
Selection lives in data_prep/manifest_projects.json ("flame_design").

    python3 -m data_prep.flame_design select   # print a per-day selection to paste there
"""

import argparse
import json
from pathlib import Path
import re

import polars as pl

from data_prep import prepare
from data_prep.prepare import (
    BAR_TO_KPA, CHANNEL_INFO, EXPLANATION_ORIGIN, SENSOR_NOTE, SETPOINT_NOTE, SETPOINTS, Skip,
    acme_scales, convert_acme, descriptor, events_sentence, number, read_series, text, tp_events)

FAMILY = "ACME-Flame-Design_PSI-10"
TABLE = "PSI-10_Experimental table_Flame Design_Normal Flame.csv"
INVESTIGATION = "ACME Flame Design"
ATM_TO_KPA = 101.325
FUELS = {"C2H4": "ethylene", "C3H8": "propane"}  # chemical formulas as written in the table
# Standard ACME radiometer array (not the CFI-G one); the other channels are shared.
INFO = {key: value for key, value in CHANNEL_INFO.items() if not key.startswith("Radiometer")} | {
    "Radiometer+60": ("Radiometer +60 (raw)", None),
    "Radiometer+30": ("Radiometer +30 (raw)", None),
    "RadiometerCenter": ("Radiometer center (raw)", None),
    "Radiometer-30": ("Radiometer -30 (raw)", None),
    "Radiometer-60": ("Radiometer -60 (raw)", None),
    "RadiometerOffset": ("Offset radiometer (raw)", None),
}


def channels_for(path: Path) -> tuple[dict, list[str]]:
    """Channel set of a recording: later runs carry the CFI-G radiometer array instead."""
    header = path.read_bytes().split(b"\n", 2)[1].decode("utf-8", errors="replace")
    if "RadiometerCenter" in header.split(","):
        return INFO, ["RadiometerCenter", "PMT1", "MFCA3FuelFlow"]
    return CHANNEL_INFO, ["Radiometer3+30LoGain", "PMT1", "MFCA3FuelFlow"]


def table(data_dir: Path) -> dict[str, list[dict]]:
    path = data_dir / FAMILY / "csv" / TABLE
    if not path.is_file():
        return {}
    raw = pl.read_csv(path, infer_schema=False)
    rows: dict[str, list[dict]] = {}
    for row in raw.rename({name: name.strip() for name in raw.columns}).to_dicts():
        rows.setdefault(re.sub(r"\s+", "", row.get("Test") or ""), []).append(row)
    return rows


def explain(row: dict, fuel: str | None, o2_pct: float | None, kpa: float | None) -> str:
    where = " and ".join(part for part in (
        None if o2_pct is None else f"{o2_pct:.1f}% oxygen",
        None if kpa is None else f"about {kpa:.0f} kPa") if part)
    sentence = f"Normal (fuel-inside) flame test with {fuel or text(row, 'Fuel')}" + (
        f" in {where}." if where else ".")
    tad = number(row, "Tad (K)")
    if tad:
        sentence += f" The table gives an adiabatic flame temperature of {tad:.0f} K."
    return sentence


def flame_design_cases(data_dir: Path, entries: list[dict]):
    if not entries:
        return
    rows = table(data_dir)
    if not rows:
        print(f"SKIPPED all Flame Design cases: {data_dir / FAMILY / 'csv' / TABLE} not found")
        return
    family = data_dir / FAMILY
    for entry in entries:
        test_id = entry["test_id"]
        case_id = f"FD-{test_id}"
        prefix = f"PSI-10_Raw Data_{test_id}_"
        try:
            if len(rows.get(test_id, [])) != 1:
                raise Skip("not exactly one row in the experimental table")
            row = rows[test_id][0]
            if len(list((family / "csv").glob(prefix + "data*.csv"))) != 1:
                raise Skip("not exactly one *_data.csv recording")
            data_path = family / "csv" / (prefix + "data.csv")
            info, default_y = channels_for(data_path)
            data, start = read_series(data_path, test_id, {key: label for key, (label, _) in info.items()})
            data = convert_acme(data, acme_scales(data_path))
            exports = [(descriptor(
                f"{case_id}-sensors", case_id, "Sensor recording (converted where documented)",
                dict(info), default_y, SENSOR_NOTE), data)]
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
        fuel = FUELS.get(text(row, "Fuel") or "")
        o2 = number(row, "XO2 (init)")
        atm = number(row, "pmean (atm)")
        o2_pct = None if o2 is None else round(o2 * 100, 3)
        kpa = None if atm is None else round(atm * ATM_TO_KPA, 3)
        z_st = number(row, "mean Zst")  # "RAMPED" runs have no single value: null
        facts = [f"fuel {text(row, 'Fuel')}"]
        for label, key, unit in (("fuel mole fraction Xfuel", "Xfuel", ""), ("fuel flow", "mfuel (mg/s)", " mg/s"),
                                 ("N₂ flow", "mN2 (mg/s)", " mg/s"), ("total flow", "mtot (mg/s)", " mg/s")):
            value = number(row, key)
            facts.append(f"{label} {value:.4g}{unit}" if value is not None
                         else f"{label} {text(row, key) or 'not given'}")
        final = number(row, "XO2 (final)")
        if final is not None:
            facts.append(f"O₂ after test {final * 100:.2f}%")
        pinit, pfinal = number(row, "pinit (atm)"), number(row, "pfinal (atm)")
        if pinit is not None and pfinal is not None:
            facts.append(f"pressure {pinit * ATM_TO_KPA:.1f} → {pfinal * ATM_TO_KPA:.1f} kPa during the test")
        tad = number(row, "Tad (K)")
        if tad is not None:
            facts.append(f"adiabatic flame temperature {tad:.0f} K")
        note = " ".join(part for part in (
            entry.get("highlight"),
            "Source table: " + "; ".join(facts) + ".",
            events_sentence(events),
            "Pressure filter = the table's mean pressure in atm × 101.325. Values marked RAMPED in the "
            "table (flow changed during the run) are left unknown. Sensor channels are described in the "
            "dataset notes.") if part)
        conditions = ", ".join(part for part in (
            fuel or text(row, "Fuel"), None if o2_pct is None else f"{o2_pct:.1f}% O₂",
            None if kpa is None else f"{kpa:.0f} kPa") if part)
        yield {
            "case_id": case_id,
            "title": f"Flame Design {test_id}: {conditions}",
            "investigation": INVESTIGATION,
            "fuel": fuel,
            "o2_pct": o2_pct,
            "pressure_kpa": kpa,
            "z_st": z_st,
            "source_label": f"NASA PSI-10 ACME Flame Design, test point {test_id}: {TABLE}; {prefix}data.csv",
            "source_url": None,
            "note": note,
            "explanation": explain(row, fuel, o2_pct, kpa),
            "explanation_origin": EXPLANATION_ORIGIN,
            "_events": [{"label": name, "time_s": round(offset, 3), "dataset_id": f"{case_id}-sensors",
                         "source": prefix + "TP.txt"} for offset, name in events],
        }, exports


def select(data_dir: Path, per_group: int = 5) -> list[str]:
    """Usable tests (single, NUL-free recording under 7 MB), up to `per_group` per fuel and day."""
    rows, chosen, taken = table(data_dir), [], {}
    for test_id, found in sorted(rows.items()):
        path = data_dir / FAMILY / "csv" / f"PSI-10_Raw Data_{test_id}_data.csv"
        if len(found) != 1 or not path.is_file() or path.stat().st_size > 7_000_000:
            continue
        if b"\x00" in path.read_bytes() or found[0].get("Fuel") not in FUELS:
            continue
        group = (found[0]["Fuel"], found[0]["Day"])
        if taken.get(group, 0) < per_group:
            taken[group] = taken.get(group, 0) + 1
            chosen.append(test_id)
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["select"])
    parser.add_argument("--data-dir", type=Path, default=prepare.BACKEND.parent / "data_nasa")
    parser.add_argument("--per-group", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps([{"test_id": test_id} for test_id in select(args.data_dir.resolve(), args.per_group)]))
