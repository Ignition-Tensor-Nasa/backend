"""Adapter: SAME-R (PSI-101), smoke detectors watching smoke from heated spacecraft materials.

One case per housekeeping (HKP) recording of the 2010 flight (file names `HKPData_gmt<day>_<n>.csv`).
The results workbook lists materials per GMT day but not per file, so the materials tested on
that day are given in the note and nothing is assigned to a single file.
"""

import argparse
import json
from pathlib import Path
import re

from openpyxl import load_workbook
import polars as pl

from data_prep import prepare
from data_prep.prepare import EXPLANATION_ORIGIN, PSI_TO_KPA, Skip, descriptor

FAMILY = "SAME-R_PSI-101"
INVESTIGATION = "SAME-R"
HEATING_ON, HEATING_OFF = "Sample heating on", "Sample heating off"
# source column -> (label, unit); units are the ones in the source column names.
CHANNELS = {
    "GasPlumeTemperature(DegC)": ("Gas plume temperature", "°C"),
    "SampleResistance(Ohms)": ("Sample resistance", "Ω"),
    "ISSDetectorObscurationMeasurement(Volts)": ("ISS smoke detector, obscuration", "V"),
    "ISSDetectorScatterMeasurement(Volts)": ("ISS smoke detector, scatter", "V"),
    "STSDetectorMeasurement(Volts)": ("Shuttle (STS) smoke detector", "V"),
    "IonDetectorAMeasurement(Volts)": ("Ionization detector A", "V"),
    "IonDetectorBMeasurement(Volts)": ("Ionization detector B", "V"),
    "PTrakMeasurement(PtPerCC)": ("P-Trak particle count", "particles/cm³"),
    "DustTrakAMeasurement(MgPerM3)": ("DustTrak A mass concentration", "mg/m³"),
    "DustTrakBMeasurement(MgPerM3)": ("DustTrak B mass concentration", "mg/m³"),
    "MFCMassFlowRate(LPM)": ("Flow rate (mass flow controller)", "L/min"),
}
NOTE = (
    "Housekeeping (HKP) recording at about 1 sample per second, every row kept; time_s counts from "
    "the first row. Units are the ones in the source column names. Smoke-detector values are the "
    "detectors' output in volts, not a smoke concentration. The file can contain several test "
    "numbers (setup and the run itself) and, in a few files, a repeated header row, which is dropped."
)


def materials_by_day(data_dir: Path) -> dict[int, list[str]]:
    path = data_dir / FAMILY / "xlsx" / "Combined SAMER Results 06 02 14.xlsx"
    if not path.is_file():
        return {}
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook["SAME R Results"].iter_rows(values_only=True))
    finally:
        workbook.close()
    found: dict[int, set[str]] = {}
    for row in rows[2:]:
        if row[1] is None or row[7] is None:
            continue
        day = re.match(r"\s*(\d{3})", str(row[7]))
        if day:
            found.setdefault(int(day.group(1)), set()).add(str(row[1]).strip())
    return {day: sorted(names) for day, names in found.items()}


def read_recording(path: Path) -> tuple[pl.DataFrame, dict]:
    raw = pl.read_csv(path.read_bytes().replace(b"\x00", b""), infer_schema=False, truncate_ragged_lines=True)
    missing = [name for name in ("Time", "HeatingInProgress", *CHANNELS) if name not in raw.columns]
    if missing:
        raise Skip(f"{path.name}: missing columns {missing}")
    numeric = raw.select(pl.col(name).cast(pl.Float64, strict=False)
                         for name in ("Time", "HeatingInProgress", *CHANNELS))
    numeric = numeric.filter(pl.col("Time").is_not_null())  # drops repeated header rows
    if numeric.height < 50:
        raise Skip(f"{path.name}: fewer than 50 samples")
    start = numeric["Time"][0]
    heating = numeric.filter(pl.col("HeatingInProgress") == 1)["Time"] - start
    frame = numeric.select((pl.col("Time") - start).alias("time_s"), *CHANNELS)
    facts = {"heating": (heating.min(), heating.max()) if len(heating) else None,
             "duration": frame["time_s"].max(),
             "pressure": raw["SampleSidePressure[AT20](PSIA)"].cast(pl.Float64, strict=False).median()
             if "SampleSidePressure[AT20](PSIA)" in raw.columns else None}
    return frame, facts


def same_cases(data_dir: Path, entries: list[dict]):
    if not entries:
        return
    folder = data_dir / FAMILY / "csv"
    days = materials_by_day(data_dir)
    for path in sorted(folder.glob("HKPData_gmt*.csv")):
        stem = path.stem.removeprefix("HKPData_")
        day = int(re.match(r"gmt(\d+)", stem).group(1))
        case_id = f"SAMER-{stem}"
        try:
            frame, facts = read_recording(path)
        except (Skip, pl.exceptions.ComputeError) as reason:
            print(f"SKIPPED {case_id}: {reason}")
            continue
        names = days.get(day)
        kpa = None if facts["pressure"] is None else round(facts["pressure"] * PSI_TO_KPA, 3)
        heating = facts["heating"]
        events = [] if heating is None else [
            {"label": HEATING_ON, "time_s": round(heating[0], 3), "dataset_id": f"{case_id}-detectors",
             "source": "first row with HeatingInProgress = 1"},
            {"label": HEATING_OFF, "time_s": round(heating[1], 3), "dataset_id": f"{case_id}-detectors",
             "source": "last row with HeatingInProgress = 1"}]
        date = f"GMT day {day} of 2010"
        yield {
            "case_id": case_id,
            "title": f"SAME-R {date}, recording {stem.split('_')[1]}: smoke detectors vs heated sample",
            "investigation": INVESTIGATION, "fuel": None, "o2_pct": None, "pressure_kpa": kpa, "z_st": None,
            "source_label": f"NASA PSI-101 SAME-R: {path.name}; materials from 'Combined SAMER Results "
                            "06 02 14.xlsx', sheet 'SAME R Results'",
            "source_url": None,
            "note": " ".join(part for part in (
                f"Recording {stem} ({date}), {facts['duration']:.0f} s.",
                (f"Materials listed in the results sheet for this GMT day: {', '.join(names)}; the sheet "
                 "does not say which of them this file holds." if names else
                 "The results sheet lists no material for this GMT day."),
                None if heating is None else f"Sample heating is flagged from {heating[0]:.0f} s to "
                                             f"{heating[1]:.0f} s.",
                "Pressure filter value: median sample-side pressure of the apparatus (not an ambient "
                "cabin value). Oxygen is not recorded.") if part),
            "explanation": ("A heated spacecraft-material sample produces smoke while an ISS detector, a "
                            "shuttle detector, two ionization detectors and particle counters record it."
                            + ("" if heating is None else
                               f" Heating is flagged from {heating[0]:.0f} s to {heating[1]:.0f} s.")),
            "explanation_origin": EXPLANATION_ORIGIN, "_events": events,
        }, [(descriptor(f"{case_id}-detectors", case_id, "Detectors and plume (about 1 Hz)", CHANNELS,
                        ["ISSDetectorScatterMeasurement(Volts)", "PTrakMeasurement(PtPerCC)",
                         "GasPlumeTemperature(DegC)"], NOTE), frame)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=prepare.BACKEND.parent / "data_nasa")
    args = parser.parse_args()
    print(json.dumps(materials_by_day(args.data_dir.resolve()), indent=1))
