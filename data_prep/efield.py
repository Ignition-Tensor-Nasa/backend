"""Adapter: ACME E-FIELD Flames (PSI-22), flames steered by an electric field.

Per-day fuel and burner come from Table 1 of NASA's "E-FIELD Flames data interpretation
document" (PSI-22 pdf); per-run dilution and field sweep from the run's *_VoltageSweep.txt;
pressure from the recorded chamber pressure. Selection is in data_prep/manifest_projects.json.
"""

import argparse
import json
from pathlib import Path
import re

import polars as pl

from data_prep import prepare
from data_prep.flame_design import INFO as STANDARD_INFO
from data_prep.prepare import (
    EXPLANATION_ORIGIN, PSI_TO_KPA, SENSOR_NOTE, Skip, acme_scales, convert_acme, descriptor,
    events_sentence, read_series, tp_events)

FAMILY = "ACME-E-FIELD-Flames_PSI-22"
INVESTIGATION = "ACME E-FIELD Flames"
# Table 1 of the data interpretation document: test day -> (fuel, burner).
DAYS = {
    **{d: ("methane", "coflow") for d in ("18073", "18078", "18080", "18102", "18106", "18113", "18130")},
    **{d: ("ethylene", "coflow") for d in ("18087", "18092", "18095", "18137")},
    **{d: ("methane", "gas-jet") for d in ("18269", "18282", "18298", "18312")},
    **{d: ("ethylene", "gas-jet") for d in ("18274", "18285", "18290", "18302", "18319")},
}
INFO = dict(STANDARD_INFO) | {"EfieldVoltage": ("Electric-field mesh voltage", "kV")}
LABELS = {key: label for key, (label, _) in INFO.items()}
FIELD_NOTE = (
    "Derived for this project, not a NASA series: the ion-current signal (volts) averaged in bins of "
    "0.5 kV of mesh voltage, over the time between the first logged flame event and 'MFCs off' of the "
    "test-point log. min/max are the extremes inside each bin; n is the number of samples. Bins "
    "mix rising and falling parts of the voltage sweep."
)
ONE_ATM_NOTE = (
    "Oxygen filter value: the document states the coflow was 21% oxygen in all tests. Pressure filter "
    "value: median recorded chamber pressure (PTA3)."
)


def sweep_parameters(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    pairs = re.findall(r"^(\w+):\s*(-?[\d.]+)\s*,?\s*$", path.read_text("utf-8", errors="replace"), re.M)
    return {key: float(value) for key, value in pairs}


def flame_window(path: Path) -> tuple[float, float] | None:
    """(first logged flame event, MFCs off) as clock seconds, or None when the log lacks them."""
    if not path.is_file():
        return None
    marks = {}
    for line in path.read_text("utf-8", errors="replace").splitlines():
        name, _, stamp = line.partition(":")
        clock = re.search(r"\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2})\s*$", stamp)
        if clock:
            marks[name.strip()] = prepare.clock_seconds(clock.group(1))
    start = marks.get("Ignition and retraction", marks.get("Flame stabilization #1"))
    end = marks.get("MFCs off")
    return (start, end) if start is not None and end is not None and end > start else None


def field_response(frame: pl.DataFrame, start_clock: float, window: tuple[float, float] | None) -> pl.DataFrame | None:
    if window is None:
        return None
    lo, hi = window[0] - start_clock, window[1] - start_clock
    inside = frame.filter(pl.col("time_s").is_between(lo, hi)).select(
        ((pl.col("EfieldVoltage") * 2).round(0) / 2).alias("field_kv"), pl.col("IonCurrent").alias("ion"))
    if inside.height < 100:
        return None
    return (inside.group_by("field_kv").agg(
        pl.col("ion").mean().alias("ion_mean"), pl.col("ion").min().alias("ion_min"),
        pl.col("ion").max().alias("ion_max"), pl.len().alias("n")).sort("field_kv")
        .rename({"field_kv": "time_s"}))


def efield_cases(data_dir: Path, entries: list[dict]):
    family = data_dir / FAMILY
    for entry in entries:
        test_id = entry["test_id"]
        case_id = f"EF-{test_id}"
        data_path = family / "csv" / f"{test_id}_data.csv"
        try:
            if test_id[:5] not in DAYS:
                raise Skip("test day is not in Table 1 of the data interpretation document")
            data, start = read_series(data_path, test_id, LABELS)
            data = convert_acme(data, acme_scales(data_path))
            field = data["EfieldVoltage"] * 2486.1795 + 141.9733
            data = data.with_columns((field / 1000).alias("EfieldVoltage"))
            if max(abs(data["EfieldVoltage"].min()), abs(data["EfieldVoltage"].max())) < 1:
                raise Skip("no electric field above 1 kV in this run")
        except Skip as reason:
            print(f"SKIPPED {case_id}: {reason}")
            continue
        fuel, burner = DAYS[test_id[:5]]
        sweep = sweep_parameters(family / "txt" / f"{test_id}_VoltageSweep.txt")
        log = family / "txt" / f"{test_id}_TP.txt"
        events = tp_events(log, start)
        extremes = (data["EfieldVoltage"].min(), data["EfieldVoltage"].max())
        sign = "negative" if abs(extremes[0]) > abs(extremes[1]) else "positive"
        peak = extremes[0] if sign == "negative" else extremes[1]
        kpa = round(data["PTA3"].median(), 3)
        dilution = sweep.get("concentrationInitial")
        facts = [f"fuel {fuel}", f"{burner} burner (by test day)"]
        if dilution is not None:
            facts.append(f"fuel concentration {dilution * 100:.0f}% in nitrogen" + (
                f" → {sweep['concentrationFinal'] * 100:.0f}%" if sweep.get("concentrationFinal") not in (None, dilution) else ""))
        if "efieldMinVoltage" in sweep:
            facts.append(f"commanded sweep {sweep['efieldMinVoltage']:.0f} to {sweep['efieldMaxVoltage']:.0f} V, "
                         f"{sweep.get('efieldCycles', 0):.0f} cycle(s)")
        facts.append(f"measured field extremes {extremes[0]:.1f} / {extremes[1]:.1f} kV ({sign} mesh)")
        exports = [(descriptor(
            f"{case_id}-sensors", case_id, "Sensor recording (converted where documented)", dict(INFO),
            ["IonCurrent", "EfieldVoltage", "PMT1"],
            SENSOR_NOTE + " The mesh voltage is converted to kV with NASA's empirical equation "
            "(V = signal × 2486.18 + 141.97)."), data)]
        response = field_response(data, start, flame_window(log))
        if response is not None:
            exports.append((descriptor(
                f"{case_id}-field-response", case_id, "Ion-current response to the field (binned)",
                {"ion_mean": ("Ion current signal, mean (raw voltage)", "V"),
                 "ion_min": ("Ion current signal, min (raw voltage)", "V"),
                 "ion_max": ("Ion current signal, max (raw voltage)", "V"),
                 "n": ("Samples in bin", None)},
                ["ion_mean"], FIELD_NOTE, time_label="Electric-field mesh voltage"), response))
            exports[-1][0]["columns"][0]["unit"] = "kV"
        yield {
            "case_id": case_id,
            "title": f"E-FIELD {test_id}: {fuel}" + (f" {dilution * 100:.0f}%" if dilution is not None else "")
                     + f", {sign} field to {peak:.1f} kV",
            "investigation": INVESTIGATION, "fuel": fuel, "o2_pct": 21.0, "pressure_kpa": kpa, "z_st": None,
            "source_label": f"NASA PSI-22 ACME E-FIELD Flames, test {test_id}: {test_id}_data.csv; "
                            "E-FIELD Flames data interpretation document, Table 1",
            "source_url": None,
            "note": " ".join(part for part in (
                entry.get("highlight"), "Source: " + "; ".join(facts) + ".", events_sentence(events),
                ONE_ATM_NOTE) if part),
            "explanation": (f"Laminar {fuel} diffusion flame in microgravity with an electric field of up to "
                            f"{abs(peak):.1f} kV ({sign} mesh) applied across it"
                            + (f", fuel diluted to {dilution * 100:.0f}% with nitrogen." if dilution is not None else ".")),
            "explanation_origin": EXPLANATION_ORIGIN,
            "_events": [{"label": name, "time_s": round(offset, 3), "dataset_id": f"{case_id}-sensors",
                         "source": f"{test_id}_TP.txt"} for offset, name in events],
        }, exports


def select(data_dir: Path, per_day: int = 3, max_mb: float = 9.0) -> list[str]:
    folder = data_dir / FAMILY / "csv"
    chosen, taken = [], {}
    for path in sorted(folder.glob("*_data.csv")):
        test_id = path.name.removesuffix("_data.csv")
        if test_id[:5] not in DAYS or path.stat().st_size > max_mb * 1e6 or b"\x00" in path.read_bytes():
            continue
        if not (data_dir / FAMILY / "txt" / f"{test_id}_TP.txt").is_file():
            continue
        head = path.read_bytes().split(b"\n", 1)[0]
        if b"EField: 1" in head and taken.get(test_id[:5], 0) < per_day:
            taken[test_id[:5]] = taken.get(test_id[:5], 0) + 1
            chosen.append(test_id)
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["select"])
    parser.add_argument("--data-dir", type=Path, default=prepare.BACKEND.parent / "data_nasa")
    args = parser.parse_args()
    print(json.dumps([{"test_id": t} for t in select(args.data_dir.resolve())]))
