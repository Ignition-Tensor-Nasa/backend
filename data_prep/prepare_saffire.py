"""Curated SAFFIRE-III (NASA PSI-100) data preparation matching polars_analysis/saffire3.py.

Run standalone from backend/:
    python3 -m data_prep.prepare_saffire --data-dir ../data --out-dir data/curated

Reads calibrated 30 Hz sensor workbooks (Sample 1 & Sample 2) and 1 Hz Cygnus
downlink telemetry CSV, writes Parquet files into out_dir/parquet/, and builds
out_dir/saffire3.json for the bespoke SAFFIRE-III frontend view and AI summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from openpyxl import load_workbook
import polars as pl

from polars_analysis.loaders import downsample

BACKEND = Path(__file__).resolve().parents[1]
FAMILY = "SAFFIRE-III_PSI-100"
TELEMETRY = "Saffire_Data_Table2017.161.00.42.03-EU.csv"
PSI_TO_KPA = 6.894757293168
POINT_LIMIT = 2500  # per curve; min/max envelope keeps peaks, see downsample()
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#22c55e"]

SAMPLE_TC = [f"Sample Temp {n} (TC{n + 6})" for n in range(1, 7)]
PLUME_TC = [f"Plume Temp {n} (TC{n})" for n in range(1, 7)]
RADIOMETERS = [f"Radiometer {n}" for n in range(1, 5)]

PANELS = [
    ("sample_temp", "Sample surface temperature", "°C", {c: f"TC{i + 7}" for i, c in enumerate(SAMPLE_TC)}, 1.0),
    ("radiant_heat_flux", "Radiant heat flux", "mW/cm²", {c: f"Radiometer {i + 1}" for i, c in enumerate(RADIOMETERS)}, 1.0),
    ("plume_temp", "Plume (gas) temperature", "°C", {c: f"TC{i + 1}" for i, c in enumerate(PLUME_TC)}, 1.0),
    ("oxygen", "Oxygen", "%", {"O2": "O₂"}, 1.0),
    ("carbon_dioxide", "Carbon dioxide", "%", {"CO2": "CO₂"}, 1.0),
    ("ambient_pressure", "Ambient pressure", "kPa", {"Ambient Pressure": "Pressure"}, PSI_TO_KPA),
    ("air_flow", "Air flow speed", "cm/s", {"Anemometer 1": "Anemometer 1", "Anemometer 2": "Anemometer 2"}, 1.0),
]


def _clean_float(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    f = float(value)
    if not math.isfinite(f):
        return None
    return round(f, digits)


def load_sample(root: Path, number: int) -> pl.DataFrame:
    """Calibrated 30 Hz sensor table of one sample; columns keep the source names."""
    path = root / FAMILY / "xlsx" / f"Saffire III Sample {number} Sensors.xlsx"
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook["Calibrated Data"].iter_rows(values_only=True))
    finally:
        workbook.close()
    names = [str(value).strip() for value in rows[0][:42] if value is not None]
    table = pl.DataFrame(
        [row[:len(names)] for row in rows[2:] if row[0] is not None],
        schema=names,
        orient="row",
        infer_schema_length=None,
    )
    return (
        table.drop("Timestamp (ms)")
        .cast(pl.Float64, strict=False)
        .rename({"Elapsed Time (s)": "t"})
        .sort("t")
    )


def load_telemetry(root: Path) -> pl.DataFrame:
    """1 Hz downlinked telemetry with t in seconds; rows with a broken timestamp are dropped."""
    raw = pl.read_csv(root / FAMILY / "csv" / TELEMETRY, infer_schema_length=20000)
    return (
        raw.rename({name: name.replace("P@", "").replace("sfsHk.", "") for name in raw.columns})
        .filter(pl.col("RelativeSeconds") >= 0)
        .with_columns((pl.col("RelativeSeconds") / 1000).alias("t"))
        .sort("t")
    )


def reported(telemetry: pl.DataFrame, column: str) -> pl.DataFrame:
    """Cygnus channels are multiplexed: a row carries a channel only when it is non-zero."""
    return telemetry.filter(pl.col(column) != 0).select("t", pl.col(column).alias("value"))


def on_window(table: pl.DataFrame, column: str, threshold: float) -> tuple[float, float] | None:
    times = table.filter(pl.col(column) > threshold)["t"]
    if not len(times):
        return None
    t0, t1 = _clean_float(times.min(), 3), _clean_float(times.max(), 3)
    return None if t0 is None or t1 is None else (t0, t1)


def thin_series(table: pl.DataFrame, column: str, factor: float = 1.0, limit: int = POINT_LIMIT) -> list[list[float | None]]:
    """Return [[t, value], ...] downsampled via min/max envelope preserving peaks and gaps."""
    prepared = table.select(
        pl.col("t"),
        pl.when((pl.col(column) * factor).is_finite())
        .then(pl.col(column) * factor)
        .otherwise(None)
        .alias("value"),
    )
    thinned = downsample(prepared, "t", limit)
    out: list[list[float | None]] = []
    for t_val, y_val in thinned.iter_rows():
        t_clean = _clean_float(t_val, 3)
        if t_clean is None:
            continue
        out.append([t_clean, _clean_float(y_val, 4)])
    return out


def compute_spacecraft_events(telemetry: pl.DataFrame) -> dict:
    scatter = reported(telemetry, "CDH_F2PCM_SMK_DET_SCAT_SENS_P")
    saffire = telemetry.filter(pl.col("Radiometer1value") > -1000)
    wire = saffire.filter(pl.col("SmokeWireCurrent") > 0.2)["t"].to_list()
    baseline = scatter.filter(pl.col("t") < wire[0])["value"].max()
    first_rise = scatter.filter((pl.col("t") >= wire[0]) & (pl.col("value") > 2 * baseline))["t"].min()
    peak_idx = scatter["value"].arg_max()
    scatter_peak_t = scatter.row(peak_idx)[0]
    scatter_peak = scatter["value"].max()
    return {
        "igniter1": list(on_window(saffire, "Sample1IgniterCurrent", 0.3) or ()),
        "igniter2": list(on_window(saffire, "Sample2IgniterCurrent", 0.3) or ()),
        "smoke_wire": [round(float(x), 3) for x in wire],
        "window": [_clean_float(saffire["t"].min() - 60, 3), _clean_float(saffire["t"].max(), 3)],
        "scatter_baseline": _clean_float(baseline, 4),
        "scatter_peak": _clean_float(scatter_peak, 4),
        "scatter_peak_t": _clean_float(scatter_peak_t, 3),
        "first_rise": _clean_float(first_rise, 3),
        "smoke_lag_s": _clean_float(float(first_rise) - float(wire[0]), 2),
    }


def compute_sample_facts(sample: pl.DataFrame, telemetry: pl.DataFrame, events: dict) -> tuple[list[dict], dict]:
    igniter = on_window(sample, "Igniter Current", 0.3)
    before = sample.filter(pl.col("t") < igniter[0]) if igniter else sample.head(100)
    hottest = max(SAMPLE_TC, key=lambda column: sample[column].max())
    plume_hottest = max(PLUME_TC, key=lambda column: sample[column].max())
    radiometer = max(RADIOMETERS, key=lambda column: sample[column].max())
    peak_row = sample.row(sample[hottest].arg_max(), named=True)

    tc_peak = float(sample[hottest].max())
    plume_peak = float(sample[plume_hottest].max())
    rad_peak = float(sample[radiometer].max())
    rad_idle = float(before[radiometer].mean())
    o2_before = float(before["O2"].mean())
    o2_min = float(sample["O2"].min())
    co2_before = float(before["CO2"].mean())
    co2_max = float(sample["CO2"].max())
    p_before_kpa = float(before["Ambient Pressure"].mean()) * PSI_TO_KPA
    p_rise_kpa = (float(sample["Ambient Pressure"].max()) - float(before["Ambient Pressure"].mean())) * PSI_TO_KPA
    flow_mean = float(before["Anemometer 1"].mean())
    lag_s = float(events["smoke_lag_s"])
    peak_after_ig = (float(peak_row["t"]) - igniter[0]) if igniter else float(peak_row["t"])

    tiles = [
        {
            "value": f"{tc_peak:.0f} °C" if tc_peak >= 100 else f"{tc_peak:.1f} °C",
            "label": "peak sample temperature",
            "detail": (
                f"{hottest.split('(')[1].rstrip(')')}, {peak_after_ig:.0f} s after ignition"
                if tc_peak >= 100
                else f"{hottest.split('(')[1].rstrip(')')}; plume {plume_hottest.split('(')[1].rstrip(')')} reached {plume_peak:.1f} °C during igniter"
            ),
        },
        {
            "value": f"{rad_peak:.0f} mW/cm²" if rad_peak >= 10 else f"{rad_peak:.1f} mW/cm²",
            "label": "peak radiant heat flux",
            "detail": f"{radiometer}; idle reading is about {rad_idle:.0f}",
        },
        {
            "value": f"{o2_before:.2f} → {o2_min:.2f} %",
            "label": "oxygen, before → lowest",
            "detail": f"CO₂ {co2_before:.2f} → {co2_max:.2f} %",
        },
        {
            "value": f"+{p_rise_kpa:.2f} kPa",
            "label": "ambient pressure rise",
            "detail": f"from {p_before_kpa:.1f} kPa",
        },
        {
            "value": f"{flow_mean:.0f} cm/s",
            "label": "air flow through the test section",
            "detail": "Anemometer 1, average before ignition",
        },
        {
            "value": f"{lag_s:.0f} s",
            "label": "smoke wire → Cygnus detector",
            "detail": f"scatter {events['scatter_baseline']:.3f} V → peak {events['scatter_peak']:.2f} V",
        },
    ]

    summary = {
        "duration_s": _clean_float(sample["t"].max(), 3),
        "igniter_window": list(igniter) if igniter else None,
        "peak_sample_tc": hottest,
        "peak_sample_temp_c": _clean_float(tc_peak, 2),
        "peak_sample_time_s": _clean_float(peak_row["t"], 3),
        "peak_after_ignition_s": _clean_float(peak_after_ig, 2),
        "peak_plume_tc": plume_hottest,
        "peak_plume_temp_c": _clean_float(plume_peak, 2),
        "peak_radiometer": radiometer,
        "peak_heat_flux_mw_cm2": _clean_float(rad_peak, 2),
        "idle_heat_flux_mw_cm2": _clean_float(rad_idle, 2),
        "o2_before_pct": _clean_float(o2_before, 3),
        "o2_min_pct": _clean_float(o2_min, 3),
        "co2_before_pct": _clean_float(co2_before, 3),
        "co2_max_pct": _clean_float(co2_max, 3),
        "pressure_before_kpa": _clean_float(p_before_kpa, 2),
        "pressure_rise_kpa": _clean_float(p_rise_kpa, 3),
        "air_flow_cm_s": _clean_float(flow_mean, 2),
    }
    return tiles, summary


def build_sample_payload(sample: pl.DataFrame, number: int, telemetry: pl.DataFrame, events: dict) -> dict:
    tiles, stats = compute_sample_facts(sample, telemetry, events)
    max_t = _clean_float(sample["t"].max(), 3) or 0.0
    igniter = stats["igniter_window"]
    hottest = stats["peak_sample_tc"]
    peak_t = float(sample["t"][sample[hottest].arg_max()])

    presets: dict[str, list[float]] = {"Whole run": [0.0, max_t]}
    if igniter:
        presets["Ignition"] = [round(max(0.0, igniter[0] - 15.0), 2), round(min(max_t, igniter[1] + 60.0), 2)]
    if number == 1:
        presets["Peak burn"] = [round(max(0.0, peak_t - 60.0), 2), round(min(max_t, peak_t + 150.0), 2)]

    burn_panels = []
    for panel_id, title, unit, columns, factor in PANELS:
        series_list = []
        for idx, (col, label) in enumerate(columns.items()):
            series_list.append({
                "name": label,
                "color": COLORS[idx % len(COLORS)],
                "points": thin_series(sample, col, factor, POINT_LIMIT),
            })
        burn_panels.append({
            "id": panel_id,
            "title": title,
            "unit": unit,
            "series": series_list,
        })

    # Section 2: Arrival times per thermocouple
    labels = [f"TC{n + 7}" for n in range(6)]
    thresholds = [100, 300, 600] if number == 1 else [15, 17, 100]
    default_threshold = 300 if number == 1 else 15
    arrivals_by_threshold = {}
    for threshold in thresholds:
        vals = []
        for col in SAMPLE_TC:
            hit = sample.filter(pl.col(col) >= threshold)["t"].min()
            vals.append(_clean_float(hit, 2))
        arrivals_by_threshold[str(threshold)] = vals
    peaks_c = [_clean_float(sample[col].max(), 2) or 0.0 for col in SAMPLE_TC]

    # Section 3: 3-second binned playback frames across TC7..TC12
    step = 3.0
    binned = (
        sample.with_columns((pl.col("t") // step * step).alias("bin"))
        .group_by("bin")
        .agg([pl.col(col).max() for col in SAMPLE_TC])
        .sort("bin")
    )
    top_c = max(float(binned[col].max()) for col in SAMPLE_TC) * 1.08
    frames = []
    for row in binned.iter_rows(named=True):
        frames.append({
            "t_s": round(float(row["bin"]), 1),
            "values_c": [_clean_float(row[col], 2) or 0.0 for col in SAMPLE_TC],
        })

    arrival_note = (
        "Thermocouple positions are not given in the files used here, so this shows order and timing, not a spread rate."
        if number == 1
        else "Thermocouple positions are not given in the files used here. Sample 2 thermocouples remained ≤ 20.0 °C (≥ 100 °C was never reached); thresholds include 15 °C and 17 °C alongside 100 °C for direct comparison."
    )

    return {
        "sample_number": number,
        "case_id": f"SAFFIRE-III-S{number}",
        "tab_label": f"Sample 1 · the large burn ({max_t:.0f} s)" if number == 1 else f"Sample 2 · {max_t:.0f} s recording",
        "duration_s": max_t,
        "point_limit": POINT_LIMIT,
        "igniter_window": igniter,
        "presets": presets,
        "tiles": tiles,
        "stats": stats,
        "burn_panels": burn_panels,
        "arrival": {
            "thermocouples": labels,
            "thresholds_c": thresholds,
            "default_threshold_c": default_threshold,
            "arrivals_by_threshold": arrivals_by_threshold,
            "peaks_c": peaks_c,
            "max_t": max_t,
            "note": arrival_note,
        },
        "playback": {
            "thermocouples": labels,
            "step_s": step,
            "y_max_c": round(top_c, 2),
            "frames": frames,
        },
    }


def build_spacecraft_payload(telemetry: pl.DataFrame, events: dict) -> dict:
    saffire = telemetry.filter(pl.col("Radiometer1value") > -1000)

    def _to_points(df: pl.DataFrame) -> list[list[float | None]]:
        out = []
        for t_val, y_val in df.iter_rows():
            tc = _clean_float(t_val, 3)
            yc = _clean_float(y_val, 4)
            if tc is not None:
                out.append([tc, yc])
        return out

    raw_panels = [
        (
            "smoke_scatter",
            "Cygnus smoke detector FIZ 2, scatterometer",
            "V",
            [("Scatter", reported(telemetry, "CDH_F2PCM_SMK_DET_SCAT_SENS_P"))],
        ),
        (
            "saffire_radiometers",
            "Saffire radiometers, 1 Hz downlink",
            "mW/cm²",
            [(f"Radiometer {n}", saffire.select("t", pl.col(f"Radiometer{n}value").alias("value"))) for n in range(1, 5)],
        ),
        (
            "cabin_pressure",
            "Cygnus cabin pressure",
            "kPa",
            [
                (
                    f"EPS {n}",
                    reported(telemetry, f"F{n}PCM_AIR_PRES_EPS{n}_SIG_POS").with_columns(
                        pl.col("value") * PSI_TO_KPA
                    ),
                )
                for n in (1, 2, 3)
            ],
        ),
        (
            "cabin_temp",
            "Cygnus cabin air temperature",
            "°C",
            [("Air temperature", reported(telemetry, "PCM_ETS_TEMP_SIG"))],
        ),
        (
            "saffire_airflow",
            "Saffire air flow",
            "cm/s",
            [("Anemometer 1", saffire.select("t", pl.col("Anemometer1").alias("value")))],
        ),
    ]

    panels = []
    for panel_id, title, unit, series_items in raw_panels:
        panels.append({
            "id": panel_id,
            "title": title,
            "unit": unit,
            "series": [
                {
                    "name": label,
                    "color": COLORS[idx % len(COLORS)],
                    "points": _to_points(df),
                }
                for idx, (label, df) in enumerate(series_items)
            ],
        })

    t_max = _clean_float(telemetry["t"].max(), 3) or 6900.0
    sw0 = events["smoke_wire"][0]
    presets = {
        "Experiment": events["window"],
        "First smoke": [round(sw0 - 30.0, 2), round(sw0 + 150.0, 2)],
        "Whole downlink": [0.0, t_max],
    }

    return {
        "events": events,
        "presets": presets,
        "t_max": t_max,
        "panels": panels,
    }


def prepare_saffire(data_dir: Path, out_dir: Path) -> dict | None:
    """Build Parquet exports and saffire3.json if SAFFIRE-III_PSI-100 exists in data_dir."""
    saffire_root = data_dir / FAMILY
    if not saffire_root.is_dir():
        return None

    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    sample1 = load_sample(data_dir, 1)
    sample2 = load_sample(data_dir, 2)
    telemetry = load_telemetry(data_dir)

    sample1.write_parquet(parquet_dir / "SAFFIRE-III-sample1-sensors.parquet")
    sample2.write_parquet(parquet_dir / "SAFFIRE-III-sample2-sensors.parquet")
    telemetry.write_parquet(parquet_dir / "SAFFIRE-III-cygnus-telemetry.parquet")

    events = compute_spacecraft_events(telemetry)
    s1_payload = build_sample_payload(sample1, 1, telemetry, events)
    s2_payload = build_sample_payload(sample2, 2, telemetry, events)
    spacecraft_payload = build_spacecraft_payload(telemetry, events)

    s1_stats = s1_payload["stats"]
    s2_stats = s2_payload["stats"]

    payload = {
        "bespoke_notice": (
            "Dedykowana analiza SAFFIRE-III (NASA PSI-100): ten 4-sekcyjny układ wykresów "
            "(czujniki 30 Hz próbek 1 i 2, kolejność dotarcia płomienia TC7–TC12, animowana "
            "oś nagrzewania oraz 1 Hz telemetria statku Cygnus OA-7 + podsumowanie AI) został "
            "przygotowany wyłącznie dla tych 2 specjalnych case'ów."
        ),
        "items": [
            {
                "case_id": "SAFFIRE-III-S1",
                "sample_number": 1,
                "title": "Saffire-III Sample 1 · Large Spacecraft Fire (690 s, 30 Hz + 1 Hz Cygnus)",
                "subtitle": (
                    f"Główny pożar w module Cygnus (0.4 × 0.94 m): szczyt {s1_stats['peak_sample_temp_c']:.0f} °C "
                    f"na TC11 ({s1_stats['peak_after_ignition_s']:.0f} s po zapłonie), radiometr {s1_stats['peak_heat_flux_mw_cm2']:.0f} mW/cm² "
                    f"oraz odpowiedź detektora dymu FIZ2 po {events['smoke_lag_s']:.0f} s."
                ),
                "investigation": "Saffire-III (PSI-100)",
                "capsule": "Cygnus OA-7",
                "fuel": "Saffire-III Sample 1 (0.4 × 0.94 m)",
                "o2_pct": round(float(s1_stats["o2_before_pct"]), 2),
                "pressure_kpa": round(float(s1_stats["pressure_before_kpa"]), 1),
                "recording_duration_s": round(float(s1_stats["duration_s"]), 1),
                "peak_temp_c": round(float(s1_stats["peak_sample_temp_c"]), 1),
                "peak_heat_flux_mw_cm2": round(float(s1_stats["peak_heat_flux_mw_cm2"]), 1),
                "highlight_badge": f"BESPOKE · {s1_stats['peak_sample_temp_c']:.0f} °C Peak",
            },
            {
                "case_id": "SAFFIRE-III-S2",
                "sample_number": 2,
                "title": "Saffire-III Sample 2 · Companion 30 Hz Sensor Record & Cygnus Downlink (160 s)",
                "subtitle": (
                    f"Drugi zestaw 30 Hz (160 s): impuls zapalnika 60.0–68.0 s, szczyt w smudze "
                    f"{s2_stats['peak_plume_temp_c']:.1f} °C (TC3), powierzchnia próbki ≤ {s2_stats['peak_sample_temp_c']:.1f} °C "
                    f"oraz wspólna telemetria 1 Hz statku Cygnus OA-7."
                ),
                "investigation": "Saffire-III (PSI-100)",
                "capsule": "Cygnus OA-7",
                "fuel": "Saffire-III Sample 2",
                "o2_pct": round(float(s2_stats["o2_before_pct"]), 2),
                "pressure_kpa": round(float(s2_stats["pressure_before_kpa"]), 1),
                "recording_duration_s": round(float(s2_stats["duration_s"]), 1),
                "peak_temp_c": round(float(s2_stats["peak_sample_temp_c"]), 1),
                "peak_heat_flux_mw_cm2": round(float(s2_stats["peak_heat_flux_mw_cm2"]), 1),
                "highlight_badge": "BESPOKE · 160 s Companion",
            },
        ],
        "samples": {
            "1": s1_payload,
            "2": s2_payload,
        },
        "spacecraft": spacecraft_payload,
        "methodology_notes": [
            "Curves are drawn from a min/max envelope of the 30 Hz record (at most 2500 points per curve), keeping peaks and missing-data boundaries without smoothing or interpolation.",
            "Radiometers read about −17 to −18 mW/cm² when idle; values are shown directly as calibrated by NASA PSI-100.",
            "Cygnus channels arrive roughly every 6 s, each on its own row; rows where a channel is absent are skipped, never filled in.",
            "Cygnus FIZ 2 smoke-detector readings are detector output in volts (V) as labelled by the source, not a calibrated smoke concentration.",
            "The 1 Hz telemetry CSV uses its own clock (seconds from downlink start). The Saffire 30 Hz and Cygnus 1 Hz clocks differ by ~1 day in source timestamps, so sections 1–3 and section 4 keep separate time axes.",
        ],
        "sources": [
            "Saffire III Sample 1 Sensors.xlsx and Sample 2 Sensors.xlsx, sheet “Calibrated Data” (30 Hz).",
            f"{TELEMETRY} (1 Hz telemetry); channel units from “Saffire-III 1 Hz Telemetry Data.xlsx”.",
            "Sample dimensions and mission sequence: SAFFIRE-III Operations.pdf. Pressure converted from psi (1 psi = 6.894757293168 kPa).",
            "NASA PSI-100 Investigation Repository: https://psi.nasa.gov/physci/repo/data/investigations/PSI-100",
        ],
    }

    (out_dir / "saffire3.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        "utf-8",
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=BACKEND.parent / "data")
    parser.add_argument("--out-dir", type=Path, default=BACKEND / "data" / "curated")
    args = parser.parse_args()
    result = prepare_saffire(args.data_dir.resolve(), args.out_dir.resolve())
    if result is None:
        raise SystemExit(f"SAFFIRE-III data not found in {args.data_dir / FAMILY}")
    print(f"Prepared {len(result['items'])} SAFFIRE-III special cases -> {args.out_dir / 'saffire3.json'}")
