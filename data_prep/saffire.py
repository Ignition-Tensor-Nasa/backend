"""Adapter: SAFFIRE-III (PSI-100) spacecraft fire experiment as three cases.

- SAF3-FILE1 / SAF3-FILE2: the calibrated 30 Hz sensor workbooks "Sample 1/2 Sensors". The
  source does not explain the numbering, so cases are named after the files and described
  only by what their data show.
- SAF3-TELEMETRY: the 1 Hz downlink (Saffire housekeeping and the Cygnus vehicle's own
  smoke detector, pressure and temperature channels).

Units are the workbook's (second row). Pressure is converted from psia (1 psi = 6.894757 kPa).
"""

from pathlib import Path

from openpyxl import load_workbook
import polars as pl

from data_prep.prepare import EXPLANATION_ORIGIN, PSI_TO_KPA, descriptor

FAMILY = "SAFFIRE-III_PSI-100"
INVESTIGATION = "Saffire-III"
IGNITER_LABEL = "Igniter on"
IGNITER_THRESHOLD_A = 0.3
SAMPLE_TC = [f"Sample Temp {n} (TC{n + 6})" for n in range(1, 7)]
SENSOR_COLUMNS = (
    ["Average Fan Speed", "CO2", "O2", "Anemometer 1", "Anemometer 2", "Ambient Pressure",
     "Radiometer 1", "Radiometer 2", "Radiometer 3", "Radiometer 4", "Humidity",
     "Test Section Inlet Temp"]
    + [f"Plume Temp {n} (TC{n})" for n in range(1, 7)] + SAMPLE_TC
    + ["Igniter Current", "Igniter Voltage"])
RENAMES = {"CO2": "CO₂ concentration", "O2": "O₂ concentration"}
SENSOR_NOTE = (
    "Calibrated 30 Hz table of the source workbook (every row kept, no sampling or interpolation); "
    "units are the workbook's. Ambient pressure is converted from psia to kPa. Radiometers read "
    "about -17 mW/cm² when idle (source calibration). Thermocouple positions are not given in "
    "the files used here. time_s is the workbook's elapsed time."
)
TELEMETRY_NOTE = (
    "1 Hz downlinked telemetry of the Saffire experiment on the file's own clock (seconds from the "
    "start of the downlink); rows with a broken timestamp are dropped. Radiometer and anemometer "
    "units are not stated in this file. When the experiment is off the file holds a sentinel "
    "(about -4000) in these channels."
)
CYGNUS_NOTE = (
    "Cygnus vehicle channels from the same downlink. Each row carries one of these channels at a "
    "time; a row where the channel was not reported holds 0 in the source and is null here, so "
    "genuine zero readings cannot be told apart. Smoke-detector values are the detector's output "
    "in volts, not a smoke concentration. Pressure is converted from psi to kPa."
)


def load_workbook_table(path: Path) -> tuple[pl.DataFrame, dict[str, str | None]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook["Calibrated Data"].iter_rows(values_only=True))
    finally:
        workbook.close()
    names = [str(value).strip() for value in rows[0][:42] if value is not None]
    units = {name: (str(rows[1][i]).strip() if rows[1][i] is not None else None)
             for i, name in enumerate(names)}
    table = pl.DataFrame([row[:len(names)] for row in rows[2:] if row[0] is not None],
                         schema=names, orient="row", infer_schema_length=None)
    table = table.drop("Timestamp (ms)").cast(pl.Float64, strict=False)
    return table.rename({"Elapsed Time (s)": "t"}).sort("t"), units


UNIT_MAP = {"mw/cm^2": "mW/cm²", "Degrees C": "°C", "cm/sec": "cm/s", "PSIA": "kPa", "Volts": "V",
            "Amps": "A", "RPM": "rpm"}


def sensor_dataset(case_id: str, table: pl.DataFrame, units: dict) -> tuple[dict, pl.DataFrame]:
    frame = table.select(pl.col("t").alias("time_s"), *SENSOR_COLUMNS)
    frame = frame.with_columns((pl.col("Ambient Pressure") * PSI_TO_KPA).alias("Ambient Pressure"))
    columns = {name: (RENAMES.get(name, name), UNIT_MAP.get(units.get(name), units.get(name)))
               for name in SENSOR_COLUMNS}
    return descriptor(f"{case_id}-sensors", case_id, "Sensor recording (calibrated, 30 Hz)", columns,
                      ["Sample Temp 3 (TC9)", "Radiometer 1", "O2"], SENSOR_NOTE), frame


def burn_facts(table: pl.DataFrame) -> dict:
    """Numbers computed from one sensor table (igniter window, temperature and flux rise)."""
    ignited = table.filter(pl.col("Igniter Current") > IGNITER_THRESHOLD_A)["t"]
    first, last = (ignited.min(), ignited.max()) if len(ignited) else (None, None)
    before = table.filter(pl.col("t") < (first if first is not None else table["t"].max()))
    peak_tc = max(SAMPLE_TC, key=lambda column: table[column].max())
    return {
        "igniter": (first, last),
        "peak_temp": table[peak_tc].max(), "peak_tc": peak_tc.split("(")[1].rstrip(")"),
        "temp_rise": table[peak_tc].max() - before[peak_tc].mean() if before.height else None,
        "peak_flux": max(table[f"Radiometer {n}"].max() for n in range(1, 5)),
        "o2_before": before["O2"].median(), "o2_min": table["O2"].min(),
        "p_kpa": before["Ambient Pressure"].median() * PSI_TO_KPA, "duration": table["t"].max(),
        "o2_pct": before["O2"].median(), "flow": before["Anemometer 1"].median(),
    }


def file_case(number: int, table: pl.DataFrame) -> dict:
    facts = burn_facts(table)
    case_id = f"SAF3-FILE{number}"
    burned = facts["temp_rise"] is not None and facts["temp_rise"] > 100
    outcome = (f"The sample thermocouples rise by about {facts['temp_rise']:.0f} K to {facts['peak_temp']:.0f} °C "
               f"({facts['peak_tc']}) and the radiometers reach {facts['peak_flux']:.0f} mW/cm²: a flame spreads "
               "over the sample." if burned else
               f"The igniter fires but the sample thermocouples change by only about "
               f"{facts['temp_rise']:.0f} K (peak {facts['peak_temp']:.0f} °C) and the radiometers stay near "
               f"their idle reading: no flame spread is visible in these data.")
    first, last = facts["igniter"]
    note = (f"File 'Saffire III Sample {number} Sensors.xlsx', {facts['duration']:.0f} s recorded. "
            f"Igniter current is above {IGNITER_THRESHOLD_A} A from {first:.0f} s to {last:.0f} s. {outcome} "
            "The source does not explain what 'Sample 1' and 'Sample 2' denote; the mission documents "
            "describe a single large SIBAL cotton-fiberglass sample in a 30 cm/s air flow (the anemometer "
            f"reads about {facts['flow']:.0f} cm/s). O₂ and "
            "pressure filter values are the experiment's own sensor readings before ignition.")
    return {
        "case_id": case_id, "title": f"Saffire-III, sensor file {number}: SIBAL fabric, "
                                     f"{facts['o2_pct']:.1f}% O₂, {facts['p_kpa']:.0f} kPa",
        "investigation": INVESTIGATION, "fuel": "SIBAL fabric",
        "o2_pct": round(facts["o2_pct"], 3), "pressure_kpa": round(facts["p_kpa"], 3), "z_st": None,
        "source_label": f"NASA PSI-100 Saffire-III: Saffire III Sample {number} Sensors.xlsx (Calibrated Data)",
        "source_url": None, "note": note,
        "explanation": (f"Spacecraft fire experiment inside Cygnus: a SIBAL fabric sample in forced air flow "
                        f"(about {facts['flow']:.0f} cm/s measured) and {facts['o2_pct']:.1f}% oxygen. {outcome}"),
        "explanation_origin": EXPLANATION_ORIGIN,
        "_events": [{"label": IGNITER_LABEL, "time_s": round(first, 3), "dataset_id": f"{case_id}-sensors",
                     "source": f"first sample with igniter current above {IGNITER_THRESHOLD_A} A"},
                    {"label": "Igniter off", "time_s": round(last, 3), "dataset_id": f"{case_id}-sensors",
                     "source": f"last sample with igniter current above {IGNITER_THRESHOLD_A} A"}],
    }


def telemetry_datasets(data_dir: Path) -> tuple[dict, list]:
    path = data_dir / FAMILY / "csv" / "Saffire_Data_Table2017.161.00.42.03-EU.csv"
    raw = pl.read_csv(path, infer_schema_length=20000)
    raw = raw.rename({name: name.replace("P@", "").replace("sfsHk.", "") for name in raw.columns})
    raw = raw.filter(pl.col("RelativeSeconds") >= 0).with_columns(
        (pl.col("RelativeSeconds") / 1000).alias("time_s")).sort("time_s")
    case_id = "SAF3-TELEMETRY"
    saffire = {"Radiometer1value": "Radiometer 1", "Radiometer2value": "Radiometer 2",
               "Radiometer3value": "Radiometer 3", "Radiometer4value": "Radiometer 4",
               "Anemometer1": "Anemometer 1", "Anemometer2": "Anemometer 2", "FanTachometer": "Fan tachometer",
               "Sample1IgniterCurrent": "Sample 1 igniter current", "Sample2IgniterCurrent": "Sample 2 igniter current",
               "SmokeWireCurrent": "Smoke wire current"}
    frame = raw.select("time_s", *saffire)
    sets = [(descriptor(f"{case_id}-saffire", case_id, "Saffire telemetry (1 Hz)",
                        {key: (label, None) for key, label in saffire.items()},
                        ["Radiometer1value", "Anemometer1"], TELEMETRY_NOTE), frame)]
    cygnus = {"CDH_F2PCM_SMK_DET_SCAT_SENS_P": ("Smoke detector FIZ 2, scatterometer", "V"),
              "CDH_F2PCM_SMK_DET_OBS_SENS_P": ("Smoke detector FIZ 2, obscuration", "V"),
              "F1PCM_AIR_PRES_EPS1_SIG_POS": ("Cabin pressure EPS 1", "kPa"),
              "F2PCM_AIR_PRES_EPS2_SIG_POS": ("Cabin pressure EPS 2", "kPa"),
              "F3PCM_AIR_PRES_EPS3_SIG_POS": ("Cabin pressure EPS 3", "kPa"),
              "PCM_ETS_TEMP_SIG": ("Cabin air temperature", "°C"),
              "CDH_F2PCM_ECLS_FAN_PRESSURE_P": ("Cabin fan pressure difference FIZ 2", "Pa")}
    cframe = raw.select("time_s", *[
        pl.when(pl.col(key) != 0).then(pl.col(key)).alias(key) for key in cygnus])
    cframe = cframe.with_columns([(pl.col(key) * PSI_TO_KPA).alias(key) for key in cygnus if key.startswith("F")]
                                 ).filter(pl.any_horizontal(pl.col(*cygnus).is_not_null()))
    sets.append((descriptor(f"{case_id}-cygnus", case_id, "Cygnus vehicle channels (multiplexed)", cygnus,
                            ["CDH_F2PCM_SMK_DET_SCAT_SENS_P"], CYGNUS_NOTE), cframe))
    wire = raw.filter(pl.col("Radiometer1value") > -1000).filter(pl.col("SmokeWireCurrent") > 0.2)["time_s"].to_list()
    stats = {"wire": wire, "duration": raw["time_s"].max(), "o2": None}
    return stats, sets


def saffire_cases(data_dir: Path, entries: list[dict]):
    if not entries:
        return
    folder = data_dir / FAMILY / "xlsx"
    if not folder.is_dir():
        print(f"SKIPPED Saffire-III: {folder} not found")
        return
    for entry in (e for e in entries if "file" in e):
        number = entry["file"]
        table, units = load_workbook_table(folder / f"Saffire III Sample {number} Sensors.xlsx")
        case = file_case(number, table)
        case["note"] = (entry.get("highlight", "") + " " + case["note"]).strip()
        yield case, [sensor_dataset(case["case_id"], table, units)]
    if any(entry.get("telemetry") for entry in entries):
        stats, sets = telemetry_datasets(data_dir)
        wire = ", ".join(f"{t:.0f}" for t in stats["wire"][:6])
        case_id = "SAF3-TELEMETRY"
        yield {
            "case_id": case_id, "title": "Saffire-III, 1 Hz downlink: whole mission with Cygnus channels",
            "investigation": INVESTIGATION, "fuel": "SIBAL fabric", "o2_pct": None, "pressure_kpa": None,
            "z_st": None,
            "source_label": "NASA PSI-100 Saffire-III: Saffire_Data_Table2017.161.00.42.03-EU.csv",
            "source_url": None,
            "note": (f"Telemetry of the whole {stats['duration'] / 3600:.1f} h downlink: the Saffire experiment's "
                     f"housekeeping (active in the last ~25 min) and the Cygnus vehicle's own smoke detector, "
                     f"cabin pressure and temperature. Smoke-wire pulses (current above 0.2 A) begin at about "
                     f"{wire} s of this clock. Atmosphere filter values are left unknown here: the downlink "
                     "has no oxygen channel (see the sensor files for O₂)."),
            "explanation": "Everything the Saffire-III downlink recorded, including what the spacecraft's "
                           "own smoke detector reported during the fire.",
            "explanation_origin": EXPLANATION_ORIGIN, "_events": [],
        }, sets
