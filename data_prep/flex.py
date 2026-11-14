"""B07 adapter: CFI droplet tests (FLEX-nnnn tabular TXT) as cases and datasets.

A test is linked by its FLEX number: the file name prefix, the `Flex Number`
row of the Master Sheet, and the hand-reviewed per-test summary PDF, whose CIR
run ID must equal the sheet's. Times are the source's own time columns.
"""

from pathlib import Path
import re

from openpyxl import load_workbook
import polars as pl

from data_prep.prepare import Skip, descriptor

FAMILY = "CFI_PSI-39"
SHEET = "CFI_DataSummary_Spreadsheet_Jan2021.xlsx"
INVESTIGATION = "CFI (FLEX)"
ATM_TO_KPA = 101.325
MMHG_TO_KPA = 101.325 / 760
FUELS = {"C12H26": "dodecane"}  # the summary PDFs of the selected tests name it "Dodecane"

# kind -> (source time column, label, {source column: (label, unit)}, default y)
# Units are the ones stated in the source column names.
KINDS = {
    "DropletData": ("DropletTime_sec", "Droplet diameter history", {
        "DropletDiameter_mm": ("Droplet diameter", "mm")}, ["DropletDiameter_mm"]),
    "CoolFlameData": ("CoolFlameTime_s", "Cool-flame diameter", {
        "CoolFlameDiameterFWHM_mm": ("Cool-flame diameter (FWHM)", "mm"),
        "CoolFlameDiameterP2P_mm": ("Cool-flame diameter (peak to peak)", "mm")},
        ["CoolFlameDiameterFWHM_mm", "CoolFlameDiameterP2P_mm"]),
    "RadiometerData": ("RadiometerTime_sec", "Radiometer power", {
        "TotalWideBandPower_W": ("Total wide-band power", "W"),
        "TotalNarrowBandPower_W": ("Total narrow-band power", "W")},
        ["TotalNarrowBandPower_W"]),
}


def read_tsv(path: Path, time: str, columns: dict) -> pl.DataFrame:
    """One FLEX tab-separated series with time_s first; unparsable cells become null."""
    if not path.is_file() or path.stat().st_size == 0:
        raise Skip(f"missing or empty {path.name}")
    content = path.read_bytes()
    # Some archive entries are named .txt but hold PDF/binary data: trust the header only.
    if not content.startswith((time + "\t").encode()) or b"\x00" in content:
        raise Skip(f"{path.name}: not the expected tab-separated table")
    # Legacy files end lines with a lone CR; normalise CR and CRLF to LF.
    raw = pl.read_csv(content.replace(b"\r\n", b"\n").replace(b"\r", b"\n"),
                      separator="\t", infer_schema=False)
    missing = [name for name in (time, *columns) if name not in raw.columns]
    if missing:
        raise Skip(f"{path.name}: missing columns {missing}")
    frame = raw.select(
        pl.col(time).str.strip_chars().cast(pl.Float64, strict=False).alias("time_s"),
        *(pl.col(name).str.strip_chars().cast(pl.Float64, strict=False) for name in columns),
    ).filter(pl.col("time_s").is_finite())
    if frame.height == 0:
        raise Skip(f"{path.name}: no samples")
    return frame


def sheet_rows(path: Path) -> dict[str, dict]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook["Master Sheet"].iter_rows(values_only=True))
    finally:
        workbook.close()
    headers = [str(value).strip() for value in rows[0]]
    return {f"FLEX-{row[0]}": dict(zip(headers, row)) for row in rows[1:] if row[0] is not None}


def flex_cases(data_dir: Path, entries: list[dict]):
    """Yield (case row, [(descriptor, frame), ...]) for the reviewed FLEX tests."""
    if not entries:
        return
    family = data_dir / FAMILY
    sheet_path = family / "xlsx" / SHEET
    if not sheet_path.is_file():
        print(f"SKIPPED all FLEX cases: {sheet_path} not found")
        return
    rows = sheet_rows(sheet_path)
    for entry in entries:
        test_id, pdf = entry["test_id"], entry["summary_pdf"]
        try:
            row = rows.get(test_id)
            if row is None:
                raise Skip("no row in the Master Sheet")
            if str(row["CIR"]).strip().lower() != pdf["cir"].lower():
                raise Skip("CIR run ID differs between the Master Sheet and the summary PDF")
            exports = []
            for kind, (time, label, columns, default_y) in KINDS.items():
                frame = read_tsv(family / "txt" / f"{test_id}_{kind}.txt", time, columns)
                exports.append((descriptor(
                    f"{test_id}-{kind.removesuffix('Data').lower()}", test_id, label, columns,
                    default_y, time_label="Time (source time base)"), frame))
        except Skip as reason:
            print(f"SKIPPED {test_id}: {reason}")
            continue
        sheet_o2 = float(row["Oxygen"])
        # The two sources must agree before the value is used as a filter.
        o2_pct = round(sheet_o2 * 100, 3) if abs(sheet_o2 - pdf["yo2"]) <= 0.005 else None
        atm = re.fullmatch(r"\s*(\d*\.?\d+)\s*atm\s*", str(row["Pressure"]))
        kpa = round(float(atm.group(1)) * ATM_TO_KPA, 3) if atm else None
        fuel_label = str(row["Fuel"]).strip()
        fuel = FUELS.get(fuel_label)
        conditions = ", ".join(part for part in (
            f"{fuel or fuel_label} droplet", None if o2_pct is None else f"{o2_pct:.1f}% O₂",
            None if kpa is None else f"{kpa:.0f} kPa") if part)
        helium = float(row["Helium"] or 0)
        note = [
            entry.get("highlight"),
            f"Master Sheet: fuel {fuel_label}, pressure {str(row['Pressure']).strip()}, "
            f"O₂ {sheet_o2:g}, N₂ {float(row['Nitrogen']):g}, He {helium:g} (fractions); "
            f"CIR run {pdf['cir']}.",
            f"Summary PDF: P = {pdf['p_mmhg']} mmHg ({pdf['p_mmhg'] * MMHG_TO_KPA:.1f} kPa), "
            f"YO2 = {pdf['yo2']}, D0 = {pdf['d0_mm']} mm, kavg = {pdf['kavg_mm2_s']} mm²/s.",
            None if o2_pct is not None else
            "The two sources disagree on oxygen, so the O₂ filter value is left unknown.",
            "The pressure filter uses the Master Sheet's nominal value converted from atm. "
            "Each dataset keeps its own source time column; the source does not document "
            "how the time bases are synchronised. The wide-band radiometer values are large "
            "and negative in the source file and are shown as provided.",
        ]
        yield {
            "case_id": test_id,
            "title": f"CFI {test_id}: {conditions}",
            "investigation": INVESTIGATION,
            "fuel": fuel,
            "o2_pct": o2_pct,
            "pressure_kpa": kpa,
            "source_label": f"NASA PSI-39 CFI, {test_id} (CIR {pdf['cir']}): {SHEET}; "
                            f"{test_id}_Summary.pdf; {test_id}_*Data.txt",
            "source_url": None,
            "note": " ".join(part for part in note if part),
        }, exports
