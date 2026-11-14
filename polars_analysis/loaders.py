"""Read original PSI files; retain source names and flag questionable values."""

from pathlib import Path
import re

from openpyxl import load_workbook
import polars as pl


def numeric(column: str) -> pl.Expr:
    return pl.col(column).str.strip_chars().cast(pl.Float64, strict=False)


def load_atmosphere(root: Path) -> tuple[pl.DataFrame, dict]:
    frames = []
    raw_rows = 0
    for flame in ("Inverse", "Normal"):
        path = root / "ACME-CFI-G_PSI-159" / "csv" / (
            f"PSI-159_Experimental table_CFI-G_{flame} Flames.csv"
        )
        raw = pl.read_csv(path, infer_schema=False)
        raw_rows += raw.height
        raw = raw.rename({name: name.strip() for name in raw.columns})
        oxygen = "XO2" if flame == "Inverse" else "XO2 before test"
        burn = "HC burnrate (mg/s)" if flame == "Inverse" else "mHC (mg/s)"
        frames.append(raw.select(
            pl.lit(flame).alias("flame_type"),
            pl.col("Test").str.replace_all(r"\s+", "").alias("test_id"),
            pl.col("Fuel").str.strip_chars().alias("fuel"),
            (numeric(oxygen) * 100).alias("o2_pct"),
            numeric("approx p (bar)").alias("pressure_bar"),
            numeric(burn).alias("burn_rate_mg_s"),
            numeric("Tad (K)").alias("tad_k"),
            numeric("burn time (sec)").alias("burn_time_s"),
            pl.col("Hot ignition").str.strip_chars().str.to_lowercase().alias("ignition"),
            pl.col("Comments").alias("comments"),
            pl.lit(path.relative_to(root).as_posix()).alias("source_file"),
        ))
    data = pl.concat(frames).filter(
        pl.col("test_id").is_not_null() & (pl.col("test_id") != "")
    )
    # Keep failed ignitions in the normalized export, but not in burning heatmaps.
    data = data.with_columns(
        ((pl.col("ignition") == "yes") & (pl.col("burn_time_s") > 0))
        .fill_null(False).alias("sustained_ignition"),
        (pl.col("o2_pct").is_between(0, 100)
         & (pl.col("pressure_bar") > 0)
         & pl.col("pressure_bar").is_finite()
         & pl.col("burn_rate_mg_s").is_finite()
         & (pl.col("burn_rate_mg_s") >= 0))
        .fill_null(False).alias("valid_conditions"),
    )
    return data, {
        "raw_rows": raw_rows, "test_rows": data.height,
        "rows_without_test_id": raw_rows - data.height,
        "invalid_conditions": data.filter(~pl.col("valid_conditions")).height,
        "sustained_ignitions": data.filter(pl.col("sustained_ignition")).height,
    }


def atmosphere_grid(data: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Bin O2 to 5 percentage points and pressure to 0.1 bar; never fill gaps."""
    if metric not in {"burn_rate_mg_s", "tad_k"}:
        raise ValueError(f"Unsupported heatmap metric: {metric}")
    return (
        data.filter(pl.col("valid_conditions") & pl.col("sustained_ignition")
                    & pl.col(metric).is_finite())
        .with_columns(
            ((pl.col("o2_pct") / 5).floor() * 5).alias("o2_bin_lower_pct"),
            ((pl.col("pressure_bar") * 10).floor() / 10).alias("pressure_bin_lower_bar"),
        )
        .group_by("fuel", "flame_type", "o2_bin_lower_pct", "pressure_bin_lower_bar")
        .agg(pl.col(metric).mean().alias("value"), pl.len().alias("n_tests"))
        .sort("fuel", "flame_type", "pressure_bin_lower_bar", "o2_bin_lower_pct")
    )


def load_cfi_metadata(root: Path) -> pl.DataFrame:
    path = root / "CFI_PSI-39/xlsx/CFI_DataSummary_Spreadsheet_Jan2021.xlsx"
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook["Master Sheet"].iter_rows(values_only=True))
    finally:
        workbook.close()
    headers = [str(value).strip() for value in rows[0]]
    raw = pl.DataFrame(
        [{headers[i]: None if value is None else str(value).strip()
          for i, value in enumerate(row[:len(headers)])}
         for row in rows[1:] if row[0] is not None],
        infer_schema_length=None,
    )
    return raw.select(
        pl.concat_str(pl.lit("FLEX-"), pl.col("Flex Number").str.replace(r"\.0$", ""))
        .alias("test_id"),
        pl.col("CIR").alias("cir_test_id"),
        pl.col("Fuel").alias("fuel"),
        pl.col("Pressure").alias("pressure_original"),
        (numeric("Oxygen") * 100).alias("o2_pct"),
        (numeric("Nitrogen") * 100).alias("n2_pct"),
        (numeric("Helium") * 100).alias("he_pct"),
        pl.lit(path.relative_to(root).as_posix()).alias("source_file"),
    )


CFI_CHANNELS = {
    "DropletData": ("DropletTime_sec", {"DropletDiameter_mm": ("droplet_diameter", "mm")}),
    "CoolFlameData": ("CoolFlameTime_s", {
        "CoolFlameDiameterFWHM_mm": ("cool_flame_fwhm", "mm"),
        "CoolFlameDiameterP2P_mm": ("cool_flame_p2p", "mm"),
    }),
    "RadiometerData": ("RadiometerTime_sec", {
        "TotalWideBandPower_W": ("wideband_power", "W"),
        "TotalNarrowBandPower_W": ("narrowband_power", "W"),
    }),
}


def load_cfi(root: Path) -> tuple[pl.DataFrame, list[dict]]:
    frames = []
    excluded = []
    for kind, (time, channels) in CFI_CHANNELS.items():
        paths = sorted((root / "CFI_PSI-39/txt").glob(f"FLEX-*_{kind}.txt"))
        for path in paths:
            # Some NASA archive entries have .txt names but contain PDF/binary data.
            # Inspect the actual header instead of trusting the extension.
            with path.open("rb") as stream:
                header = stream.read(4096)
            if not header.startswith((time + "\t").encode("utf-8")):
                excluded.append({
                    "path": path.relative_to(root).as_posix(),
                    "reason": "Expected TSV header missing; file contains PDF or other non-tabular content",
                })
                continue
            # Polars handles LF and CRLF with its default LF terminator, but
            # legacy CFI files use standalone CR, which must be set explicitly.
            first_cr = header.find(b"\r")
            first_lf = header.find(b"\n")
            standalone_cr = first_cr >= 0 and (
                first_lf < 0 or first_cr < first_lf - 1
            )
            raw = pl.read_csv(
                path, separator="\t", infer_schema=False,
                eol_char="\r" if standalone_cr else "\n",
            )
            for source, (channel, unit) in channels.items():
                frames.append(raw.select(
                    pl.lit(path.name.split("_")[0]).alias("test_id"),
                    numeric(time).alias("time_s"), numeric(source).alias("value"),
                    pl.lit(channel).alias("channel"), pl.lit(unit).alias("unit"),
                    pl.lit(path.relative_to(root).as_posix()).alias("source_file"),
                ))
    if not frames:
        raise FileNotFoundError(f"No FLEX-*_DropletData/CoolFlameData/RadiometerData.txt in {root}")
    data = pl.concat(frames).with_columns(
        (pl.col("value") < 0).fill_null(False).alias("negative_value"),
        (pl.col("value").is_finite() & pl.col("time_s").is_finite())
        .fill_null(False).alias("finite_value"),
    )
    diameter = data.filter(pl.col("channel") == "droplet_diameter").with_columns(
        pl.col("value").pow(2).alias("value"),
        pl.lit("droplet_diameter_squared").alias("channel"), pl.lit("mm²").alias("unit"),
    )
    return pl.concat([data, diameter]).sort("test_id", "channel", "time_s"), excluded


def load_daft(root: Path) -> pl.DataFrame:
    frames = []
    for path in sorted((root / "DAFT-DAFT-2_PSI-47/txt").glob("[DP]Trk*.txt")):
        # NUL bytes at EOF are transport padding, not particle observations.
        text = path.read_text().rstrip("\x00\r\n \t")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        header_end = next(i for i, line in enumerate(lines) if line.startswith("Number of points:"))
        metadata = dict(line.split(":", 1) for line in lines[:header_end + 1])
        metadata = {key: value.strip() for key, value in metadata.items()}
        values = lines[header_end + 1:]
        declared_points = int(metadata["Number of points"])
        if len(values) < declared_points:
            raise ValueError(f"Point count mismatch: {path}")
        if not all(re.fullmatch(r"[-+\d.eE]+", value) for value in values):
            raise ValueError(f"Unexpected nonnumeric DAFT sample: {path}")
        frames.append(pl.DataFrame({"raw": values}).with_row_index("sample_index").select(
            pl.col("sample_index"), numeric("raw").alias("value"),
            pl.lit(declared_points).alias("declared_points"),
            (pl.col("sample_index") >= declared_points).alias("beyond_declared_count"),
            pl.lit(metadata["Instrument"]).alias("instrument"),
            pl.lit(metadata["TestID"]).alias("test_id"),
            pl.lit(metadata["Units"]).alias("unit"),
            pl.lit(metadata["StartDate"] + " " + metadata["StartTime"]).alias("start_original"),
            pl.lit(path.relative_to(root).as_posix()).alias("source_file"),
        ))
    if not frames:
        raise FileNotFoundError(f"No DAFT DTrk/PTrk files in {root}")
    return pl.concat(frames)


def downsample(data: pl.DataFrame, x: str, limit: int = 1000) -> pl.DataFrame:
    """Per-curve min/max envelope, retaining peaks, endpoints and original x values."""
    if limit < 4:
        raise ValueError("Point limit must be at least 4")
    data = data.sort(x)
    if data.height <= limit:
        return data
    # Reserve room for a null marker in each bucket to avoid bridging missing data.
    bucket_count = max(1, (limit - 2) // 3)
    bucket_size = (data.height + bucket_count - 1) // bucket_count
    indexed = data.with_row_index("_row").with_columns(
        (pl.col("_row") // bucket_size).alias("_bucket")
    )
    extrema = indexed.group_by("_bucket").agg(
        pl.col("_row").sort_by("value", nulls_last=True).first().alias("_min"),
        pl.col("_row").sort_by("value", descending=True, nulls_last=True).first().alias("_max"),
        pl.col("_row").filter(pl.col("value").is_null()).first().alias("_gap"),
    )
    indices = {0, data.height - 1, *extrema["_min"].to_list(), *extrema["_max"].to_list()}
    for gap in extrema["_gap"].drop_nulls().to_list():
        if len(indices) < limit:
            indices.add(gap)
    return indexed.filter(pl.col("_row").is_in(sorted(indices))).drop("_row", "_bucket")
