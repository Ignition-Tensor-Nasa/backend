"""Run with python -m polars_analysis from backend/."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import polars as pl

from .charts import atmosphere_chart, cfi_chart, daft_chart
from .loaders import atmosphere_grid, load_atmosphere, load_cfi, load_cfi_metadata, load_daft


def run(data_dir: Path, output: Path, point_limit: int, test_id: str | None) -> dict:
    atmosphere, quality = load_atmosphere(data_dir)
    metadata = load_cfi_metadata(data_dir)
    cfi, excluded = load_cfi(data_dir)
    daft = load_daft(data_dir)
    if test_id and test_id not in cfi["test_id"].unique().to_list():
        raise ValueError(f"Unknown CFI test: {test_id}")
    output.mkdir(parents=True, exist_ok=True)
    processed = output / "processed"
    processed.mkdir(exist_ok=True)
    tables = {"acme_conditions": atmosphere, "cfi_metadata": metadata,
              "cfi_series": cfi, "daft_series": daft,
              "acme_burn_rate_grid": atmosphere_grid(atmosphere, "burn_rate_mg_s"),
              "acme_tad_grid": atmosphere_grid(atmosphere, "tad_k")}
    for name, table in tables.items():
        table.write_parquet(processed / f"{name}.parquet")
    atmosphere.write_csv(processed / "acme_conditions.csv")
    metadata.write_csv(processed / "cfi_metadata.csv")

    # Snapshot precisely the inputs used; hashes make later reruns auditable.
    sources = sorted(set().union(*(set(table["source_file"].unique().to_list())
                                  for table in (atmosphere, metadata, cfi, daft))))
    sources = sorted(set(sources) | {entry["path"] for entry in excluded})
    manifest = []
    for relative in sources:
        original = data_dir / relative
        target = output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.resolve() != target.resolve():
            shutil.copyfile(original, target)
        manifest.append({"path": relative, "bytes": original.stat().st_size,
                         "sha256": sha256(original.read_bytes()).hexdigest()})

    atmosphere_chart(atmosphere, output)
    cfi_chart(cfi, metadata, output, point_limit, test_id)
    daft_chart(daft, output, point_limit)
    cfi_quality = cfi.group_by("channel").agg(
        pl.len().alias("rows"), pl.col("negative_value").sum().alias("negative_values"),
        (~pl.col("finite_value")).sum().alias("nonfinite_values"),
    ).sort("channel")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {name: version(name) for name in ("polars", "plotly", "openpyxl")},
        "source_catalog": "https://psi.nasa.gov/physci/repo",
        "source_mode": "local NASA PSI files, copied into sources/; no synthetic data",
        "atmosphere_quality": quality,
        "cfi_quality": cfi_quality.to_dicts(),
        "excluded_files": excluded,
        "cfi_tests": cfi["test_id"].n_unique(),
        "daft_rows": daft.height,
        "daft_quality": daft.group_by("source_file").agg(
            pl.len().alias("actual_points"), pl.col("declared_points").first(),
            pl.col("beyond_declared_count").sum().alias("extra_points"),
        ).sort("source_file").to_dicts(),
        "downsampling": {"method": "min/max envelope per channel", "max_points": point_limit},
        "sources": manifest,
        "limitations": [
            "INSTRUKCJA.md was empty at implementation time; scope follows VISUALIZATION_IDEAS.md A, B and DAFT part of D.",
            "Burn rate is a source-table quantity; ignition and positive burn time filter sustained ignitions.",
            "Heatmaps show conditional means, not fire probability or a habitat risk index. Missing cells are not interpolated.",
            "Negative radiometer readings retained and flagged; calibration and synchronization require source documentation.",
            "No burning constant K fitted automatically: a physical burning phase must first be selected.",
            "DAFT sampling interval not verified. Instrument runs not synchronized; no SAME smoke comparison implemented.",
            "C, E, F and G need sensor calibration/geometry, units or curated metadata before implementation.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    links = [("A_atmosphere.html", "A. ACME CFI-G — atmosfera i spalanie"),
             ("B_cfi_lifecycle.html", "B. CFI/FLEX — kropla, chłodny płomień i radiometria"),
             ("D_daft_aerosol.html", "D. DAFT — sygnały aerozolu")]
    items = "".join(f'<li><a href="{path}">{escape(label)}</a></li>' for path, label in links)
    notes = "".join(f"<li>{escape(note)}</li>" for note in report["limitations"])
    (output / "index.html").write_text(
        '<!doctype html><html lang="pl"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>NASA PSI — analiza Polars</title><style>'
        'body{font:17px/1.6 system-ui;max-width:1000px;margin:60px auto;padding:0 24px;background:#f8fafc;color:#172033}'
        'h1{font-size:36px}a{color:#1656b8}li{margin:12px 0}</style>'
        '<h1>NASA PSI — analiza Polars</h1>'
        f'<p>ACME: {atmosphere.height} testów. CFI: {report["cfi_tests"]} testy. '
        f'DAFT: {daft.height} pomiarów. Źródła: lokalne pliki NASA PSI.</p><ul>{items}</ul>'
        '<p>Wykresy HTML działają bez internetu. Dane po przetworzeniu: katalog processed/. '
        '<a href="report.json">Raport jakości i manifest SHA-256</a>.</p>'
        f'<p>Pliki CFI pominięte z powodu niezgodnej zawartości: {len(excluded)}. '
        'Lista przyczyn znajduje się w raporcie; oryginały zachowano w sources/.</p>'
        f'<h2>Metoda i ograniczenia</h2><ul>{notes}</ul></html>', encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="NASA PSI: Polars processing and offline Plotly charts")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--max-points", type=int, default=1000, help="Maximum displayed points per curve (>=4)")
    parser.add_argument("--test-id", help="Show a single FLEX test instead of the full selector")
    args = parser.parse_args()
    if args.max_points < 4:
        parser.error("--max-points must be >= 4")
    try:
        report = run(args.data_dir.resolve(), args.output_dir.resolve(), args.max_points, args.test_id)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(f"Created: {args.output_dir.resolve() / 'index.html'}")
    print(f"Sources: {len(report['sources'])}; CFI tests: {report['cfi_tests']}")


if __name__ == "__main__":
    main()
