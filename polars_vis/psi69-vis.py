
import polars as pl
from pathlib import Path

columns = {
    "FLEX Test #": "test_id",
    "FLEX Identifier": "identifier",
    "Fuel": "fuel",
    "Ambient pressure; mmHg": "pressure",
    "O initial ambient composition; mole fraction": "o2",
    "N initial ambient composition; mole fraction": "n2",
    "CO initial ambient composition; mole fraction": "co2",
    "He initial ambient composition; mole fraction": "he",
    "Droplet initial diameter; mm": "d0",
    "Visible flame extinction diameter; mm": "d_ext",
    "Burning rate; mm": "k",
    "Burn time; s": "burn_time",
    "Test end": "test_end",
}

numeric = [
    "pressure", "o2", "n2", "co2",
    "he", "d0", "d_ext", "k"
]

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "manual_saved_csv"
    / "PSI-69_Experimental table_FLEX.csv"
)

lf = (
    pl.scan_csv(
        CSV_PATH,
        infer_schema=False,
        encoding="utf8-lossy",
        null_values=["", "\uFFFD"],
    )
    .rename(columns)
    .with_columns(
        [pl.col(c).cast(pl.Float64, strict=False)
         for c in numeric]
        + [
            pl.col("burn_time")
              .str.starts_with("~")
              .alias("burn_time_approx"),
            pl.col("burn_time")
              .str.replace(r"^~", "")
              .cast(pl.Float64, strict=False)
              .alias("burn_time"),
        ]
    )
    .with_columns(
        pl.when(pl.col("pressure") > 0)
          .then(pl.col("pressure") / 760)
          .otherwise(None)
          .alias("pressure_atm"),

        pl.when(
            (pl.col("test_end") == "Extinction")
            & (pl.col("d0") > 0)
            & (pl.col("d_ext") >= 0)
            & (pl.col("d_ext") <= pl.col("d0"))
        )
        .then(pl.col("d_ext") / pl.col("d0"))
        .otherwise(None)
        .alias("extinction_ratio"),
    )
)

# First interactive visualisation
plot_data = (
    lf.filter(
        pl.col("fuel") == "Methanol",
        pl.col("pressure_atm").is_between(0.95, 1.05),
        pl.col("co2") > 0,
        pl.col("extinction_ratio").is_not_null(),
    )
    .select(
        "test_id", "o2", "co2", "d0",
        "extinction_ratio", "burn_time", "k"
    )
    .collect()
    .to_pandas()
)

import plotly.express as px

fig = px.scatter(
    plot_data,
    x="co2",
    y="o2",
    size="d0",
    color="extinction_ratio",
    hover_data=["test_id", "k", "burn_time"],
    color_continuous_scale="RdYlBu",
    labels={
        "co2": "CO₂ mole fraction",
        "o2": "O₂ mole fraction",
        "extinction_ratio": "D_ext / D_initial",
        "d0": "Initial diameter (mm)",
    },
)

fig.update_xaxes(tickformat=".0%")
fig.update_yaxes(tickformat=".0%")
fig.show()
