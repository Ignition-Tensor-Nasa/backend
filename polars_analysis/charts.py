"""Interactive, standalone HTML figures using already processed Polars tables."""

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from .loaders import atmosphere_grid, downsample


def save_figure(figure: go.Figure, path: Path, title: str) -> None:
    figure.update_layout(
        title=title, template="plotly_white", font=dict(family="Arial", size=13),
        margin=dict(t=150, b=80), hovermode="closest",
    )
    # Embed Plotly: opening the resulting HTML requires no CDN or running server.
    figure.write_html(path, include_plotlyjs=True, full_html=True,
                      config={"responsive": True, "displaylogo": False})


def atmosphere_chart(data: pl.DataFrame, output: Path) -> None:
    figure = go.Figure()
    labels = []
    for metric, unit in (("burn_rate_mg_s", "mg/s"), ("tad_k", "K")):
        grid = atmosphere_grid(data, metric)
        for group in grid.partition_by("fuel", "flame_type", maintain_order=True):
            fuel, flame = group.row(0, named=True)["fuel"], group.row(0, named=True)["flame_type"]
            lookup = {(row["o2_bin_lower_pct"], round(row["pressure_bin_lower_bar"], 1)): row
                      for row in group.iter_rows(named=True)}
            xs = list(range(0, 100, 5))
            ys = [i / 10 for i in range(36)]
            matrix, counts = [], []
            for y in ys:
                matrix.append([lookup.get((x, y), {}).get("value") for x in xs])
                counts.append([lookup.get((x, y), {}).get("n_tests", 0) for x in xs])
            label = f"{fuel} / {flame} / {metric}"
            labels.append(label)
            figure.add_trace(go.Heatmap(
                x=[x + 2.5 for x in xs], y=[round(y + .05, 2) for y in ys],
                z=matrix, customdata=counts, visible=len(labels) == 1,
                colorbar=dict(title=unit), colorscale="YlOrRd", hoverongaps=False,
                hovertemplate="O₂ bin centre: %{x}%<br>Pressure bin centre: %{y} bar"
                              "<br>Mean: %{z}<br>Tests: %{customdata}<extra></extra>",
                name=label,
            ))
    if not labels:
        raise ValueError("No valid sustained ignitions available for heatmaps")
    figure.update_layout(updatemenus=[dict(
        buttons=[dict(label=label, method="update",
                      args=[{"visible": [i == j for j in range(len(labels))]}])
                 for i, label in enumerate(labels)],
        x=0, y=1.18, xanchor="left",
    )], height=700)
    figure.update_xaxes(title_text="Oxygen concentration [%], bins of 5 percentage points", range=[0, 100])
    figure.update_yaxes(title_text="Pressure [bar], bins of 0.1 bar", range=[0, 3.6])
    save_figure(figure, output / "A_atmosphere.html",
                "ACME CFI-G PSI-159: measured test conditions"
                "<br><sup>Mean for sustained ignitions; empty cells = no observations."
                " Burn rate from source table; Tad is adiabatic temperature, not a sensor reading.</sup>")


def cfi_chart(data: pl.DataFrame, metadata: pl.DataFrame, output: Path,
              point_limit: int, test_id: str | None = None) -> None:
    if test_id:
        data = data.filter(pl.col("test_id") == test_id)
    ids = sorted(data["test_id"].unique().to_list())
    if not ids:
        raise ValueError(f"No CFI data for test {test_id!r}")
    # Prefer a test having a measured cool flame for the initial view.
    initial = "FLEX-1213" if "FLEX-1213" in ids else ids[0]
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.07,
                           subplot_titles=("Droplet D²", "Cool flame diameter", "Wideband radiometer", "Narrowband radiometer"))
    channels = {
        "droplet_diameter_squared": (1, "D² [mm²]"),
        "cool_flame_fwhm": (2, "FWHM [mm]"), "cool_flame_p2p": (2, "P2P [mm]"),
        "wideband_power": (3, "Wideband [W]"), "narrowband_power": (4, "Narrowband [W]"),
    }
    owners = []
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"]
    for current in ids:
        for index, (channel, (row, label)) in enumerate(channels.items()):
            curve = data.filter((pl.col("test_id") == current) & (pl.col("channel") == channel)
                                & pl.col("time_s").is_finite()).with_columns(
                pl.when(pl.col("value").is_finite()).then(pl.col("value"))
                .otherwise(None).alias("value")
            )
            if curve.is_empty():
                continue
            curve = downsample(curve, "time_s", point_limit)
            figure.add_trace(go.Scatter(
                x=curve["time_s"].to_list(), y=curve["value"].to_list(), name=label,
                mode="lines", visible=current == initial, line=dict(color=colors[index]),
                connectgaps=False,
            ), row=row, col=1)
            owners.append(current)
    lookup = {row["test_id"]: row for row in metadata.iter_rows(named=True)}

    def title(current: str) -> str:
        meta = lookup.get(current, {})
        details = f"Fuel: {meta.get('fuel', 'unknown')}; O₂: {meta.get('o2_pct', 'unknown')}%; "
        details += f"He: {meta.get('he_pct', 'unknown')}%; pressure: {meta.get('pressure_original', 'unknown')}"
        return f"CFI PSI-39: {current}<br><sup>{details}</sup>"

    figure.update_layout(height=1100, updatemenus=[dict(
        buttons=[dict(label=current, method="update",
                      args=[{"visible": [owner == current for owner in owners]}, {"title.text": title(current)}])
                 for current in ids], x=0, y=1.11, xanchor="left",
    )])
    for row, unit in enumerate(["D² [mm²]", "Diameter [mm]", "Power [W]", "Power [W]"], 1):
        figure.update_yaxes(title_text=unit, row=row, col=1)
    figure.update_xaxes(title_text="Source time [s]", row=4, col=1)
    figure.add_annotation(text="Original time axes retained. Negative radiometer values require calibration review. "
                         "A missing curve means no data, not zero.", xref="paper", yref="paper",
                         x=0, y=-.10, showarrow=False, xanchor="left", font=dict(size=11))
    save_figure(figure, output / "B_cfi_lifecycle.html", title(initial))


def daft_chart(data: pl.DataFrame, output: Path, point_limit: int) -> None:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=False,
                           subplot_titles=("P-Trak particle number", "DustTrak mass concentration"))
    for group in data.partition_by("source_file", maintain_order=True):
        instrument = group["instrument"][0]
        current = group["test_id"][0]
        start = group["start_original"][0]
        curve = downsample(group.filter(pl.col("value").is_finite()), "sample_index", point_limit)
        figure.add_trace(go.Scatter(
            x=curve["sample_index"].to_list(), y=curve["value"].to_list(),
            mode="lines", name=f"{instrument} {current} ({start})",
        ), row=1 if instrument == "P-Trak" else 2, col=1)
    figure.update_yaxes(title_text="Particle number [particles/cc]", row=1, col=1)
    figure.update_yaxes(title_text="Mass concentration [mg/m³]", row=2, col=1)
    figure.update_xaxes(title_text="Sample index (sampling interval unverified)")
    figure.update_layout(height=850, legend=dict(orientation="h", y=-.18))
    save_figure(figure, output / "D_daft_aerosol.html",
                "DAFT / DAFT-2 PSI-47: aerosol instrument signals"
                "<br><sup>Independent runs; matching TestID values do not establish synchronization."
                " No smoke/dust classification or alarm threshold inferred.</sup>")
