"""Interactive NASA FLEX (PSI-69) explorer.

Run from the backend/ directory:
    python -m streamlit run polars_vis/psi69_streamlit.py

CSV is resolved relative to this script, not the current working directory.
Required: polars, plotly, streamlit.
"""

from pathlib import Path

import polars as pl
import plotly.graph_objects as go
import streamlit as st


CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "manual_saved_csv"
    / "PSI-69_Experimental table_FLEX.csv"
)

COLUMNS = {
    "FLEX Test #": "test_id",
    "FLEX Identifier": "identifier",
    "Test Date": "test_date",
    "Test GMT": "test_gmt",
    "Fuel": "fuel",
    "Ambient pressure; mmHg": "pressure",
    "O initial ambient composition; mole fraction": "o2",
    "N initial ambient composition; mole fraction": "n2",
    # The original table's "CO" refers to CO₂ (lost subscript).
    "CO initial ambient composition; mole fraction": "co2",
    "He initial ambient composition; mole fraction": "he",
    "Droplet initial diameter; mm": "d0",
    "Visible flame extinction diameter; mm": "d_ext",
    # Source header says "mm"; physical unit of the D²-law rate is mm²/s.
    "Burning rate; mm": "k",
    "Burn time; s": "burn_time",
    "Test end": "test_end",
}

NUMERIC_COLUMNS = ["pressure", "o2", "n2", "co2", "he", "d0", "d_ext", "k"]

METRICS = {
    "extinction_ratio": "D_ext / D_initial",
    "burn_time": "Burn time [s]",
    "k": "Burning rate [mm²/s]",
}


# Every filter is applied on the LazyFrame before collecting a small result.
def scan_flex(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(
            path,
            infer_schema=False,  # Read as strings, then cast explicitly.
            encoding="utf8-lossy",  # NASA CSV may contain non-UTF-8 bytes.
            null_values=["", "\uFFFD"],
        )
        .rename(COLUMNS)
        .with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC_COLUMNS]
            + [
                pl.col("test_id").cast(pl.Int64, strict=False),
                pl.col("burn_time")
                .str.starts_with("~")
                .fill_null(False)
                .alias("burn_time_approx"),
                pl.col("burn_time")
                .str.replace(r"^~", "")
                .cast(pl.Float64, strict=False)
                .alias("burn_time"),
            ]
        )
        .with_columns(
            pl.when(pl.col("pressure") > 0)
            .then(pl.col("pressure") / 760.0)
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


def format_value(value: float | None, decimals: int = 2) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def scatter_figure(rows: list[dict], gas: str, metric: str) -> go.Figure:
    gas_col = "co2" if gas == "CO₂" else "he"
    gas_title = "CO₂ [% mol]" if gas == "CO₂" else "He [% mol]"

    hover = []
    for row in rows:
        hover.append(
            f"<b>FLEX #{row['test_id']} — {row['identifier']}</b><br>"
            f"Paliwo: {row['fuel']}<br>"
            f"O₂: {100 * row['o2']:.1f}%<br>"
            f"{gas}: {100 * row[gas_col]:.1f}%<br>"
            f"Ciśnienie: {format_value(row['pressure_atm'])} atm<br>"
            f"D₀: {format_value(row['d0'])} mm<br>"
            f"D przy zgaśnięciu: {format_value(row['d_ext'])} mm<br>"
            f"R: {format_value(row['extinction_ratio'], 3)}<br>"
            f"Burn time: {format_value(row['burn_time'])} s<br>"
            f"Burning rate: {format_value(row['k'])} mm²/s<br>"
            f"Test end: {row['test_end']}"
        )

    color_kwargs = {"cmin": 0, "cmax": 1} if metric == "extinction_ratio" else {}
    fig = go.Figure(
        go.Scatter(
            x=[row[gas_col] * 100 for row in rows],
            y=[row["o2"] * 100 for row in rows],
            mode="markers",
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            customdata=[[int(row["test_id"])] for row in rows],
            marker=dict(
                size=[max(11, (row["d0"] or 1.5) * 9) for row in rows],
                color=[row[metric] for row in rows],
                colorscale="RdYlBu" if metric == "extinction_ratio" else "Viridis",
                showscale=True,
                colorbar=dict(title=dict(text=METRICS[metric])),
                opacity=0.85,
                line=dict(width=1, color="#FFFFFF"),
                **color_kwargs,
            ),
        )
    )
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=20, t=25, b=10),
        xaxis_title=gas_title,
        yaxis_title="O₂ [% mol]",
        dragmode="select",  # Box selection by default; click also selects points.
        clickmode="event+select",
        hovermode="closest",
    )
    return fig


def comparison_figure(rows: list[dict], metric: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[f"Test #{r['test_id']}" for r in rows],
            y=[r[metric] for r in rows],
            text=[format_value(r[metric], 3) for r in rows],
            textposition="auto",
            marker_color="#4C78A8",
        )
    )
    fig.update_layout(
        height=350,
        yaxis_title=METRICS[metric],
        xaxis_title="Eksperyment",
        margin=dict(l=10, r=10, t=25, b=10),
    )
    if metric == "extinction_ratio":
        fig.update_yaxes(range=[0, 1.05])
    return fig


def main() -> None:
    st.set_page_config(page_title="NASA FLEX | PSI-69", page_icon="🔥", layout="wide")
    st.title("NASA FLEX · PSI-69")
    st.caption("Interaktywny eksplorator spalania kropli w mikrograwitacji · Polars Lazy + Plotly")

    if not CSV_PATH.is_file():
        st.error(f"Nie znaleziono pliku CSV: {CSV_PATH}")
        st.stop()

    try:
        lf = scan_flex(CSV_PATH)
        with st.sidebar:
            st.header("Filtry")
            fuel = st.selectbox("Paliwo", ["Methanol", "Heptane"], index=0)
            gas = st.selectbox("Gaz rozcieńczający", ["CO₂", "He"], index=0)
            center_atm = st.slider("Ciśnienie docelowe [atm]", 0.5, 3.5, 1.0, 0.05)
            tolerance = st.slider("Tolerancja ciśnienia [atm]", 0.01, 0.35, 0.05, 0.01)
            metric = st.selectbox(
                "Kolor punktów",
                list(METRICS),
                index=0,
                format_func=lambda key: METRICS[key],
            )
            st.caption(f"Filtr ciśnienia: {center_atm - tolerance:.2f}–{center_atm + tolerance:.2f} atm")

        gas_col = "co2" if gas == "CO₂" else "he"
        other_gas_col = "he" if gas == "CO₂" else "co2"
        filtered = lf.filter(
            (pl.col("fuel") == fuel)
            & (pl.col("pressure_atm").is_between(center_atm - tolerance, center_atm + tolerance))
            & (pl.col(gas_col) > 0)
            & (pl.col(other_gas_col) == 0)
        )

        summary = filtered.select(
            pl.len().alias("all_count"),
            pl.col(metric).is_not_null().sum().alias("valid_count"),
        ).collect().row(0, named=True)
        visible = (
            filtered.filter(pl.col(metric).is_not_null())
            .select(
                "test_id", "identifier", "fuel", "o2", "n2", "co2", "he",
                "pressure_atm", "d0", "d_ext", "extinction_ratio", "k",
                "burn_time", "burn_time_approx", "test_end",
            )
            .sort("test_id")
            .collect()
        )
    except Exception as exc:
        st.error("Nie udało się odczytać lub przetworzyć CSV.")
        st.exception(exc)
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Testy po filtrach", summary["all_count"])
    col2.metric("Punkty na wykresie", visible.height)
    col3.metric("Pominięte (brak metryki)", summary["all_count"] - summary["valid_count"])

    st.info(
        "R = D przy zgaśnięciu widocznego płomienia / D początkowe. "
        "Duże R oznacza, że widoczny płomień zgasł, gdy kropla była jeszcze duża. "
        "R jest obliczane tylko dla testów typu Extinction z poprawnymi średnicami. "
        "O₂ i stężenie gazu często zmieniają się jednocześnie — wykres nie dowodzi przyczynowości."
    )

    if visible.is_empty():
        st.warning("Brak punktów dla tych filtrów. Zwiększ tolerancję ciśnienia lub zmień gaz/paliwo.")
        return

    rows = visible.to_dicts()  # No conversion to pandas or pyarrow needed.
    fig = scatter_figure(rows, gas, metric)
    st.subheader("Mapa eksperymentów")
    st.caption(
        "Każdy punkt = jeden test; jego wielkość oznacza średnicę początkową kropli. "
        "Najedź myszą, aby zobaczyć parametry. Klikaj punkty albo zaznacz prostokąt/lasso, "
        "aby wskazać testy do porównania. Punkty o identycznych warunkach mogą się nakładać."
    )
    selected_event = st.plotly_chart(
        fig,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        key=f"flex_chart_{fuel}_{gas}_{center_atm:.2f}_{tolerance:.2f}_{metric}",
        use_container_width=True,
    )
    chart_ids = sorted(
        {
            int(point["customdata"][0])
            for point in selected_event.selection.points
            if point.get("customdata")
        }
    )

    st.subheader("Porównaj eksperymenty")
    test_ids = [int(r["test_id"]) for r in rows]
    id_to_label = {int(r["test_id"]): r["identifier"] for r in rows}

    if "compare_tests" not in st.session_state:
        st.session_state["compare_tests"] = []
    # Remove IDs that no longer satisfy current filters before rendering the widget.
    st.session_state["compare_tests"] = [
        item for item in st.session_state["compare_tests"] if item in set(test_ids)
    ][:6]

    if chart_ids:
        st.caption("Zaznaczone na wykresie: " + ", ".join(f"#{i}" for i in chart_ids))
    if st.button("Dodaj zaznaczone z wykresu do porównania", disabled=not chart_ids):
        st.session_state["compare_tests"] = list(
            dict.fromkeys(st.session_state["compare_tests"] + chart_ids)
        )[:6]
        st.rerun()

    chosen = st.multiselect(
        "Wybierz maksymalnie 6 testów (można też ręcznie, np. gdy punkty się nakładają)",
        options=test_ids,
        key="compare_tests",
        max_selections=6,
        format_func=lambda x: f"#{x} · {id_to_label[x]}",
    )

    if not chosen:
        st.caption("Wybierz co najmniej jeden test. Wybierając dwa lub więcej, łatwo porównasz wyniki.")
    else:
        compared = [row for row in rows if row["test_id"] in chosen]
        compared.sort(key=lambda row: chosen.index(row["test_id"]))
        metric_to_compare = st.selectbox(
            "Metryka na wykresie porównawczym",
            list(METRICS),
            index=list(METRICS).index(metric),
            format_func=lambda key: METRICS[key],
            key="metric_to_compare",
        )
        good_for_bar = [r for r in compared if r[metric_to_compare] is not None]
        if good_for_bar:
            st.plotly_chart(
                comparison_figure(good_for_bar, metric_to_compare),
                use_container_width=True,
            )
        else:
            st.warning("Wybrane eksperymenty nie mają wartości tej metryki.")

        display_rows = [
            {
                "Test #": r["test_id"],
                "Identifier": r["identifier"],
                "O₂ [%]": round(r["o2"] * 100, 2),
                "CO₂ [%]": round(r["co2"] * 100, 2),
                "He [%]": round(r["he"] * 100, 2),
                "Pressure [atm]": r["pressure_atm"],
                "D initial [mm]": r["d0"],
                "D extinction [mm]": r["d_ext"],
                "Extinction ratio": r["extinction_ratio"],
                "Burn time [s]": r["burn_time"],
                "Burn time approx.": r["burn_time_approx"],
                "Burning rate [mm²/s]": r["k"],
                "Test end": r["test_end"],
            }
            for r in compared
        ]
        st.dataframe(display_rows, hide_index=True, use_container_width=True)

    with st.expander("Surowe rekordy widoczne na wykresie"):
        st.dataframe(visible, hide_index=True, use_container_width=True)
        st.download_button(
            "Eksport przefiltrowanych danych do CSV",
            data=visible.write_csv().encode("utf-8"),
            file_name="psi69_filtered.csv",
            mime="text/csv",
        )

    st.caption(
        "Uwagi: Extinction = widoczne zgaśnięcie płomienia, nie ocena ogólnego "
        "ryzyka pożaru. Braki danych i niepoprawne średnice dają null w R; "
        "oryginalny CSV pozostaje niezmieniony."
    )


if __name__ == "__main__":
    main()
