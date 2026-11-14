"""Projects (missions) over the curated catalogue, and multi-run overlays for charts."""

from functools import lru_cache
import json

import polars as pl

from app.services import catalogue
from app.services.catalogue import (
    CATEGORY_KEYS, RANGE_KEYS, SUMMARY_COLUMNS, CatalogueUnavailable, Filters,
    check_categories, dataset_frame, filter_options, load_catalogue, load_datasets, load_events,
    search_cases)

IGNITION = "Ignition and retraction"
ALIGNS = ("start", "ignition")


@lru_cache(maxsize=1)
def load_projects() -> dict[str, dict]:
    path = catalogue.CURATED_DIR / "projects.json"
    if not path.is_file():
        raise CatalogueUnavailable(
            "Project registry is missing; run the B01 data preparation (python3 -m data_prep.prepare).")
    projects = json.loads(path.read_text("utf-8"))["projects"]
    return {item["project_id"]: item for item in sorted(projects, key=lambda p: p["order"])}


def _kind(dataset: dict) -> str:
    """'sensors', 'droplet', ... : the dataset id without its case id prefix."""
    return dataset["dataset_id"][len(dataset["case_id"]) + 1:]


def _scope(project: dict) -> pl.DataFrame:
    return load_catalogue().filter(pl.col("investigation").is_in(project["investigations"]))


def _views(project: dict) -> list[dict]:
    """Overlay views that exist for this project: one per dataset kind with its channels."""
    cases = set(_scope(project)["case_id"])
    kinds: dict[str, dict] = {}
    for dataset in load_datasets().values():
        if dataset["case_id"] not in cases:
            continue
        entry = kinds.setdefault(_kind(dataset), {"kind": _kind(dataset), "label": dataset["label"],
                                                  "default_y": dataset["default_y"], "channels": [],
                                                  "cases": 0})
        entry["cases"] += 1
        for column in dataset["columns"]:
            if column["key"] != dataset["default_x"] and column["key"] not in {
                    c["key"] for c in entry["channels"]}:
                entry["channels"].append(column)
    return sorted(kinds.values(), key=lambda view: -view["cases"])


def project_summary(project: dict) -> dict:
    scope = _scope(project)
    return {key: project[key] for key in ("project_id", "title", "mission", "summary", "measures", "source")} | {
        "case_count": scope.height,
        "fuels": sorted(scope["fuel"].drop_nulls().unique()),
    }


def list_projects() -> list[dict]:
    return [project_summary(project) for project in load_projects().values()]


def get_project(project_id: str) -> dict | None:
    project = load_projects().get(project_id)
    if project is None:
        return None
    scope = _scope(project)
    options: dict = {key: sorted(scope[key].drop_nulls().unique()) for key in CATEGORY_KEYS}
    for key in RANGE_KEYS:
        known = scope[key].drop_nulls()
        options[key] = {"min": known.min(), "max": known.max()} if len(known) else None
    return project_summary(project) | {"filters": options, "views": _views(project)}


def scoped_search(project_id: str, filters: Filters) -> list[dict] | None:
    """Search restricted to the project's investigations (the filter cannot widen it)."""
    project = load_projects().get(project_id)
    if project is None:
        return None
    inside = set(project["investigations"])
    wanted = set(filters.investigation or inside)
    return search_cases(filters.model_copy(update={"investigation": sorted(wanted & inside) or sorted(inside)}))


def _envelope(frame: pl.DataFrame, limit: int) -> pl.DataFrame:
    """Min/max per bucket, keeping first/last rows and peaks; frame has columns x, y."""
    if frame.height <= limit:
        return frame
    buckets = max(1, (limit - 2) // 2)
    size = -(-frame.height // buckets)
    indexed = frame.with_row_index("i").with_columns((pl.col("i") // size).alias("b"))
    extremes = indexed.group_by("b").agg(
        pl.col("i").sort_by("y", nulls_last=True).first().alias("lo"),
        pl.col("i").sort_by("y", descending=True, nulls_last=True).first().alias("hi"))
    keep = sorted({0, frame.height - 1, *extremes["lo"].to_list(), *extremes["hi"].to_list()})
    return indexed.filter(pl.col("i").is_in(keep)).drop("i", "b")


def overlay(project_id: str, kind: str, channel: str, filters: Filters, align: str,
            max_series: int, max_points: int) -> dict | None:
    """Several runs of one project on one axis, time-aligned, sampled with a stated method."""
    project = load_projects().get(project_id)
    if project is None:
        return None
    check_categories(filters)
    cases = scoped_search(project_id, filters)
    events = load_events()
    datasets = {d["case_id"]: d for d in load_datasets().values() if _kind(d) == kind}
    series, skipped_no_ignition, unit, label, x_unit = [], 0, None, channel, "s"
    found_channel = False
    for case in cases:
        dataset = datasets.get(case["case_id"])
        if dataset is None:
            continue
        column = next((c for c in dataset["columns"] if c["key"] == channel), None)
        if column is None:
            continue
        found_channel = True
        t0 = 0.0
        if align == "ignition":
            label = project.get("ignition_label", IGNITION)
            mark = next((e["time_s"] for e in events.get(case["case_id"], [])
                         if e["label"] == label and e["dataset_id"] == dataset["dataset_id"]), None)
            if mark is None:
                skipped_no_ignition += 1
                continue
            t0 = mark
        found = dataset_frame(dataset["dataset_id"])
        if found is None:
            continue
        frame = found[1].select(
            (pl.col(dataset["default_x"]) - t0).alias("x"), pl.col(channel).alias("y")
        ).drop_nulls("x")
        if frame.height == 0:
            continue
        series.append((case, frame))
        unit, label, x_unit = column["unit"], column["label"], next(
            c["unit"] for c in dataset["columns"] if c["key"] == dataset["default_x"])
    if not found_channel:
        raise ValueError(f"No dataset of kind '{kind}' in this project has channel '{channel}'")
    truncated = len(series) > max_series
    items = []
    for case, frame in series[:max_series]:
        thinned = _envelope(frame, max_points)
        items.append({**case, "n_source": frame.height, "sampled": thinned.height < frame.height,
                      "points": thinned.rows()})
    notes = [f"Each curve is a min/max envelope of at most {max_points} points per run (peaks kept, "
             "no smoothing or interpolation); n_source gives the full sample count."]
    if align == "ignition":
        notes.append(f"t = 0 is the '{project.get('ignition_label', IGNITION)}' event of each run; {skipped_no_ignition} "
                     "matching run(s) have no such event and are left out.")
    else:
        notes.append("t = 0 is the first sample of each recording; runs are not aligned to an event.")
    if truncated:
        notes.append(f"{len(series)} runs matched; the first {max_series} by case ID are returned.")
    return {"project_id": project_id, "kind": kind,
            "channel": {"key": channel, "label": label, "unit": unit},
            "x": {"label": "Time since ignition" if align == "ignition" else "Elapsed time",
                  "unit": x_unit, "align": align},
            "matched": len(series), "series": items, "note": " ".join(notes)}
