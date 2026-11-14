"""B11: offline nearest-neighbour artifact over the curated catalogue."""

import math

import polars as pl

FEATURES = ("o2_pct", "pressure_kpa")
K = 5
METRIC = (
    "Euclidean distance between cases of the same investigation in the plane of "
    "(o2_pct, pressure_kpa), each feature standardised to zero mean and unit standard "
    "deviation within that investigation. Cases missing either value have no neighbours and "
    f"are nobody's neighbour. Up to {K} nearest cases, ties ordered by case ID. Fuel and all "
    "other metadata are ignored. The distance only says how close the test conditions are; "
    "it is not a calibrated similarity and not a safety score."
)


def nearest_neighbours(catalogue: pl.DataFrame, k: int = K) -> dict[str, list[dict]]:
    result = {case_id: [] for case_id in catalogue["case_id"]}
    known = catalogue.drop_nulls(list(FEATURES))
    for group in known.partition_by("investigation"):
        points = {}
        columns = []
        for name in FEATURES:
            values = group[name]
            spread = values.std() if group.height > 1 else None
            # A feature that does not vary inside the group contributes no distance.
            columns.append([(value - values.mean()) / spread if spread else 0.0 for value in values])
        for case_id, *point in zip(group["case_id"], *columns):
            points[case_id] = point
        for case_id, point in points.items():
            ranked = sorted((round(math.dist(point, other), 4), other_id)
                            for other_id, other in points.items() if other_id != case_id)
            result[case_id] = [{"case_id": other_id, "distance": distance}
                               for distance, other_id in ranked[:k]]
    return result
