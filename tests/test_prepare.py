import polars as pl
import pytest

from data_prep.flex import d2_fit
from data_prep.prepare import Skip, elapsed_seconds, tp_events


def test_d2_fit_recovers_a_known_burning_rate_and_exports_the_line_only_in_the_window():
    times = [index * 0.5 for index in range(41)]  # 0..20 s
    # D² = 16 - 0.4 t, flat before t = 2 (outside the window)
    droplet = pl.DataFrame({"time_s": times,
                            "DropletDiameter_mm": [(16 - 0.4 * max(t, 2.0)) ** 0.5 for t in times]})
    frame, fit = d2_fit(droplet, 2.0, 18.0)
    assert fit["k_mm2_s"] == pytest.approx(0.4, abs=1e-4) and fit["r2"] == 1.0 and fit["n"] == 33
    assert fit["intercept_mm2"] == pytest.approx(16.0, abs=1e-3)
    assert frame.columns == ["time_s", "D2_mm2", "D2_fit_mm2"]
    assert frame["D2_fit_mm2"].null_count() == 8  # 4 samples before and 4 after the window
    with pytest.raises(Skip):
        d2_fit(droplet, 100.0, 200.0)


def test_tp_events_are_relative_to_the_first_sample_and_skip_blank_entries(tmp_path):
    log = tmp_path / "TP.txt"
    log.write_text("Test Point: X\nMFCs off: 2021-07-15 17:13:25\nPreheat igniter: ***\n"
                   "Ignition and retraction: 2021-07-15 17:12:48\nScript start: 2021-07-15 17:11:32\n")
    assert tp_events(log, 17 * 3600 + 12 * 60 + 12.5) == [
        (35.5, "Ignition and retraction"), (72.5, "MFCs off")]
    assert tp_events(tmp_path / "absent.txt", 0.0) == []


def test_elapsed_seconds_uses_first_sample_and_continues_past_midnight():
    assert elapsed_seconds(["23:59:59:500", "23:59:59:500", "00:00:00:250"]) == [0.0, 0.0, 0.75]


def test_elapsed_seconds_against_another_recordings_start():
    assert elapsed_seconds(["17:12:12:622", "17:12:13:631"], start=61932.631) == [-0.009, 1.0]
