from data_prep.prepare import elapsed_seconds, tp_events


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
