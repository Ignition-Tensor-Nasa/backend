from data_prep.prepare import elapsed_seconds


def test_elapsed_seconds_uses_first_sample_and_continues_past_midnight():
    assert elapsed_seconds(["23:59:59:500", "23:59:59:500", "00:00:00:250"]) == [0.0, 0.0, 0.75]


def test_elapsed_seconds_against_another_recordings_start():
    assert elapsed_seconds(["17:12:12:622", "17:12:13:631"], start=61932.631) == [-0.009, 1.0]
