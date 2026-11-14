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


def test_nearest_neighbours_scale_within_investigation_and_skip_missing_values():
    from data_prep.similarity import nearest_neighbours

    catalogue = pl.DataFrame({
        "case_id": ["a", "b", "c", "d", "x", "y"],
        "investigation": ["one"] * 4 + ["two"] * 2,
        "o2_pct": [20.0, 21.0, 40.0, None, 20.0, 20.5],
        "pressure_kpa": [100.0, 100.0, 100.0, 100.0, 50.0, 500.0],
    })
    result = nearest_neighbours(catalogue, k=2)
    assert [item["case_id"] for item in result["a"]] == ["b", "c"]  # pressure is constant: no effect
    assert result["a"][0]["distance"] < result["a"][1]["distance"]
    assert result["d"] == []  # unknown oxygen: no neighbours, and nobody's neighbour
    assert all(item["case_id"] != "d" for items in result.values() for item in items)
    assert [item["case_id"] for item in result["x"]] == ["y"]  # never across investigations


def test_explanation_uses_only_table_fields():
    from data_prep.prepare import explain_cfig

    row = {"Fuel": "30% butane", "Hot ignition": "Yes", "burn time (sec)": "35",
           "Self-ext or flow term": "SE", "Cool flame?": "yes, PCI-D"}
    assert explain_cfig(row, 39.537, 202.66) == (
        "Gas flame test with 30% butane in 39.5% oxygen and about 203 kPa. It ignited, burned for "
        "35 s and went out by itself. The table marks a cool flame in this run.")
    assert explain_cfig({"Fuel": "propane", "Hot ignition": "No", "Cool flame?": "no, PCI-hot"},
                        None, 101.33) == (
        "Gas flame test with propane in about 101 kPa. The source table records no hot ignition.")


def test_two_pass_vocabulary_proposal_classification_and_comparison():
    from data_prep import two_pass

    assert two_pass.propose_fuel(" 30% butane ") == "butane"
    assert two_pass.propose_fuel("Farnesane") == "farnesane"
    assert two_pass.propose_fuel("C12H26") is None  # formulas need a reviewer
    assert two_pass.propose_fuel("[.60]/[.40] C12H26/iso-dodecane") is None
    vocabulary = {"investigations": {"INV": {"fuel": {"C12H26": {"canonical": "dodecane"},
                                                       "blend": {"canonical": None}}}}}
    assert two_pass.classify_fuel(vocabulary, "INV", " C12H26 ") == "dodecane"
    assert two_pass.classify_fuel(vocabulary, "INV", "blend") is None
    assert two_pass.classify_fuel(vocabulary, "INV", "kerosene") is None  # not in the vocabulary
    assert two_pass.classify_fuel(vocabulary, "OTHER", "C12H26") is None
    row = {"case_id": "a", "investigation": "INV", "fuel": "dodecane", "o2_pct": 21.0,
           "pressure_kpa": 101.3, "z_st": None}
    same = pl.DataFrame([row])
    assert two_pass.differences(same, same.with_columns(pl.lit("x").alias("note"))) == []
    changed = pl.DataFrame([row | {"fuel": None}, row | {"case_id": "b"}])
    assert two_pass.differences(same, changed) == [
        "b: only in curated output", "a.fuel: two-pass 'dodecane' != curated None"]


def test_acme_conversions_follow_the_nasa_formulas(tmp_path):
    from data_prep.prepare import acme_scales, convert_acme

    recording = tmp_path / "x_data.csv"
    recording.write_text("2021-01-01 1 00:00:00:000 ,MFCA1 Full Scale: 20.000000,MFCA3 Full Scale: 0.500000\n")
    scales = acme_scales(recording)
    assert scales == {"MFCA1": 20.0, "MFCA3": 0.5}
    frame = pl.DataFrame({"PTA3": [0.5, 1.32], "MFCA1StaticMixerFlow": [2.5, 5.0],
                          "MFCA3FuelFlow": [1.0, 5.0], "MFCA2BypassFlow": [1.0, 1.0], "PMT1": [1.0, 2.0]})
    out = convert_acme(frame, scales)
    assert out["PTA3"].to_list() == pytest.approx([0.0, (1.32 - 0.5) * 36.2595 * 6.894757293168 * 1 / 1.0], rel=1e-3)
    assert out["PTA3"][1] == pytest.approx(205.0, abs=0.5)  # 29.7 psia
    assert out["MFCA1StaticMixerFlow"].to_list() == [10.0, 20.0]
    assert out["MFCA3FuelFlow"].to_list() == [0.1, 0.5]
    assert out["MFCA2BypassFlow"].to_list() == [1.0, 1.0]  # no scale in the header: left untouched
    assert out["PMT1"].to_list() == [1.0, 2.0]


def test_flame_design_channel_sets_and_explanation(tmp_path):
    from data_prep import flame_design

    standard = tmp_path / "a.csv"
    standard.write_text("pre\nTime,PTA1,RadiometerCenter,RadiometerOffset\n")
    later = tmp_path / "b.csv"
    later.write_text("pre\nTime,PTA1,Radiometer3+30LoGain\n")
    assert flame_design.channels_for(standard)[1][0] == "RadiometerCenter"
    assert flame_design.channels_for(later)[1][0] == "Radiometer3+30LoGain"
    row = {"Fuel": "C2H4", "Tad (K)": "2352.14"}
    assert flame_design.explain(row, "ethylene", 20.7, 103.2) == (
        "Normal (fuel-inside) flame test with ethylene in 20.7% oxygen and about 103 kPa. "
        "The table gives an adiabatic flame temperature of 2352 K.")
    assert flame_design.explain({"Fuel": "C3H8", "Tad (K)": "RAMPED"}, None, None, None) == (
        "Normal (fuel-inside) flame test with C3H8.")
