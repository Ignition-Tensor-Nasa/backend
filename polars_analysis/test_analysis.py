"""Run from backend/: python -m unittest polars_analysis.test_analysis -v."""

from pathlib import Path
import tempfile
import unittest

import polars as pl

from .loaders import atmosphere_grid, downsample, load_cfi, load_daft


class ProcessingTests(unittest.TestCase):
    def test_heatmap_excludes_failed_ignition_and_keeps_groups_separate(self):
        data = pl.DataFrame({
            "fuel": ["ethane", "ethane", "ethane", "propane"],
            "flame_type": ["Normal"] * 4,
            "o2_pct": [21., 22., 21., 21.], "pressure_bar": [1.01] * 4,
            "burn_rate_mg_s": [2., 4., 100., 8.],
            "valid_conditions": [True] * 4,
            "sustained_ignition": [True, True, False, True],
        })
        result = atmosphere_grid(data, "burn_rate_mg_s")
        ethane = result.filter(pl.col("fuel") == "ethane").row(0, named=True)
        self.assertEqual(ethane["value"], 3.)
        self.assertEqual(ethane["n_tests"], 2)
        self.assertEqual(ethane["o2_bin_lower_pct"], 20.)
        self.assertEqual(result.height, 2)

    def test_downsampling_retains_peak_trough_endpoints_and_limit(self):
        values = [0.] * 10001
        values[2743] = 100.
        values[8419] = -80.
        data = pl.DataFrame({"time": range(len(values)), "value": values})
        result = downsample(data, "time", 100)
        self.assertLessEqual(result.height, 100)
        self.assertEqual(result["time"][0], 0)
        self.assertEqual(result["time"][-1], 10000)
        self.assertEqual(result["value"].max(), 100.)
        self.assertEqual(result["value"].min(), -80.)
        self.assertTrue(result["time"].is_sorted())

    def test_downsampling_keeps_a_missing_value_marker(self):
        data = pl.DataFrame({"time": range(100), "value": [float(i) if i != 40 else None for i in range(100)]})
        result = downsample(data, "time", 20)
        self.assertLessEqual(result.height, 20)
        self.assertEqual(result["value"].null_count(), 1)
        self.assertEqual(result["value"].min(), 0.)
        self.assertEqual(result["value"].max(), 99.)

    def test_cfi_squares_diameter_and_flags_negative_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "CFI_PSI-39/txt"
            folder.mkdir(parents=True)
            (folder / "FLEX-1_DropletData.txt").write_text(
                "DropletTime_sec\tDropletDiameter_mm\n-1\t3\n0\t2\n"
            )
            (folder / "FLEX-1_RadiometerData.txt").write_text(
                "RadiometerTime_sec\tTotalWideBandPower_W\tTotalNarrowBandPower_W\n0\t-100\t0.2\n"
            )
            (folder / "FLEX-2_RadiometerData.txt").write_bytes(b"%PDF-1.3\nnot a TSV")
            result, excluded = load_cfi(root)
            self.assertEqual(len(excluded), 1)
            self.assertIn("FLEX-2_RadiometerData.txt", excluded[0]["path"])
            squared = result.filter(pl.col("channel") == "droplet_diameter_squared")
            self.assertEqual(squared["value"].to_list(), [9., 4.])
            self.assertEqual(squared["time_s"].to_list(), [-1., 0.])
            power = result.filter(pl.col("channel") == "wideband_power")
            self.assertEqual(power["value"][0], -100.)
            self.assertTrue(power["negative_value"][0])

    def test_daft_rejects_truncated_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "DAFT-DAFT-2_PSI-47/txt"
            folder.mkdir(parents=True)
            (folder / "PTrk0001.txt").write_text("Number of points: 2\n1\n")
            with self.assertRaisesRegex(ValueError, "Point count mismatch"):
                load_daft(root)

    def test_cfi_reads_cr_lf_and_crlf_identically(self):
        expected = None
        for ending in (b"\n", b"\r", b"\r\n"):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = root / "CFI_PSI-39/txt"
                folder.mkdir(parents=True)
                files = {
                    "DropletData": [
                        b"DropletTime_sec\tDropletDiameter_mm", b"-1\t3", b"0\t2",
                    ],
                    "CoolFlameData": [
                        b"CoolFlameTime_s\tCoolFlameDiameterFWHM_mm\tCoolFlameDiameterP2P_mm",
                        b"0\t5\t4", b"1\t4\t3",
                    ],
                    "RadiometerData": [
                        b"RadiometerTime_sec\tTotalWideBandPower_W\tTotalNarrowBandPower_W",
                        b"0\t-100\t0.2", b"1\t-90\t0.3",
                    ],
                }
                for kind, lines in files.items():
                    (folder / f"FLEX-1_{kind}.txt").write_bytes(ending.join(lines) + ending)
                result, excluded = load_cfi(root)
                self.assertEqual(excluded, [])
                self.assertEqual(result.height, 12)
                records = result.to_dicts()
                if expected is None:
                    expected = records
                self.assertEqual(records, expected)

    def test_daft_preserves_extra_numeric_sample_and_removes_nul_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "DAFT-DAFT-2_PSI-47/txt"
            folder.mkdir(parents=True)
            (folder / "PTrk0001.txt").write_text(
                "Instrument: P-Trak\nTestID: 1\nUnits: particles/cc\n"
                "StartDate: 03/04/05\nStartTime: 14:36:50\nNumber of points: 2\n1\n2\n3\n\x00\x00"
            )
            result = load_daft(root)
            self.assertEqual(result["value"].to_list(), [1., 2., 3.])
            self.assertEqual(result["beyond_declared_count"].to_list(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
