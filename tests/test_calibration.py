"""Public acquisition and CLI contracts for the independent calibration study."""
from __future__ import annotations

import sys
import unittest
import csv
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments import run_calibration as calibration
from experiments.run_calibration import calibrate_blank
from twoscan_ptycho.reconstruction import build_joint_dictionary, extract_raw_data
import numpy as np


class PrescribedCounts:
    """Deterministic randomness boundary, with deliberately unequal count sums."""

    def poisson(self, mean):
        np.testing.assert_array_equal(mean, [[2.0, 2.0], [2.0, 2.0]])
        return np.array([[1, 2], [0, 6]])


class CalibrationAcquisitionTests(unittest.TestCase):
    def test_blank_uses_known_exposure_gain_not_observed_count_normalization(self):
        result = calibrate_blank(np.full((2, 2), 2.0), 8.0, PrescribedCounts())
        np.testing.assert_array_equal(result["intensities"], [[1.0, 2.0], [0.0, 6.0]])
        np.testing.assert_array_equal(result["autocorrelations"], [[1.5, -0.5], [3.0, -3.0]])
        np.testing.assert_array_equal(result["exposure_gain"], [1.0, 1.0])
        self.assertEqual(result["photons_per_frame"], 4.0)
        self.assertEqual(result["total_expected_photons"], 8.0)

    def test_exact_blank_matches_existing_raw_data_and_does_not_mutate_measurements(self):
        model = build_joint_dictionary(31, 0.05, (0, 1), range(-5, 6), 2)
        frames = model.background_intensities + np.arange(62).reshape(2, 31) / 100
        before = frames.copy()
        exact_cw = np.fft.ifft(model.background_intensities, axis=1)
        raw = calibration.raw_lags_with_blank_autocorrelation(frames, model, exact_cw)
        np.testing.assert_allclose(raw, extract_raw_data(frames, model), atol=1e-14)
        np.testing.assert_array_equal(frames, before)
        offset_cw = exact_cw.copy()
        offset_cw[0, 1] += 0.25 + 0.5j
        shifted = calibration.raw_lags_with_blank_autocorrelation(frames, model, offset_cw)
        expected_shift = np.zeros(16)
        expected_shift[0], expected_shift[4] = -0.25, -0.5
        np.testing.assert_allclose(shifted - raw, expected_shift, atol=1e-14)

    def test_blank_cli_pairs_objects_and_measurements_and_retains_exact_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "blank"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "experiments/run_calibration.py"),
                 "blank", "--output", str(output), "--trials", "2",
                 "--snrs", "40", "inf", "--calibration-photons", "1000000"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (output / "trials.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            for trial in ("0", "1"):
                paired = [row for row in rows if row["trial"] == trial]
                self.assertEqual(len({row["truth_sha256"] for row in paired}), 1)
                for snr in ("40", "inf"):
                    same_measurement = [row for row in paired if row["snr"] == snr]
                    self.assertEqual(len({row["measurement_sha256"] for row in same_measurement}), 1)
                    exact = next(row for row in same_measurement if row["calibration"] == "exact")
                    noisy = next(row for row in same_measurement if row["calibration"] == "poisson")
                    self.assertEqual(float(exact["blank_lag_error_norm"]), 0.0)
                    self.assertLess(float(exact["exact_control_existing_api_difference"]), 1e-12)
                    self.assertEqual(float(noisy["calibration_photons_per_frame"]), 500000.0)
                    self.assertGreater(float(noisy["blank_lag_error_norm"]), 0.0)
                    if snr == "inf":
                        self.assertEqual(float(exact["measurement_raw_lag_noise_norm"]), 0.0)
                        self.assertEqual(exact["support_exact"], "1")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["settings"]["scope"], "joint_only_blank_calibration")
            self.assertFalse(manifest["settings"]["observed_count_total_renormalization"])
            self.assertEqual(len(list((output / "measurements").glob("*.npz"))), 4)
            self.assertEqual(len(list((output / "calibrations").glob("*.npz"))), 2)

    def test_timing_setup_cli_separates_cold_creation_from_online_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "timing"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "experiments/run_calibration.py"),
                 "timing_setup", "--output", str(output), "--trials", "2",
                 "--sparsities", "1"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (output / "trials.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["legal_support_count"], "22")
                self.assertGreater(float(row["dictionary_build_seconds"]), 0)
                self.assertGreater(float(row["solver_precompute_seconds"]), 0)
                self.assertGreater(float(row["cached_online_seconds"]), 0)
                self.assertGreaterEqual(float(row["cold_setup_seconds"]),
                                        float(row["dictionary_build_seconds"]))
                self.assertAlmostEqual(
                    float(row["total_cold_reconstruction_seconds"]),
                    float(row["cold_setup_seconds"]) + float(row["cached_online_seconds"]),
                    places=12,
                )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["settings"]["setup_cache_disabled"])
            self.assertEqual(manifest["settings"]["warmup_per_sparsity"], 1)
            self.assertEqual(manifest["workers"], 1)
            self.assertFalse(manifest["settings"]["concurrency_isolation_user_confirmed"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["aggregate"][0]["cold_setup_seconds"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
