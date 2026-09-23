"""Small CLI checks of paired inputs, source integrity and output protection."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def run_cli(self, module, output, *arguments):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run(
            [sys.executable, "-m", f"experiments.{module}", "--output", str(output), *arguments],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_validation_has_two_methods_and_preserves_seed_formula(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "validation"
            self.run_cli("run_validation", output, "--trials", "1")
            with (output / "trials.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["method"] for row in rows}, {"joint_global", "spectral_algebraic"})
            for row in rows:
                self.assertEqual(int(row["seed"]), 20260920 + 10_000_019 + int(row["N"]) * 100_003)
            with np.load(output / "raw_data.npz") as archive:
                self.assertEqual(len(archive.files), 9)
                self.assertEqual(archive["N3_clean_intensities"].shape, (1, 2, 257))

    def test_case_smoke_uses_full_budget_indices_and_paired_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "case2d"
            self.run_cli("run_case2d", output, "--smoke", "--workers", "1")
            output = output / "smoke"
            with (output / "trials.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual({row["method"] for row in rows}, {"calibrated", "gamma_plus_2pct"})
            with np.load(output / "measured_intensities.npz") as archive:
                self.assertEqual(archive["intensities"].shape, (4, 2, 65, 65))
                np.testing.assert_array_equal(
                    archive["seeds"],
                    [20260920 + 30_000_041 + index * 100_003 + trial * 1_009
                     for index in (0, 3) for trial in (0, 1)],
                )
            self.assertEqual({p.name for p in output.iterdir()},
                             {"trials.csv", "summary.json", "manifest.json", "measured_intensities.npz"})

    def test_setup_cli_times_fresh_objects_and_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "timing_setup"
            self.run_cli("run_timing_setup", output, "--trials", "2", "--sparsities", "1")
            with (output / "trials.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertGreater(float(row["dictionary_build_seconds"]), 0)
                self.assertGreater(float(row["solver_precompute_seconds"]), 0)
                self.assertAlmostEqual(
                    float(row["total_cold_reconstruction_seconds"]),
                    float(row["cold_setup_seconds"]) + float(row["cached_online_seconds"]), places=12,
                )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["settings"]["setup_cache_disabled"])
            self.assertFalse(manifest["settings"]["concurrency_isolation_user_confirmed"])
            original = (output / "trials.csv").read_bytes()
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            result = subprocess.run(
                [sys.executable, "-m", "experiments.run_timing_setup", "--output", str(output),
                 "--trials", "1", "--sparsities", "1"], cwd=ROOT, env=env, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output / "trials.csv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
