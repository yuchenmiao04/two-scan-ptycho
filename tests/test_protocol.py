"""Acquisition and paired-evaluation invariants, independent of solver rankings."""
import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.run_experiments import (
    SEEDS, calculate_metrics, formal_jobs, make_instance,
    validate_frozen_sources, fractional_frames, recover,
)


class ProtocolTests(unittest.TestCase):
    def test_complete_model_stress_grid_keeps_acquisition_namespace(self):
        jobs = formal_jobs("model_stress", 100, {})
        self.assertEqual(len(jobs), 5300)
        self.assertTrue(all(job["suite"] == "mismatch" for job in jobs))
        self.assertEqual({job["stress"] for job in jobs}, {
            "background", "probe", "position", "scale", "truncation", "quantization",
            "cluster", "boundary", "offgrid", "sparsity", "weak_defect",
        })
        self.assertEqual({job["namespace"] for job in jobs}, {SEEDS["formal"]})

    def test_offgrid_integer_limit_and_continuous_metric(self):
        from twoscan_ptycho.core import affine_object, diffraction_intensity, gaussian_probe
        probe = gaussian_probe(33, .05)
        positions, amplitudes = (-5, 4), (.3+.2j, -.4+.1j)
        expected = np.array([diffraction_intensity(affine_object(33, positions, amplitudes), probe, r)
                             for r in (0, 1)])
        np.testing.assert_allclose(fractional_frames(33, probe, (0, 1), positions, amplitudes),
                                   expected, atol=2e-13, rtol=2e-13)
        instance = make_instance(dict(suite="mismatch", namespace=SEEDS["formal"], trial=1,
                                      sparsity=2, snr="inf", stress="offgrid", value=.1))
        metrics = calculate_metrics(instance, instance["truth"], instance["positions"])
        self.assertIsNone(metrics["defect_nrmse"])
        self.assertIsNone(metrics["support_exact"])
        self.assertAlmostEqual(metrics["localization_rmse"], .1)

    def test_unknown_sparsity_uses_discrepancy_and_keeps_legal_truth(self):
        jobs = formal_jobs("unknown_sparsity", 1, {})
        self.assertEqual(len(jobs), 5)
        self.assertEqual({job["selection"] for job in jobs}, {"discrepancy"})
        instance = make_instance(jobs[-1])
        self.assertEqual(instance["upper"], 3)
        self.assertEqual(len(instance["positions"]), 2)
        self.assertGreater(instance["positions"][1] - instance["positions"][0], 6)
        _, support, _, diagnostics = recover("joint", instance, 1, {}, selection="discrepancy")
        self.assertEqual(set(support), set(instance["positions"]))
        self.assertEqual(diagnostics["selection"], "discrepancy")

    def test_beam_ablation_retains_separate_search_scope(self):
        jobs = formal_jobs("beam_ablation", 1, {})
        self.assertEqual(len(jobs), 20)
        self.assertEqual({job["beam_width"] for job in jobs}, {1, 8, 32, 128, 256})
        self.assertTrue(all(job["methods"] == ("joint", "beam") for job in jobs))
        instance = make_instance(dict(suite="beam_ablation", namespace=SEEDS["formal"],
                                      trial=0, sparsity=1, snr="inf"))
        _, _, status, diagnostics = recover("beam", instance, 1, {}, beam_width=1)
        self.assertEqual(status, "beam_heuristic")
        self.assertEqual(diagnostics["comparison_scope"], "beam_ablation_with_distinct_search_space")

    def test_pairing_does_not_depend_on_method_or_noise_level(self):
        base=dict(suite="gaussian",namespace=SEEDS["formal"],trial=9,sparsity=3,snr=20)
        a=make_instance(base)
        b=make_instance(dict(base,methods=("pie",)))
        c=make_instance(dict(base,snr=40))
        np.testing.assert_array_equal(a["measured"],b["measured"])
        np.testing.assert_array_equal(a["truth"],c["truth"])
        np.testing.assert_allclose((a["measured"]-a["clean"])/10,c["measured"]-c["clean"],atol=1e-14)

    def test_effective_snr_uses_defect_intensity_not_blank_field(self):
        instance=make_instance(dict(suite="gaussian",namespace=SEEDS["formal"],trial=4,sparsity=2,snr=30))
        blank=np.abs(np.fft.fft(instance["probe"]))**2
        expected=np.linalg.norm(instance["clean"]-blank)/np.sqrt(instance["clean"].size)*10**(-30/20)
        self.assertAlmostEqual(instance["sigma"],expected)
        self.assertGreater(instance["total_snr"],instance["effective_snr"])

    def test_validation_and_formal_seeds_are_disjoint(self):
        self.assertEqual(len(set(SEEDS.values())),3)
        jobs=formal_jobs("gaussian",200,{})
        self.assertEqual(len(jobs),3*8*200)
        self.assertTrue(all(j["namespace"]==SEEDS["formal"] for j in jobs))

    def test_only_reported_separation_and_background_conditions_are_generated(self):
        separation = formal_jobs("separation", 100, {})
        background = formal_jobs("background", 100, {})
        self.assertEqual(len(separation), 500)
        self.assertEqual(len(background), 200)
        self.assertEqual({job["value"] for job in separation}, {2, 4, 5, 6, 8})
        self.assertEqual({job["value"] for job in background}, {0, .001})
        self.assertTrue(all(job["suite"] == "mismatch" for job in separation + background))
        self.assertTrue(all(job["stress"] == "cluster" for job in separation))
        for job in separation[::100]:
            instance = make_instance(job)
            self.assertEqual(instance["spacing"], 0)
            self.assertEqual(instance["positions"][1] - instance["positions"][0], job["value"])

    def test_frozen_guard_requires_exact_solver_source_set_and_hashes(self):
        current = {"src/twoscan_ptycho/core.py": "abc", "experiments/run_experiments.py": "def"}
        with patch("experiments.run_experiments.source_hashes", return_value=current):
            validate_frozen_sources({"distribution_source_hashes": current})
            for recorded in (None, {}, {"src/twoscan_ptycho/core.py": "changed"},
                             {"experiments/run_experiments.py": "def"},
                             {**current, "src/extra.py": "ghi"}):
                with self.subTest(recorded=recorded), self.assertRaises(RuntimeError):
                    validate_frozen_sources({"distribution_source_hashes": recorded})


if __name__=="__main__":
    unittest.main()
