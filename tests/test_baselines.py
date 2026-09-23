import inspect
import unittest

import numpy as np

from twoscan_ptycho.core import (
    affine_object, circular_distance, diffraction_intensity,
    gaussian_probe, position_to_index, sparse_perturbation,
)
from twoscan_ptycho.baselines import (
    BaselineConfig, _SupportProjection, _amplitude_loss_gradient,
    _forward_design, _intensity_loss_gradient, recover_baseline,
)


class BaselineTests(unittest.TestCase):
    def test_integer_forward_matches_core_with_complex_frame_probes(self):
        n, scans, candidates = 31, (-2, 1, 4), (-7, 0, 8)
        rng = np.random.default_rng(9)
        probes = gaussian_probe(n, .015)[None, :] * np.exp(1j * rng.normal(size=(3, n)))
        coefficients = np.array([.2+.7j, -.4+.1j, 1.5-.3j])
        design, reference = _forward_design(probes, scans, candidates)
        predicted = np.abs(reference + design @ coefficients).reshape(3, n)**2
        z = affine_object(n, candidates, coefficients)
        truth = np.array([diffraction_intensity(z, p, r) for p, r in zip(probes, scans)])
        np.testing.assert_allclose(predicted, truth, rtol=2e-13, atol=2e-13)

    def test_amplitude_gradient_directional_finite_difference(self):
        rng = np.random.default_rng(7)
        design = rng.normal(size=(19, 4)) + 1j*rng.normal(size=(19, 4))
        reference = rng.normal(size=19) + 1j*rng.normal(size=19)
        coefficients = rng.normal(size=4) + 1j*rng.normal(size=4)
        target = rng.uniform(.2, 3, size=19)
        _, gradient = _amplitude_loss_gradient(coefficients, design, reference, target)
        for direction in (rng.normal(size=4), 1j*rng.normal(size=4),
                          rng.normal(size=4)+1j*rng.normal(size=4)):
            step = 1e-6
            plus, _ = _amplitude_loss_gradient(coefficients+step*direction, design, reference, target)
            minus, _ = _amplitude_loss_gradient(coefficients-step*direction, design, reference, target)
            numerical = (plus[0]-minus[0])/(2*step)
            analytic = np.vdot(gradient[0], direction).real
            self.assertAlmostEqual(numerical, analytic, delta=1e-7)

    def test_projection_maximizes_legal_retained_energy_and_handles_wrap(self):
        projector = _SupportProjection(15, (-7, -3, 0, 3, 7), 2, 3, 1000)
        values = np.array([[10+1j, 2, 7j, 4, 9]])
        result = projector.project(values)[0]
        expected = max(sum(abs(values[0, j])**2 for j in support)
                       for support in projector.supports)
        self.assertAlmostEqual(float(np.linalg.norm(result)**2), expected)
        self.assertNotIn((0, 4), projector.supports)  # cyclic distance one
        self.assertEqual(np.count_nonzero(result), 2)

    def test_intensity_gradient_directional_finite_difference(self):
        rng = np.random.default_rng(21)
        design = rng.normal(size=(17, 4)) + 1j*rng.normal(size=(17, 4))
        reference = rng.normal(size=17) + 1j*rng.normal(size=17)
        coefficients = .2*(rng.normal(size=4) + 1j*rng.normal(size=4))
        target = rng.uniform(0, 4, size=17)
        _, gradient = _intensity_loss_gradient(coefficients, design, reference, target)
        for direction in (rng.normal(size=4), 1j*rng.normal(size=4),
                          rng.normal(size=4)+1j*rng.normal(size=4)):
            step = 1e-6
            plus, _ = _intensity_loss_gradient(coefficients+step*direction, design, reference, target)
            minus, _ = _intensity_loss_gradient(coefficients-step*direction, design, reference, target)
            numerical = (plus[0]-minus[0])/(2*step)
            analytic = np.vdot(gradient[0], direction).real
            self.assertAlmostEqual(numerical, analytic, delta=2e-7)

    def test_general_complex_fixed_point_and_constraints_all_methods(self):
        n, scans, candidates, sparsity = 41, (0, 1), tuple(range(-7, 8)), 2
        probe = gaussian_probe(n, .02)
        initial = sparse_perturbation(n, (-5, 4), (1.7+.8j, -.3-.6j))
        data = np.array([diffraction_intensity(1+initial, probe, r) for r in scans])
        config = BaselineConfig(iterations=12, restarts=1, patience=3)
        for method in ("pie", "htwf", "htaf", "full_linear"):
            result = recover_baseline(method, data, probe, scans, candidates,
                sparsity, 4, seed=3, config=config, initial_u=initial)
            self.assertTrue(result.status.startswith("ok"))
            self.assertEqual(result.support, (-5, 4))
            np.testing.assert_allclose(result.u, initial, atol=1e-10)
            self.assertGreater(abs(1+result.u[position_to_index(-5, n)]), 2)
            self.assertGreaterEqual(result.runtime_seconds, result.diagnostics["setup_seconds"])
            self.assertAlmostEqual(result.runtime_seconds, result.diagnostics["setup_seconds"]
                                   + result.diagnostics["optimization_seconds"])
            self.assertLessEqual(result.diagnostics["linear_svd_setup_seconds"],
                                 result.diagnostics["setup_seconds"])

    def test_unknown_support_output_constraints_reproducibility_and_no_truth_api(self):
        n, scans, candidates = 41, (0, 1), tuple(range(-7, 8))
        probe = gaussian_probe(n, .02)
        data = np.array([diffraction_intensity(affine_object(n, (-5, 4), (.2+.3j, -.4+.1j)), probe, r)
                         for r in scans])
        config = BaselineConfig(iterations=10, restarts=2, patience=3)
        self.assertFalse({"truth", "true_support", "true_u"} & set(inspect.signature(recover_baseline).parameters))
        for method in ("pie", "htwf", "htaf", "full_linear"):
            result = recover_baseline(method, data, probe, scans, candidates, 2, 4, seed=5, config=config)
            again = recover_baseline(method, data, probe, scans, candidates, 2, 4, seed=5, config=config)
            np.testing.assert_allclose(result.u, again.u, atol=0, rtol=0)
            self.assertLessEqual(len(result.support), 2)
            self.assertTrue(set(result.support).issubset(candidates))
            self.assertTrue(all(circular_distance(a, b, n)>4 for i, a in enumerate(result.support)
                                for b in result.support[i+1:]))
            self.assertTrue(np.isfinite(result.diagnostics["amplitude_residual"]))
            self.assertTrue(np.isfinite(result.diagnostics["intensity_residual"]))
            expected_objective = "intensity_squared_error" if method == "htwf" else "amplitude_squared_error"
            self.assertEqual(result.diagnostics["selection_objective"], expected_objective)

    def test_full_linear_debias_recovers_small_general_complex_defect(self):
        n, scans, candidates = 31, (0, 1), tuple(range(-5, 6))
        probe = gaussian_probe(n, .04)
        expected = sparse_perturbation(n, (3,), (.13+.21j,))
        data = np.array([diffraction_intensity(1+expected, probe, r) for r in scans])
        result = recover_baseline("full_linear", data, probe, scans, candidates, 1, 2,
            seed=2, config=BaselineConfig(iterations=30, restarts=3, tolerance=1e-10))
        self.assertEqual(result.support, (3,))
        np.testing.assert_allclose(result.u, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
