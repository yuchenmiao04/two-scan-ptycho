"""Algebraic and small-grid exhaustive tests; no recovery-rate experiment."""

import inspect
import unittest

import numpy as np

from twoscan_ptycho.core import (
    affine_object,
    centered_grid,
    diffraction_intensity,
    gaussian_probe,
    position_to_index,
    sparse_perturbation,
)
from twoscan_ptycho.reconstruction import (
    GlobalSupportSolver,
    SpectralConfig,
    build_joint_dictionary,
    extract_raw_data,
    recover_reciprocal_moments,
    recover_two_scan_spectral,
)


def reciprocal_moments(nodes, coefficients, bound, center=0.0):
    ell = np.arange(1, 2 * bound + 1)
    y = np.arccosh(np.asarray(nodes, dtype=float))
    values = np.asarray(coefficients, dtype=complex)
    result = np.full(2 * bound, center, dtype=complex)
    if len(nodes):
        result += np.einsum("ij,j->i",np.exp(ell[:, None]*y),values,optimize=False)
        result += np.einsum("ij,j->i",np.exp(-ell[:, None]*y),values.conj(),optimize=False)
    return result


class SpectralReconstructionTests(unittest.TestCase):
    def assert_measure(self, nodes, values, bound, center=0.0, tolerance=2e-6):
        moments = reciprocal_moments(nodes, values, bound, center)
        recovery = recover_reciprocal_moments(moments, bound)
        self.assertTrue(recovery.success, recovery.status + repr(recovery.diagnostics))
        np.testing.assert_allclose(recovery.nodes, nodes, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(recovery.coefficients, values, rtol=tolerance, atol=tolerance)
        self.assertAlmostEqual(recovery.center_coefficient, center, delta=tolerance)
        np.testing.assert_allclose(recovery.reconstructed_moments, moments, rtol=2e-7, atol=2e-7)
        return recovery

    def test_mixed_complex_coefficients(self):
        self.assert_measure([1.2, 1.8], [.2 + .3j, -.4 + .1j], 2)

    def test_pure_real_and_pure_imaginary_components(self):
        self.assert_measure([1.2, 1.8], [.2, -.4], 2)
        self.assert_measure([1.2, 1.8], [.3j, .1j], 2)

    def test_real_and_imaginary_supports_need_not_agree(self):
        self.assert_measure([1.15, 1.55, 2.0], [.3, -.2j, .4 + .1j], 3, tolerance=1e-5)

    def test_missing_real_mass_can_be_zero_with_nonzero_measure(self):
        result = self.assert_measure([1.2, 1.9], [.5, -.5], 2)
        self.assertLess(np.min(np.abs(result.mass_candidates)), 1e-9)

    def test_signed_cancellation_does_not_make_full_hankel_rank_zero(self):
        nodes = np.array([1.2, 1.9])
        imaginary_mass = np.array([1.0, -1.0])
        beta = imaginary_mass / (2 * np.sqrt(nodes * nodes - 1))
        result = self.assert_measure(nodes, 1j * beta, 2)
        self.assertEqual(result.diagnostics["imaginary"]["rank"], 2)

    def test_center_and_upper_bound_are_supported(self):
        self.assert_measure([1.2], [.2 + .3j], 3, center=.4)
        self.assert_measure([], [], 3, center=-.4)

    def test_zero_measure(self):
        result = recover_reciprocal_moments(np.zeros(6, complex), 3)
        self.assertEqual(result.status, "ok_zero")
        self.assertEqual(len(result.nodes), 0)

    def test_failure_is_exposed_not_replaced_by_a_support_guess(self):
        invalid = recover_reciprocal_moments([1 + 2j, 2 - 5j, -3 + 8j, 4 + 1j], 2)
        self.assertFalse(invalid.success)
        self.assertTrue(invalid.status)
        self.assertEqual(recover_reciprocal_moments([np.nan, 1], 1).status, "invalid_input")
        with self.assertRaises(ValueError):
            SpectralConfig(rank_rtol=-1)

    def test_end_to_end_polynomial_decoder(self):
        n, gamma, scans = 97, .04, (0, 1)
        positions = (-5, 4)
        values = (.3 + .2j, -.4 + .1j)
        probe = gaussian_probe(n, gamma)
        z = affine_object(n, positions, values)
        frames = [diffraction_intensity(z, probe, r) for r in scans]
        result = recover_two_scan_spectral(frames, probe, gamma, scans, 2)
        self.assertTrue(result.success, result.status + repr(result.diagnostics))
        self.assertEqual(result.support, positions)
        np.testing.assert_allclose(result.u, z - 1, atol=2e-5, rtol=2e-5)

    def test_end_to_end_center_imaginary_and_zero(self):
        n, gamma, scans = 65, .06, (0, 1)
        probe = gaussian_probe(n, gamma)
        for positions, values in [((0,), (.7j,)), ((), ())]:
            with self.subTest(positions=positions):
                z = affine_object(n, positions, values)
                frames = [diffraction_intensity(z, probe, r) for r in scans]
                result = recover_two_scan_spectral(frames, probe, gamma, scans, 1)
                self.assertTrue(result.success, result.status + repr(result.diagnostics))
                self.assertEqual(result.support, positions)
                np.testing.assert_allclose(result.u, z - 1, atol=2e-6, rtol=2e-6)


class JointReconstructionTests(unittest.TestCase):
    def frames(self, model, positions, values):
        z = affine_object(model.n, positions, values)
        frames = np.array([diffraction_intensity(z, model.probes[i], scan) for i, scan in enumerate(model.scans)])
        return z, frames

    def coefficient_vector(self, model, positions, values):
        result = np.zeros(2 * len(model.candidates))
        for position, value in zip(positions, values):
            index = model.candidates.index(position)
            result[index] = complex(value).real
            result[len(model.candidates) + index] = complex(value).imag
        return result

    def test_exact_raw_dictionary_matches_core_frames(self):
        model = build_joint_dictionary(97, .025, (0, 1), range(-12, 13), 3)
        positions, values = (-9, 0, 9), (.3 + .2j, -.4 + .1j, .2 - .5j)
        _, frames = self.frames(model, positions, values)
        raw = extract_raw_data(frames, model)
        predicted = np.einsum("ij,j->i",model.matrix,self.coefficient_vector(model,positions,values),optimize=False)
        np.testing.assert_allclose(predicted, raw, atol=5e-14, rtol=5e-13)

    def test_complex_probe_realification_is_exact(self):
        n, gamma = 65, .03
        q = centered_grid(n)
        probe = np.exp(-(gamma + .17j) * q * q)
        model = build_joint_dictionary(n, gamma, (0, 2), range(-9, 10), 2, probe=probe)
        positions, values = (-5, 4), (.2 + .4j, -.3 + .7j)
        _, frames = self.frames(model, positions, values)
        predicted = np.einsum("ij,j->i",model.matrix,self.coefficient_vector(model,positions,values),optimize=False)
        np.testing.assert_allclose(predicted,extract_raw_data(frames,model),atol=5e-14,rtol=5e-13)

    def test_noiseless_one_two_and_three_complex_defects(self):
        cases = [
            (1, (-3,), (.7j,)),
            (2, (-5, 4), (.3 + .2j, -.4 + .1j)),
            (3, (-10, 0, 10), (.3 + .2j, -.4 + .1j, .2 - .5j)),
        ]
        for bound, positions, values in cases:
            with self.subTest(bound=bound):
                model = build_joint_dictionary(257, .025, (0, 1), range(-13, 14), bound)
                solver = GlobalSupportSolver(model, bound)
                z, frames = self.frames(model, positions, values)
                result = solver.solve_intensities(frames)
                self.assertEqual(result.support, positions, result.status)
                np.testing.assert_allclose(result.u, z - 1, atol=5e-9, rtol=5e-9)
                self.assertEqual(len(result.all_residuals), result.candidate_count)
                self.assertTrue(all(value >= 0 for value in result.timings.values()))

    def test_reflection_cancellation_does_not_break_joint_fit(self):
        model = build_joint_dictionary(97, .025, (0, 1), range(-10, 11), 2)
        solver = GlobalSupportSolver(model, 2)
        positions = (-5, 5)
        value = .3 + .4j
        values = (value, -value.conjugate())
        z, frames = self.frames(model, positions, values)
        raw = extract_raw_data(frames, model)
        self.assertLess(np.linalg.norm(raw[:8]), 1e-12)
        result = solver.solve(raw)
        self.assertEqual(result.support, positions)
        np.testing.assert_allclose(result.u, z - 1, atol=5e-10, rtol=5e-10)

    def test_sparsity_upper_bound_zero_and_lower_actual_sparsity(self):
        model = build_joint_dictionary(97, .04, (0, 1), range(-10, 11), 3)
        solver = GlobalSupportSolver(model, 3)
        for positions, values in [((), ()), ((2,), (.3 + .4j,))]:
            z, frames = self.frames(model, positions, values)
            result = solver.solve_intensities(frames)
            self.assertEqual(result.support, positions)
            np.testing.assert_allclose(result.u, z - 1, atol=1e-10, rtol=1e-10)

    def test_every_cached_svd_fit_matches_direct_lstsq(self):
        model = build_joint_dictionary(61, .04, (0, 1), range(-6, 7), 2)
        solver = GlobalSupportSolver(model, 2)
        data = np.random.default_rng(8).normal(size=model.matrix.shape[0])
        result = solver.solve(data)
        count = len(model.candidates)
        for support, reported in zip(result.all_supports, result.all_residuals):
            indices = [model.candidates.index(t) for t in support]
            columns = indices + [j + count for j in indices]
            design = model.matrix[:, columns]
            coefficients = np.linalg.lstsq(design, data, rcond=None)[0]
            expected = np.linalg.norm(np.einsum("ij,j->i",design,coefficients,optimize=False)-data)
            self.assertAlmostEqual(reported, expected, delta=2e-11)

    def test_spacing_is_strict_and_circular(self):
        model = build_joint_dictionary(31, .04, (0, 1), (-15, -11, 0, 11, 15), 2, enforce_margin=False)
        solver = GlobalSupportSolver(model, 2)
        self.assertNotIn((-15, -11), solver.all_supports)  # equality to L is excluded
        self.assertNotIn((-15, 15), solver.all_supports)   # circular distance one
        self.assertIn((-11, 0), solver.all_supports)

    def test_invalid_margin_and_noninteger_candidates_are_not_silently_removed(self):
        with self.assertRaises(ValueError):
            build_joint_dictionary(31, .04, (0, 1), (-15, 0), 2)
        with self.assertRaises(ValueError):
            build_joint_dictionary(31, .04, (0, 1), (0.5, 2), 1)
        model = build_joint_dictionary(61, .04, (0, 1), range(-10, 11), 2)
        with self.assertRaises(ValueError):
            GlobalSupportSolver(model, 2, max_supports=5)

    def test_residual_gap_diagnostic_and_discrepancy_rule(self):
        model = build_joint_dictionary(97, .04, (0, 1), range(-9, 10), 2)
        solver = GlobalSupportSolver(model, 2)
        _, frames = self.frames(model, (-5, 4), (.3 + .2j, -.4 + .1j))
        result = solver.solve_intensities(frames, noise_bound=1e-10)
        self.assertTrue(result.certificate["available"])
        self.assertTrue(result.certificate["selected_support_required_by_residuals"])
        self.assertFalse(result.certificate["is_formal_interval_certificate"])
        selected = solver.solve_intensities(frames, noise_bound=1e-10, selection="discrepancy")
        self.assertEqual(selected.support, result.support)
        self.assertEqual(selected.status, "ok_discrepancy")
        failed = solver.solve(np.random.default_rng(5).normal(size=model.matrix.shape[0]), noise_bound=0, selection="discrepancy")
        self.assertEqual(failed.status, "noise_bound_infeasible")

    def test_regularization_does_not_claim_unregularized_gap_certificate(self):
        model = build_joint_dictionary(61, .04, (0, 1), range(-7, 8), 2)
        data = np.random.default_rng(8).normal(size=model.matrix.shape[0])
        ridge = GlobalSupportSolver(model, 2, ridge=.01).solve(data, noise_bound=10)
        self.assertEqual(ridge.status, "ok_global_ridge")
        self.assertFalse(ridge.certificate["available"])
        truncated = GlobalSupportSolver(model, 2, rcond=.9).solve(data, noise_bound=10)
        self.assertIn("rank_truncated", truncated.status)
        self.assertFalse(truncated.certificate["available"])

    def test_no_truth_support_argument_exists(self):
        for function in (recover_reciprocal_moments, recover_two_scan_spectral, GlobalSupportSolver.solve, GlobalSupportSolver.solve_intensities):
            self.assertFalse(any("truth" in name for name in inspect.signature(function).parameters))


if __name__ == "__main__":
    unittest.main()
