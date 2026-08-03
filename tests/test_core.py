import unittest

import numpy as np

from twoscan_ptycho.full_data import recover_full_data_linearized

from twoscan_ptycho.core import (
    affine_object,
    centered_reflection_sum_from_orbits,
    clean_lag_moments,
    conjugate_reflection,
    diffraction_intensity,
    gaussian_probe,
    intensity_autocorrelation,
    recover_from_two_reflection_sums,
    recover_orbit_coefficients_beam,
    recover_orbit_coefficients_group_omp,
    reflect_sparse_object,
    reflection_sum,
    sparse_perturbation,
    split_null_counterexample,
    theoretical_moments,
)


class SparsePtychoTests(unittest.TestCase):
    def test_fft_intensity_inverts_to_direct_circular_autocorrelation(self):
        rng = np.random.default_rng(1)
        h = rng.normal(size=15) + 1j * rng.normal(size=15)
        intensity = np.abs(np.fft.fft(h)) ** 2
        recovered = intensity_autocorrelation(intensity)
        direct = np.array(
            [sum(h[q] * np.conj(h[(q - ell) % len(h)]) for q in range(len(h))) for ell in range(len(h))]
        )
        np.testing.assert_allclose(recovered, direct, rtol=1e-12, atol=1e-12)

    def test_two_n_clean_lags_match_theoretical_formula(self):
        n = 61
        positions = (-12, 1, 14)
        amplitudes = (0.4 + 0.3j, -0.2 + 0.8j, 0.7 - 0.4j)
        scan = 2
        gamma = 0.006
        lag_count = 2 * len(positions)
        probe = gaussian_probe(n, gamma)
        z = affine_object(n, positions, amplitudes)
        intensity = diffraction_intensity(z, probe, scan)
        measured = clean_lag_moments(intensity, probe, gamma, lag_count)
        expected = theoretical_moments(
            positions, amplitudes, scan, gamma, lag_count, n
        )
        np.testing.assert_allclose(measured, expected, rtol=1e-10, atol=1e-10)

    def test_one_scan_conjugate_reflection_ambiguity(self):
        n = 61
        positions = (-11, 3, 16)
        amplitudes = (0.2 + 0.5j, -0.7 + 0.3j, 0.6 - 0.9j)
        scan = 2
        probe = gaussian_probe(n, 0.005)
        reflected_positions, reflected_amplitudes = reflect_sparse_object(
            n, positions, amplitudes, scan
        )
        y1 = diffraction_intensity(
            affine_object(n, positions, amplitudes), probe, scan
        )
        y2 = diffraction_intensity(
            affine_object(n, reflected_positions, reflected_amplitudes), probe, scan
        )
        np.testing.assert_allclose(y1, y2, rtol=1e-12, atol=1e-12)

    def test_beam_search_recovers_structured_moments(self):
        gamma = 0.025
        lag_count = 4
        coefficients = {2: 0.8 - 0.5j, 6: -0.4 + 0.7j}
        ell = np.arange(1, lag_count + 1, dtype=float)
        moments = sum(
            b * np.exp(2 * gamma * x * ell)
            + np.conj(b) * np.exp(-2 * gamma * x * ell)
            for x, b in coefficients.items()
        )
        recovered = recover_orbit_coefficients_beam(
            moments,
            gamma,
            max_offset=7,
            sparsity=2,
            beam_width=64,
            row_equilibrate=True,
        )
        self.assertEqual(set(recovered.support), set(coefficients))
        for x, value in coefficients.items():
            self.assertAlmostEqual(recovered.coefficients[x].real, value.real, places=8)
            self.assertAlmostEqual(recovered.coefficients[x].imag, value.imag, places=8)

    def test_group_omp_is_available_as_a_baseline(self):
        gamma = 0.025
        ell = np.arange(1, 5, dtype=float)
        moments = 0.8 * np.exp(2 * gamma * 2 * ell) + 0.8 * np.exp(-2 * gamma * 2 * ell)
        recovered = recover_orbit_coefficients_group_omp(
            moments, gamma, max_offset=7, sparsity=1
        )
        self.assertEqual(len(recovered.support), 1)

    def test_two_reflection_integration_recovers_sparse_object(self):
        n = 31
        positions = (-8, 1, 10)
        amplitudes = (0.3 + 0.8j, -0.5 + 0.2j, 0.9 - 0.4j)
        mu = sparse_perturbation(n, positions, amplitudes)
        p1 = reflection_sum(mu, 0)
        p2 = reflection_sum(mu, 1)
        recovered = recover_from_two_reflection_sums(p1, p2, 0, 1, len(positions))
        np.testing.assert_allclose(recovered.perturbation, mu, rtol=1e-11, atol=1e-11)

    def test_centered_orbit_deweighting(self):
        n = 31
        gamma = 0.01
        coefficients = {0: 0.4 + 0j, 3: 0.2 - 0.7j}
        q = centered_reflection_sum_from_orbits(n, gamma, coefficients)
        center = (n - 1) // 2
        self.assertAlmostEqual(q[center].real, 0.4)
        self.assertAlmostEqual(q[center + 3], np.exp(0.18) * (0.2 - 0.7j))
        self.assertAlmostEqual(q[center - 3], np.exp(0.18) * (0.2 + 0.7j))
        np.testing.assert_allclose(q, conjugate_reflection(q, 0))

    def test_two_n_minus_one_split_null_counterexample(self):
        result = split_null_counterexample(gamma=0.01, sparsity=3)
        self.assertLess(result["relative_residual"], 1e-9)

    def test_full_data_linearized_baseline_recovers_small_object(self):
        n = 61
        positions = (-9, 10)
        amplitudes = (0.3 + 0.4j, -0.5 + 0.2j)
        scans = (0, 1)
        probe = gaussian_probe(n, 0.008)
        z = affine_object(n, positions, amplitudes)
        intensities = np.stack(
            [diffraction_intensity(z, probe, scan) for scan in scans]
        )
        recovered = recover_full_data_linearized(
            intensities,
            probe,
            scans,
            sparsity=2,
            candidate_positions=range(-14, 15),
            beam_width=512,
            debias_iterations=4,
            physical_phase_projection=False,
        )
        self.assertEqual(set(recovered.positions), set(positions))
        np.testing.assert_allclose(
            recovered.perturbation,
            sparse_perturbation(n, positions, amplitudes),
            rtol=1e-7,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
