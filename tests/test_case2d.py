import unittest

import numpy as np

from twoscan_ptycho.case2d import (
    build_calibrated_signature_model_2d,
    clean_lag_moments_2d,
    diffraction_intensity_2d,
    gaussian_probe_2d,
    recover_calibrated_case_2d,
    sparse_phase_object_2d,
    support_exact,
    theoretical_moments_2d,
)


class CaseStudy2DTests(unittest.TestCase):
    def test_two_dimensional_clean_lag_formula(self) -> None:
        n = 65
        gamma = 0.015
        positions = ((-9, -8), (0, 8), (8, -7))
        phases = (0.8, 0.8, 0.8)
        scan = (-2, -1)
        object_array, _ = sparse_phase_object_2d(n, positions, phases)
        probe = gaussian_probe_2d(n, gamma)
        intensity = diffraction_intensity_2d(object_array, probe, scan)
        measured = clean_lag_moments_2d(intensity, probe, gamma, 12)
        predicted = theoretical_moments_2d(
            positions,
            [np.exp(1j * phase) - 1.0 for phase in phases],
            scan,
            gamma,
            12,
            n,
        )
        relative_error = np.linalg.norm(measured - predicted) / np.linalg.norm(
            predicted
        )
        self.assertLess(relative_error, 1e-11)

    def test_calibrated_two_scan_case_recovers_exact_support(self) -> None:
        n = 65
        gamma = 0.015
        scans = ((-2, -1), (3, 2))
        positions = ((-9, -8), (0, 8), (8, -7))
        phase = 0.8
        object_array, _ = sparse_phase_object_2d(n, positions, [phase] * 3)
        probe = gaussian_probe_2d(n, gamma)
        intensities = tuple(
            diffraction_intensity_2d(object_array, probe, scan) for scan in scans
        )
        model = build_calibrated_signature_model_2d(
            scans,
            gamma,
            12,
            n,
            10,
            np.exp(1j * phase) - 1.0,
            background_gamma=gamma,
        )
        recovery = recover_calibrated_case_2d(intensities, model, 3)
        self.assertTrue(support_exact(positions, recovery.positions))
        self.assertTrue(np.allclose(recovery.strengths, 1.0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
