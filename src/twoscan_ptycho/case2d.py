"""Two-dimensional sparse-defect case study for the SPL manuscript.

The object is a uniform substrate with a few defects having a calibrated phase
signature.  Recovery uses the positive clean-lag box from the tensor-product
corollary and a joint two-scan sparse matched filter followed by discrete support
refinement.  This grid-constrained decoder is considerably more stable than using
an unregularized multidimensional matrix pencil in noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np

from .core import centered_grid, position_to_index, wrap_centered


Array = np.ndarray
Point = tuple[int, int]


def gaussian_probe_2d(n: int, gamma: float) -> Array:
    q = centered_grid(n).astype(float)
    q1, q2 = np.meshgrid(q, q, indexing="ij")
    return np.exp(-gamma * (q1 * q1 + q2 * q2))


def sparse_phase_object_2d(
    n: int, positions: Sequence[Point], phases: Sequence[float]
) -> tuple[Array, Array]:
    if len(positions) != len(phases):
        raise ValueError("positions and phases must have equal length")
    u = np.zeros((n, n), dtype=np.complex128)
    for (p1, p2), phase in zip(positions, phases):
        i = position_to_index(p1, n)
        j = position_to_index(p2, n)
        u[i, j] = np.exp(1j * float(phase)) - 1.0
    return 1.0 + u, u


def diffraction_intensity_2d(z: Array, probe: Array, scan: Point) -> Array:
    n = z.shape[0]
    if z.shape != (n, n) or probe.shape != (n, n):
        raise ValueError("z and probe must be equal square arrays")
    q = centered_grid(n)
    i = position_to_index(q + int(scan[0]), n)
    j = position_to_index(q + int(scan[1]), n)
    exit_wave = probe * z[np.ix_(i, j)]
    return np.abs(np.fft.fft2(exit_wave)) ** 2


def add_poisson_noise_2d(
    intensity: Array, photons: float, rng: np.random.Generator
) -> Array:
    total = float(np.sum(intensity))
    if photons <= 0 or total <= 0:
        raise ValueError("photons and total intensity must be positive")
    rate = float(photons) * np.asarray(intensity, dtype=float) / total
    return rng.poisson(rate).astype(float) * total / float(photons)


def positive_lag_box(lag_count: int) -> Array:
    return np.asarray(list(product(range(1, lag_count + 1), repeat=2)), dtype=int)


def clean_lag_moments_2d(
    intensity: Array, assumed_probe: Array, gamma: float, lag_count: int
) -> Array:
    c_data = np.fft.ifft2(np.asarray(intensity))
    c_probe = np.fft.ifft2(np.abs(np.fft.fft2(assumed_probe)) ** 2)
    lags = positive_lag_box(lag_count)
    values = c_data[lags[:, 0], lags[:, 1]] - c_probe[lags[:, 0], lags[:, 1]]
    scales = np.exp(gamma * np.sum(lags.astype(float) ** 2, axis=1))
    return scales * values


def theoretical_moments_2d(
    positions: Sequence[Point],
    amplitudes: Sequence[complex],
    scan: Point,
    gamma: float,
    lag_count: int,
    n: int,
) -> Array:
    lags = positive_lag_box(lag_count).astype(float)
    result = np.zeros(len(lags), dtype=np.complex128)
    for position, amplitude in zip(positions, amplitudes):
        x = np.asarray(
            [
                wrap_centered(position[0] - scan[0], n),
                wrap_centered(position[1] - scan[1], n),
            ],
            dtype=float,
        )
        weight = np.exp(-2.0 * gamma * float(x @ x))
        exponent = 2.0 * gamma * (lags @ x)
        result += weight * (
            amplitude * np.exp(exponent) + np.conj(amplitude) * np.exp(-exponent)
        )
    return result


def _offset(position: Point, scan: Point, n: int) -> Point:
    return (
        int(wrap_centered(position[0] - scan[0], n)),
        int(wrap_centered(position[1] - scan[1], n)),
    )


@dataclass(frozen=True)
class CalibratedSignatureModel2D:
    """Linear clean-lag model for a known defect signature and unknown locations."""

    points: tuple[Point, ...]
    dictionary: Array
    gram: Array
    column_norms: Array
    row_weights: Array
    assumed_probe: Array
    gamma: float
    lag_count: int
    scans: tuple[Point, ...]


@dataclass(frozen=True)
class CalibratedCaseRecovery2D:
    positions: tuple[Point, ...]
    strengths: tuple[float, ...]
    relative_residual: float


def build_calibrated_signature_model_2d(
    scans: Sequence[Point],
    assumed_gamma: float,
    lag_count: int,
    n: int,
    object_bound: int,
    defect_signature: complex,
    *,
    background_gamma: float | None = None,
) -> CalibratedSignatureModel2D:
    """Build the real joint dictionary for two-scan clean-lag matched filtering."""

    lags = positive_lag_box(lag_count).astype(float)
    row_weights = np.exp(-assumed_gamma * np.sum(lags * lags, axis=1))
    points = tuple(
        (p1, p2)
        for p1 in range(-object_bound, object_bound + 1)
        for p2 in range(-object_bound, object_bound + 1)
    )
    columns = []
    for point in points:
        blocks = []
        for scan in scans:
            x = np.asarray(_offset(point, scan, n), dtype=float)
            probe_weight = np.exp(-2.0 * assumed_gamma * float(x @ x))
            exponent = 2.0 * assumed_gamma * (lags @ x)
            values = probe_weight * (
                defect_signature * np.exp(exponent)
                + np.conj(defect_signature) * np.exp(-exponent)
            )
            blocks.extend(
                (np.real(values) * row_weights, np.imag(values) * row_weights)
            )
        columns.append(np.concatenate(blocks))
    dictionary = np.asarray(columns, dtype=float).T
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gram = dictionary.T @ dictionary
    if not np.all(np.isfinite(gram)):
        raise FloatingPointError("non-finite calibrated Gram matrix")
    return CalibratedSignatureModel2D(
        points=points,
        dictionary=dictionary,
        gram=gram,
        column_norms=np.maximum(np.linalg.norm(dictionary, axis=0), 1e-15),
        row_weights=row_weights,
        assumed_probe=gaussian_probe_2d(
            n, assumed_gamma if background_gamma is None else background_gamma
        ),
        gamma=float(assumed_gamma),
        lag_count=int(lag_count),
        scans=tuple(scans),
    )


def calibrated_signature_target_2d(
    intensities: Sequence[Array], model: CalibratedSignatureModel2D
) -> Array:
    moments = [
        clean_lag_moments_2d(
            intensity, model.assumed_probe, model.gamma, model.lag_count
        )
        for intensity in intensities
    ]
    return np.concatenate(
        [
            block
            for values in moments
            for block in (
                np.real(values) * model.row_weights,
                np.imag(values) * model.row_weights,
            )
        ]
    )


def _nnls_from_gram(
    gram: Array, cross: Array, target_energy: float
) -> tuple[Array, float]:
    """Solve a tiny nonnegative least-squares problem by active-set enumeration."""

    count = len(cross)
    best_coefficients = np.zeros(count, dtype=float)
    best_residual = float(target_energy)
    for mask in range(1, 1 << count):
        active = np.asarray(
            [index for index in range(count) if mask & (1 << index)], dtype=int
        )
        coefficients = np.linalg.lstsq(
            gram[np.ix_(active, active)], cross[active], rcond=1e-12
        )[0]
        if np.any(coefficients < -1e-10):
            continue
        coefficients = np.maximum(coefficients, 0.0)
        residual = float(
            target_energy
            - 2.0 * coefficients @ cross[active]
            + coefficients @ gram[np.ix_(active, active)] @ coefficients
        )
        if residual < best_residual:
            best_residual = residual
            best_coefficients = np.zeros(count, dtype=float)
            best_coefficients[active] = coefficients
    return best_coefficients, max(best_residual, 0.0)


def recover_calibrated_signature_2d(
    target: Array,
    model: CalibratedSignatureModel2D,
    sparsity: int,
    *,
    refinement_sweeps: int = 4,
) -> CalibratedCaseRecovery2D:
    """Recover nonnegative defect severities on the calibrated signature grid."""

    dictionary = model.dictionary
    target = np.asarray(target, dtype=float)
    target_energy = float(target @ target)
    column_norms = model.column_norms
    gram = model.gram
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        full_cross = dictionary.T @ target
    if not np.all(np.isfinite(full_cross)):
        raise FloatingPointError("non-finite calibrated cross-correlation")
    selected: list[int] = []
    coefficients = np.empty(0, dtype=float)
    residual = target.copy()
    for _ in range(sparsity):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scores = (dictionary.T @ residual) / column_norms
        if selected:
            scores[np.asarray(selected, dtype=int)] = -np.inf
        selected.append(int(np.argmax(scores)))
        index = np.asarray(selected, dtype=int)
        coefficients, _ = _nnls_from_gram(
            gram[np.ix_(index, index)], full_cross[index], target_energy
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = target - dictionary[:, index] @ coefficients

    for _ in range(refinement_sweeps):
        changed = False
        for slot in range(sparsity):
            best_index = selected[slot]
            best_coefficients = coefficients
            index = np.asarray(selected, dtype=int)
            _, best_residual = _nnls_from_gram(
                gram[np.ix_(index, index)], full_cross[index], target_energy
            )
            occupied = set(selected)
            occupied.remove(selected[slot])
            fixed = np.asarray(
                [value for item, value in enumerate(selected) if item != slot],
                dtype=int,
            )
            if len(fixed):
                fixed_gram = gram[np.ix_(fixed, fixed)]
                fixed_coefficients, fixed_residual = _nnls_from_gram(
                    fixed_gram, full_cross[fixed], target_energy
                )
                fixed_cross = gram[:, fixed] @ fixed_coefficients
                correlations = full_cross - fixed_cross
                if np.all(fixed_coefficients > 0.0):
                    projected = np.linalg.lstsq(
                        fixed_gram, gram[fixed, :], rcond=1e-12
                    )[0]
                    denominators = np.diag(gram) - np.sum(
                        gram[fixed, :] * projected, axis=0
                    )
                else:
                    denominators = np.diag(gram).copy()
                gains = np.maximum(correlations, 0.0) ** 2 / np.maximum(
                    denominators, 1e-15
                )
                approximate_residuals = fixed_residual - gains
            else:
                approximate_residuals = target_energy - (
                    np.maximum(full_cross, 0.0) ** 2
                    / np.maximum(np.diag(gram), 1e-15)
                )
            if occupied:
                approximate_residuals[np.asarray(tuple(occupied), dtype=int)] = np.inf
            candidate_pool = np.argsort(approximate_residuals)[:16]
            for candidate in candidate_pool:
                trial = selected.copy()
                trial[slot] = int(candidate)
                trial_index = np.asarray(trial, dtype=int)
                trial_coefficients, trial_residual = _nnls_from_gram(
                    gram[np.ix_(trial_index, trial_index)],
                    full_cross[trial_index],
                    target_energy,
                )
                if trial_residual < best_residual - 1e-14 * max(target_energy, 1.0):
                    best_index = candidate
                    best_coefficients = trial_coefficients
                    best_residual = trial_residual
            if best_index != selected[slot]:
                selected[slot] = best_index
                coefficients = best_coefficients
                changed = True
        if not changed:
            break

    index = np.asarray(selected, dtype=int)
    coefficients, residual_energy = _nnls_from_gram(
        gram[np.ix_(index, index)], full_cross[index], target_energy
    )
    ordering = np.argsort(index)
    relative_residual = np.sqrt(residual_energy / max(target_energy, 1e-30))
    return CalibratedCaseRecovery2D(
        positions=tuple(model.points[selected[item]] for item in ordering),
        strengths=tuple(float(coefficients[item]) for item in ordering),
        relative_residual=float(relative_residual),
    )


def recover_calibrated_case_2d(
    intensities: Sequence[Array],
    model: CalibratedSignatureModel2D,
    sparsity: int,
    *,
    refinement_sweeps: int = 4,
) -> CalibratedCaseRecovery2D:
    target = calibrated_signature_target_2d(intensities, model)
    return recover_calibrated_signature_2d(
        target, model, sparsity, refinement_sweeps=refinement_sweeps
    )


def support_exact(true_positions: Iterable[Point], estimated: Iterable[Point]) -> bool:
    return set(true_positions) == set(estimated)


def perturbation_nrmse_2d(
    n: int,
    true_positions: Sequence[Point],
    true_amplitudes: Sequence[complex],
    estimated_positions: Sequence[Point],
    estimated_amplitudes: Sequence[complex],
) -> float:
    truth = np.zeros((n, n), dtype=np.complex128)
    estimate = np.zeros_like(truth)
    for point, value in zip(true_positions, true_amplitudes):
        truth[position_to_index(point[0], n), position_to_index(point[1], n)] = value
    for point, value in zip(estimated_positions, estimated_amplitudes):
        estimate[position_to_index(point[0], n), position_to_index(point[1], n)] = value
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))
