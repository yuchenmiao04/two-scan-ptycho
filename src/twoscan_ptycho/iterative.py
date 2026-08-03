"""Recognized iterative phase-retrieval baselines for the two-scan model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import position_to_index


Array = np.ndarray


@dataclass(frozen=True)
class IterativeRecovery:
    perturbation: Array
    loss: float
    restart: int
    iterations: int


def normalized_intensity_loss(
    perturbation: Array,
    intensities: Array,
    probe: Array,
    scans: Sequence[int],
) -> float:
    """Return a scale-free full-data intensity residual."""

    baseline = np.abs(np.fft.fft(probe)) ** 2
    signal_scale = max(
        float(np.linalg.norm(intensities - baseline[None, :])),
        np.finfo(float).eps,
    )
    z = 1.0 + np.asarray(perturbation, dtype=np.complex128)
    predictions = []
    for scan in scans:
        shifted = np.roll(z, -int(scan))
        predictions.append(np.abs(np.fft.fft(probe * shifted)) ** 2)
    return float(np.linalg.norm(np.asarray(predictions) - intensities) / signal_scale)


def _project_sparse(
    object_estimate: Array,
    candidate_indices: Array,
    sparsity: int,
) -> Array:
    perturbation = np.zeros_like(object_estimate, dtype=np.complex128)
    candidate_values = object_estimate[candidate_indices] - 1.0
    keep_local = np.argpartition(np.abs(candidate_values), -sparsity)[-sparsity:]
    keep = candidate_indices[keep_local]
    perturbation[keep] = object_estimate[keep] - 1.0
    return 1.0 + perturbation


def fienup_error_reduction(
    intensities: Array,
    probe: Array,
    scans: Sequence[int],
    sparsity: int,
    candidate_positions: Sequence[int],
    *,
    iterations: int = 400,
    restarts: int = 8,
    relaxation: float = 0.85,
    seed: int = 0,
) -> IterativeRecovery:
    """Fienup error reduction with the known affine and sparsity constraints.

    Each Fourier projection enforces the measured magnitudes.  The object projection
    enforces the known unit background and retains the largest ``sparsity`` deviations
    on the declared candidate grid.  Multiple random starts are scored by the complete
    two-scan intensity residual.
    """

    intensities = np.asarray(intensities, dtype=float)
    probe = np.asarray(probe, dtype=np.complex128)
    scans = tuple(int(value) for value in scans)
    n = len(probe)
    if intensities.shape != (len(scans), n):
        raise ValueError("intensities must have shape (number of scans, n)")
    if not 0 < sparsity <= len(candidate_positions):
        raise ValueError("invalid sparsity")
    if iterations <= 0 or restarts <= 0:
        raise ValueError("iterations and restarts must be positive")
    candidate_indices = np.asarray(
        [position_to_index(int(position), n) for position in candidate_positions],
        dtype=int,
    )
    magnitudes = np.sqrt(np.maximum(intensities, 0.0))
    probe_power = np.abs(probe) ** 2
    rng = np.random.default_rng(seed)
    best: IterativeRecovery | None = None

    for restart in range(restarts):
        z = np.ones(n, dtype=np.complex128)
        if restart:
            support = rng.choice(candidate_indices, size=sparsity, replace=False)
            phases = rng.uniform(-np.pi, np.pi, size=sparsity)
            z[support] = np.exp(1j * phases)

        normalization = max(float(np.max(probe_power)), np.finfo(float).eps)
        for _ in range(iterations):
            for scan_index, scan in enumerate(scans):
                shifted = np.roll(z, -scan)
                exit_wave = probe * shifted
                transformed = np.fft.fft(exit_wave)
                phase = np.exp(1j * np.angle(transformed))
                projected_exit = np.fft.ifft(magnitudes[scan_index] * phase)
                correction = (
                    relaxation
                    * np.conj(probe)
                    * (projected_exit - exit_wave)
                    / normalization
                )
                z = np.roll(shifted + correction, scan)
            z = _project_sparse(z, candidate_indices, sparsity)

        perturbation = z - 1.0
        loss = normalized_intensity_loss(
            perturbation, intensities, probe, scans
        )
        recovery = IterativeRecovery(
            perturbation=perturbation,
            loss=loss,
            restart=restart,
            iterations=iterations,
        )
        if best is None or recovery.loss < best.loss:
            best = recovery

    if best is None:
        raise RuntimeError("error-reduction baseline produced no result")
    return best


def hard_thresholded_wirtinger_flow(
    intensities: Array,
    probe: Array,
    scans: Sequence[int],
    sparsity: int,
    candidate_positions: Sequence[int],
    *,
    iterations: int = 1200,
    restarts: int = 32,
    learning_rate: float = 0.08,
    hard_threshold_after: int = 450,
    initialization_scale: float = 0.03,
    maximum_amplitude: float = 3.0,
    seed: int = 0,
) -> IterativeRecovery:
    """Hard-thresholded Wirtinger flow using both complete diffraction frames."""

    intensities = np.asarray(intensities, dtype=float)
    probe = np.asarray(probe, dtype=float)
    scans = tuple(int(value) for value in scans)
    candidates = tuple(int(value) for value in candidate_positions)
    n = len(probe)
    if intensities.shape != (len(scans), n):
        raise ValueError("intensities must have shape (number of scans, n)")
    if not 0 < sparsity <= len(candidates):
        raise ValueError("invalid sparsity")

    design = np.empty((len(scans), n, len(candidates)), dtype=np.complex128)
    for scan_index, scan in enumerate(scans):
        for column_index, position in enumerate(candidates):
            offset = position - scan
            atom = np.zeros(n, dtype=np.complex128)
            atom[position_to_index(offset, n)] = probe[
                position_to_index(offset, n)
            ]
            design[scan_index, :, column_index] = np.fft.fft(atom)
    reference = np.fft.fft(probe)
    baseline = np.abs(reference) ** 2
    scale = np.sqrt(
        np.mean((intensities - baseline[None, :]) ** 2, axis=-1)
    )
    scale = np.maximum(scale, np.finfo(float).eps)

    rng = np.random.default_rng(seed)
    coefficients = initialization_scale * (
        rng.standard_normal((restarts, len(candidates)))
        + 1j * rng.standard_normal((restarts, len(candidates)))
    )
    coefficients[0] = 0.0
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients.real)
    beta1, beta2 = 0.9, 0.999

    for iteration in range(1, iterations + 1):
        transformed = reference[None, None, :] + np.einsum(
            "snk,rk->rsn", design, coefficients, optimize=True
        )
        residual = (np.abs(transformed) ** 2 - intensities[None, :, :])
        weighted = residual * transformed / (scale[None, :, None] ** 2)
        gradient = np.einsum(
            "snk,rsn->rk", np.conj(design), weighted, optimize=True
        ) / (len(scans) * n)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * np.abs(gradient) ** 2
        corrected_first = first_moment / (1.0 - beta1**iteration)
        corrected_second = second_moment / (1.0 - beta2**iteration)
        coefficients -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        amplitudes = np.abs(coefficients)
        large = amplitudes > maximum_amplitude
        coefficients[large] *= maximum_amplitude / amplitudes[large]
        if iteration > hard_threshold_after:
            keep = np.argpartition(
                np.abs(coefficients), -sparsity, axis=-1
            )[:, -sparsity:]
            mask = np.zeros_like(coefficients, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=-1)
            coefficients[~mask] = 0.0

    transformed = reference[None, None, :] + np.einsum(
        "snk,rk->rsn", design, coefficients, optimize=True
    )
    losses = np.mean(
        (
            (np.abs(transformed) ** 2 - intensities[None, :, :])
            / scale[None, :, None]
        )
        ** 2,
        axis=(-2, -1),
    )
    best_restart = int(np.argmin(losses))
    estimate = np.zeros(n, dtype=np.complex128)
    for column_index, position in enumerate(candidates):
        estimate[position_to_index(position, n)] = coefficients[
            best_restart, column_index
        ]
    return IterativeRecovery(
        perturbation=estimate,
        loss=float(losses[best_restart]),
        restart=best_restart,
        iterations=iterations,
    )
