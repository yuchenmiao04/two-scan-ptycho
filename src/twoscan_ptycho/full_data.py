"""Known-background full-diffraction sparse linearization baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import (
    centered_grid,
    position_to_index,
    wrap_centered,
)


Array = np.ndarray


@dataclass(frozen=True)
class FullDataRecovery:
    perturbation: Array
    positions: tuple[int, ...]
    amplitudes: tuple[complex, ...]
    residual_norm: float
    support_history: tuple[tuple[int, ...], ...]


def holographic_linear_dictionaries(
    probe: Array, scans: Sequence[int], candidate_positions: Sequence[int]
) -> tuple[Array, Array]:
    """Build derivatives with respect to real and imaginary atom amplitudes."""

    n = len(probe)
    reference = np.fft.fft(probe)
    real_columns: list[Array] = []
    imag_columns: list[Array] = []
    for position in candidate_positions:
        real_blocks = []
        imag_blocks = []
        for scan in scans:
            offset = wrap_centered(int(position) - int(scan), n)
            delta_exit = np.zeros(n, dtype=np.complex128)
            delta_exit[position_to_index(offset, n)] = probe[
                position_to_index(offset, n)
            ]
            transformed = np.fft.fft(delta_exit)
            cross = np.conj(reference) * transformed
            real_blocks.append(2.0 * np.real(cross))
            imag_blocks.append(-2.0 * np.imag(cross))
        real_columns.append(np.concatenate(real_blocks))
        imag_columns.append(np.concatenate(imag_blocks))
    return np.column_stack(real_columns), np.column_stack(imag_columns)


def _fit_support(
    a_real: Array, a_imag: Array, target: Array, support: Sequence[int]
) -> tuple[Array, float]:
    support = tuple(sorted(int(value) for value in support))
    if not support:
        return np.empty(0, dtype=float), float(np.linalg.norm(target))
    design = np.column_stack((a_real[:, support], a_imag[:, support]))
    coefficients = np.linalg.lstsq(design, target, rcond=1e-12)[0]
    if not np.all(np.isfinite(coefficients)):
        return coefficients, float("inf")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fitted = design @ coefficients
        residual = float(np.linalg.norm(fitted - target))
    if not np.isfinite(residual):
        residual = float("inf")
    return coefficients, residual


def _beam_support(
    a_real: Array,
    a_imag: Array,
    target: Array,
    sparsity: int,
    beam_width: int,
) -> tuple[tuple[int, ...], Array, float]:
    beam: list[tuple[float, tuple[int, ...], Array]] = [
        (float(np.linalg.norm(target)), tuple(), np.empty(0))
    ]
    n_candidates = a_real.shape[1]
    for _ in range(sparsity):
        children: list[tuple[float, tuple[int, ...], Array]] = []
        for _, support, _ in beam:
            start = support[-1] + 1 if support else 0
            for candidate in range(start, n_candidates):
                child = support + (candidate,)
                coefficients, residual = _fit_support(
                    a_real, a_imag, target, child
                )
                children.append((residual, child, coefficients))
        children.sort(key=lambda item: item[0])
        beam = children[:beam_width]
        if not beam:
            raise RuntimeError("full-data beam search produced no candidates")
    residual, support, coefficients = beam[0]
    return support, coefficients, residual


def _quadratic_atom_intensity(
    perturbation: Array, probe: Array, scan: int
) -> Array:
    n = len(probe)
    q = centered_grid(n)
    shifted = perturbation[position_to_index(q + int(scan), n)]
    return np.abs(np.fft.fft(probe * shifted)) ** 2


def _full_intensity(
    perturbation: Array, probe: Array, scan: int
) -> Array:
    n = len(probe)
    q = centered_grid(n)
    shifted = perturbation[position_to_index(q + int(scan), n)]
    return np.abs(np.fft.fft(probe * (1.0 + shifted))) ** 2


def recover_full_data_linearized(
    intensities: Array,
    probe: Array,
    scans: Sequence[int],
    sparsity: int,
    candidate_positions: Sequence[int],
    *,
    beam_width: int = 128,
    debias_iterations: int = 3,
    physical_phase_projection: bool = True,
    maximum_amplitude: float = 2.5,
) -> FullDataRecovery:
    """Recover support from all diffraction samples and iteratively remove |Fu|^2."""

    intensities = np.asarray(intensities, dtype=float)
    if intensities.shape != (len(scans), len(probe)):
        raise ValueError("intensities must have shape (number of scans, n)")
    candidates = tuple(int(value) for value in candidate_positions)
    a_real, a_imag = holographic_linear_dictionaries(probe, scans, candidates)
    baseline = np.abs(np.fft.fft(probe)) ** 2
    data_difference = intensities - baseline[None, :]
    perturbation = np.zeros(len(probe), dtype=np.complex128)
    support_history: list[tuple[int, ...]] = []
    best_residual = float("inf")
    best_perturbation = perturbation.copy()

    for _ in range(max(debias_iterations, 1)):
        quadratic = np.stack(
            [
                _quadratic_atom_intensity(perturbation, probe, scan)
                for scan in scans
            ]
        )
        target = (data_difference - quadratic).reshape(-1)
        support, coefficients, _ = _beam_support(
            a_real, a_imag, target, sparsity, beam_width
        )
        support_history.append(tuple(candidates[index] for index in support))
        real = coefficients[: len(support)]
        imag = coefficients[len(support) :]
        perturbation = np.zeros(len(probe), dtype=np.complex128)
        for index, real_value, imag_value in zip(support, real, imag):
            amplitude = complex(real_value, imag_value)
            if physical_phase_projection:
                amplitude = complex(
                    np.exp(1j * np.angle(1.0 + amplitude)) - 1.0
                )
            elif abs(amplitude) > maximum_amplitude:
                amplitude *= maximum_amplitude / abs(amplitude)
            perturbation[position_to_index(candidates[index], len(probe))] = amplitude
        full_residual = float(
            np.sqrt(
                sum(
                    np.linalg.norm(
                        _full_intensity(perturbation, probe, scan)
                        - intensities[scan_index]
                    )
                    ** 2
                    for scan_index, scan in enumerate(scans)
                )
            )
        )
        if full_residual < best_residual:
            best_residual = full_residual
            best_perturbation = perturbation.copy()

    perturbation = best_perturbation
    support_indices = np.flatnonzero(np.abs(perturbation) > 0)
    grid = centered_grid(len(probe))
    positions = tuple(int(grid[index]) for index in support_indices)
    amplitudes = tuple(complex(perturbation[index]) for index in support_indices)
    return FullDataRecovery(
        perturbation=perturbation,
        positions=positions,
        amplitudes=amplitudes,
        residual_norm=best_residual,
        support_history=tuple(support_history),
    )
