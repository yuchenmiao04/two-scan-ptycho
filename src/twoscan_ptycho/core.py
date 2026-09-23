"""Finite-DFT sparse affine ptychography utilities.

The implementation follows the notation in the theory note.  Arrays are stored in
centered spatial order ``[-M, ..., M]`` for odd ``n = 2M+1``.  NumPy's ordinary FFT
may be applied directly in this order because the induced Fourier phase has unit
magnitude and therefore does not change the measured intensities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray


def _require_odd(n: int) -> int:
    if n <= 0 or n % 2 != 1:
        raise ValueError(f"n must be a positive odd integer, got {n}")
    return (n - 1) // 2


def centered_grid(n: int) -> Array:
    """Return the integer representatives ``[-M, ..., M]`` of ``Z_n``."""

    m = _require_odd(n)
    return np.arange(-m, m + 1, dtype=int)


def wrap_centered(values: int | Array, n: int) -> int | Array:
    """Wrap integer values into the centered representatives of ``Z_n``."""

    m = _require_odd(n)
    wrapped = (np.asarray(values) + m) % n - m
    if np.ndim(values) == 0:
        return int(wrapped)
    return wrapped.astype(int, copy=False)


def position_to_index(values: int | Array, n: int) -> int | Array:
    """Map centered positions to array indices."""

    m = _require_odd(n)
    indices = (np.asarray(values) + m) % n
    if np.ndim(values) == 0:
        return int(indices)
    return indices.astype(int, copy=False)


def circular_distance(p: int, q: int, n: int) -> int:
    return abs(wrap_centered(p - q, n))


def gaussian_probe(n: int, gamma: float, dtype=np.float64) -> Array:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    q = centered_grid(n).astype(dtype)
    return np.exp(-gamma * q * q)


def sparse_perturbation(
    n: int, positions: Sequence[int], amplitudes: Sequence[complex]
) -> Array:
    if len(positions) != len(amplitudes):
        raise ValueError("positions and amplitudes must have the same length")
    if len(set(int(wrap_centered(p, n)) for p in positions)) != len(positions):
        raise ValueError("positions must be distinct modulo n")
    u = np.zeros(n, dtype=np.complex128)
    for position, amplitude in zip(positions, amplitudes):
        u[position_to_index(position, n)] = complex(amplitude)
    return u


def affine_object(
    n: int, positions: Sequence[int], amplitudes: Sequence[complex]
) -> Array:
    return np.ones(n, dtype=np.complex128) + sparse_perturbation(
        n, positions, amplitudes
    )


def exit_wave(z: Array, probe: Array, scan: int) -> Array:
    n = len(z)
    if probe.shape != (n,):
        raise ValueError("z and probe must be one-dimensional arrays of equal length")
    q = centered_grid(n)
    shifted_indices = position_to_index(q + int(scan), n)
    return probe * z[shifted_indices]


def diffraction_intensity(z: Array, probe: Array, scan: int) -> Array:
    h = exit_wave(z, probe, scan)
    return np.abs(np.fft.fft(h)) ** 2


def intensity_autocorrelation(intensity: Array) -> Array:
    """Invert a full DFT intensity into circular autocorrelation."""

    return np.fft.ifft(np.asarray(intensity))


def clean_lag_moments(
    intensity: Array, probe: Array, gamma: float, lag_count: int
) -> Array:
    """Return ``m[1], ..., m[lag_count]`` from a diffraction pattern."""

    if lag_count <= 0 or lag_count >= len(intensity):
        raise ValueError("lag_count must lie between 1 and n-1")
    c_data = intensity_autocorrelation(intensity)
    probe_intensity = np.abs(np.fft.fft(probe)) ** 2
    c_probe = intensity_autocorrelation(probe_intensity)
    lags = np.arange(1, lag_count + 1, dtype=float)
    return np.exp(gamma * lags * lags) * (c_data[1 : lag_count + 1] - c_probe[1 : lag_count + 1])


def theoretical_moments(
    positions: Sequence[int],
    amplitudes: Sequence[complex],
    scan: int,
    gamma: float,
    lag_count: int,
    n: int,
) -> Array:
    """Evaluate the clean Gaussian moment formula directly."""

    lags = np.arange(1, lag_count + 1, dtype=float)
    result = np.zeros(lag_count, dtype=np.complex128)
    for position, amplitude in zip(positions, amplitudes):
        x = float(wrap_centered(int(position) - int(scan), n))
        weight = np.exp(-2.0 * gamma * x * x)
        result += weight * (
            amplitude * np.exp(2.0 * gamma * x * lags)
            + np.conj(amplitude) * np.exp(-2.0 * gamma * x * lags)
        )
    return result


def centered_reflection_sum_from_orbits(
    n: int, gamma: float, coefficients: Mapping[int, complex]
) -> Array:
    """Convert deweighted reciprocal-orbit coefficients into centered ``Q_r``."""

    q_sum = np.zeros(n, dtype=np.complex128)
    for offset, coefficient in coefficients.items():
        offset = int(offset)
        if offset < 0:
            raise ValueError("orbit offsets must be nonnegative")
        if offset == 0:
            q_sum[position_to_index(0, n)] += float(np.real(coefficient))
        else:
            q_coefficient = np.exp(2.0 * gamma * offset * offset) * coefficient
            q_sum[position_to_index(offset, n)] += q_coefficient
            q_sum[position_to_index(-offset, n)] += np.conj(q_coefficient)
    return q_sum


def translate_centered_sum(q_sum: Array, scan: int) -> Array:
    """Translate scan-centered ``Q_r`` into global ``P_r = mu + R_r mu``."""

    n = len(q_sum)
    result = np.zeros(n, dtype=np.complex128)
    for offset in centered_grid(n):
        result[position_to_index(offset + int(scan), n)] = q_sum[
            position_to_index(offset, n)
        ]
    return result


def conjugate_reflection(values: Array, center: int) -> Array:
    """Return ``R_center values`` with ``(R f)[s] = conj(f[2r-s])``."""

    n = len(values)
    grid = centered_grid(n)
    source = position_to_index(2 * int(center) - grid, n)
    return np.conj(values[source])


def reflection_sum(
    perturbation: Array, center: int
) -> Array:
    return perturbation + conjugate_reflection(perturbation, center)


@dataclass(frozen=True)
class TwoScanRecovery:
    perturbation: Array
    positions: tuple[int, ...]
    amplitudes: tuple[complex, ...]
    difference_residual: float


def recover_from_two_reflection_sums(
    p1: Array,
    p2: Array,
    r1: int,
    r2: int,
    sparsity: int,
) -> TwoScanRecovery:
    """Integrate ``mu - T_d mu`` around cyclic scan orbits and sparsify."""

    if p1.shape != p2.shape or p1.ndim != 1:
        raise ValueError("p1 and p2 must be one-dimensional arrays of equal size")
    n = len(p1)
    _require_odd(n)
    d = (2 * (int(r2) - int(r1))) % n
    if d == 0:
        raise ValueError("the two scan centers must generate a nonzero translation")
    g = p2 - conjugate_reflection(p1, r2)
    estimate = np.zeros(n, dtype=np.complex128)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue
        orbit: list[int] = []
        current = start
        while not visited[current]:
            visited[current] = True
            orbit.append(current)
            current = (current + d) % n
        increments = g[np.asarray(orbit)]
        increments = increments - np.mean(increments)
        integrated = np.zeros(len(orbit), dtype=np.complex128)
        for k in range(1, len(orbit)):
            integrated[k] = integrated[k - 1] + increments[k]
        majority_level = np.median(integrated.real) + 1j * np.median(integrated.imag)
        estimate[np.asarray(orbit)] = integrated - majority_level

    if sparsity < n:
        keep = np.argpartition(np.abs(estimate), -sparsity)[-sparsity:]
        sparse_estimate = np.zeros_like(estimate)
        sparse_estimate[keep] = estimate[keep]
        estimate = sparse_estimate
    support_indices = np.flatnonzero(np.abs(estimate) > 0)
    positions = tuple(int(centered_grid(n)[idx]) for idx in support_indices)
    amplitudes = tuple(complex(estimate[idx]) for idx in support_indices)
    residual = np.linalg.norm(g - (estimate - np.roll(estimate, d)))
    return TwoScanRecovery(
        perturbation=estimate,
        positions=positions,
        amplitudes=amplitudes,
        difference_residual=float(residual),
    )


def reflect_sparse_object(
    n: int,
    positions: Sequence[int],
    amplitudes: Sequence[complex],
    center: int,
) -> tuple[tuple[int, ...], tuple[complex, ...]]:
    reflected_positions = tuple(
        int(wrap_centered(2 * int(center) - int(position), n))
        for position in positions
    )
    reflected_amplitudes = tuple(complex(np.conj(a)) for a in amplitudes)
    return reflected_positions, reflected_amplitudes


def sample_separated_positions(
    n: int,
    sparsity: int,
    minimum_distance: int,
    scans: Sequence[int],
    margin: int,
    rng: np.random.Generator,
    *,
    location_bound: int | None = None,
    max_attempts: int = 20_000,
) -> tuple[int, ...]:
    """Sample positions satisfying circular separation and both no-wrap margins."""

    m = _require_odd(n)
    if location_bound is None:
        location_bound = m
    candidates = [
        int(t)
        for t in centered_grid(n)
        if abs(t) <= location_bound
        and all(abs(wrap_centered(t - int(r), n)) <= m - margin for r in scans)
    ]
    if len(candidates) < sparsity:
        raise ValueError("not enough admissible positions")
    for _ in range(max_attempts):
        available = candidates.copy()
        selected: list[int] = []
        while available and len(selected) < sparsity:
            value = int(available[int(rng.integers(0, len(available)))])
            selected.append(value)
            available = [
                candidate
                for candidate in available
                if candidate != value
                and all(
                    circular_distance(candidate, existing, n) > minimum_distance
                    for existing in selected
                )
            ]
        if len(selected) == sparsity:
            return tuple(sorted(selected))
    raise RuntimeError("failed to sample a separated configuration")


def split_null_counterexample(
    gamma: float, sparsity: int, offsets: Sequence[int] | None = None
) -> dict[str, Array | float]:
    """Construct the ``2N-1`` clean-lag split-null ambiguity for the real channel."""

    if offsets is None:
        offsets = tuple(range(1, 2 * sparsity + 1))
    if len(offsets) != 2 * sparsity or min(offsets) <= 0:
        raise ValueError("offsets must contain exactly 2N positive entries")
    lag_count = 2 * sparsity - 1
    ell = np.arange(1, lag_count + 1, dtype=float)[:, None]
    x = np.asarray(offsets, dtype=float)[None, :]
    matrix = 2.0 * np.cosh(2.0 * gamma * ell * x)
    _, _, vh = np.linalg.svd(matrix)
    null_vector = vh[-1]
    left = matrix[:, :sparsity] @ null_vector[:sparsity]
    right = matrix[:, sparsity:] @ (-null_vector[sparsity:])
    relative_residual = float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(left), np.finfo(float).eps)
    )
    return {
        "matrix": matrix,
        "null_vector": null_vector,
        "left_moments": left,
        "right_moments": right,
        "relative_residual": relative_residual,
    }
