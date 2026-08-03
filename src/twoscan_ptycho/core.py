"""Finite-DFT sparse affine ptychography utilities.

The implementation follows the notation in the theory note.  Arrays are stored in
centered spatial order ``[-M, ..., M]`` for odd ``n = 2M+1``.  NumPy's ordinary FFT
may be applied directly in this order because the induced Fourier phase has unit
magnitude and therefore does not change the measured intensities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
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


def add_intensity_awgn(
    intensity: Array,
    snr_db: float,
    rng: np.random.Generator,
    *,
    clip_nonnegative: bool = True,
) -> Array:
    """Add real white Gaussian detector noise at the requested RMS SNR."""

    intensity = np.asarray(intensity, dtype=float)
    rms = float(np.linalg.norm(intensity) / np.sqrt(intensity.size))
    sigma = rms * 10.0 ** (-float(snr_db) / 20.0)
    noisy = intensity + rng.normal(scale=sigma, size=intensity.shape)
    if clip_nonnegative:
        noisy = np.maximum(noisy, 0.0)
    return noisy


def add_intensity_poisson(
    intensity: Array, photons: float, rng: np.random.Generator
) -> Array:
    """Apply Poisson noise with a specified expected total photon count."""

    intensity = np.asarray(intensity, dtype=float)
    total = float(np.sum(intensity))
    if photons <= 0 or total <= 0:
        raise ValueError("photons and total intensity must be positive")
    rate = photons * intensity / total
    counts = rng.poisson(rate)
    return counts.astype(float) * total / photons


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


def moment_dictionaries(
    gamma: float, lag_count: int, max_offset: int
) -> tuple[Array, Array]:
    """Build real hyperbolic-cosine and imaginary hyperbolic-sine dictionaries.

    ``C`` has offsets ``0, ..., max_offset``.  Its zero-offset column is one and
    positive-offset columns are ``2 cosh(2 gamma ell x)``.  ``S`` has offsets
    ``1, ..., max_offset`` and columns ``2 sinh(2 gamma ell x)``.
    """

    if lag_count <= 0 or max_offset < 0:
        raise ValueError("invalid lag_count or max_offset")
    ell = np.arange(1, lag_count + 1, dtype=float)[:, None]
    x0 = np.arange(0, max_offset + 1, dtype=float)[None, :]
    c = 2.0 * np.cosh(2.0 * gamma * ell * x0)
    c[:, 0] = 1.0
    if max_offset == 0:
        s = np.empty((lag_count, 0), dtype=float)
    else:
        x1 = np.arange(1, max_offset + 1, dtype=float)[None, :]
        s = 2.0 * np.sinh(2.0 * gamma * ell * x1)
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise FloatingPointError("moment dictionary overflow; reduce gamma or the field of view")
    return c, s


@dataclass(frozen=True)
class OrbitRecovery:
    coefficients: Mapping[int, complex]
    support: tuple[int, ...]
    residual_norm: float
    condition_real: float
    condition_imag: float


def _equilibrate_rows(c: Array, s: Array, target: Array) -> tuple[Array, Array, Array]:
    scales = np.maximum(np.max(np.abs(c), axis=1), 1.0)
    if s.shape[1]:
        scales = np.maximum(scales, np.max(np.abs(s), axis=1))
    return c / scales[:, None], s / scales[:, None], target / scales


def _scale_moment_rows(
    c: Array,
    s: Array,
    target: Array,
    gamma: float,
    mode: str,
) -> tuple[Array, Array, Array]:
    if mode == "dictionary":
        return _equilibrate_rows(c, s, target)
    if mode == "noise":
        lags = np.arange(1, c.shape[0] + 1, dtype=float)
        # White intensity noise remains homoscedastic after the unitary-equivalent
        # inverse DFT, whereas m_l multiplies it by exp(gamma*l^2).
        weights = np.exp(-gamma * lags * lags)
        return c * weights[:, None], s * weights[:, None], target * weights
    if mode == "none":
        return c, s, target
    raise ValueError("row_scaling must be 'dictionary', 'noise', or 'none'")


def _normalized_correlations(dictionary: Array, residual: Array) -> Array:
    if dictionary.shape[1] == 0:
        return np.empty(0, dtype=float)
    norms = np.linalg.norm(dictionary, axis=0)
    values = np.zeros(dictionary.shape[1], dtype=float)
    valid = norms > 0
    values[valid] = np.abs(dictionary[:, valid].T @ residual) / norms[valid]
    return values


def _fit_orbit_support(
    c: Array,
    s: Array,
    target_real: Array,
    target_imag: Array,
    support: Sequence[int],
) -> tuple[Array, Array, Array, Array, float]:
    support = tuple(sorted(int(x) for x in support))
    if support:
        c_sel = c[:, support]
        alpha = np.linalg.lstsq(c_sel, target_real, rcond=None)[0]
        residual_real = target_real - c_sel @ alpha
    else:
        alpha = np.empty(0, dtype=float)
        residual_real = target_real.copy()
    positive_support = tuple(x for x in support if x > 0)
    if positive_support:
        s_sel = s[:, np.asarray(positive_support) - 1]
        beta = np.linalg.lstsq(s_sel, target_imag, rcond=None)[0]
        residual_imag = target_imag - s_sel @ beta
    else:
        beta = np.empty(0, dtype=float)
        residual_imag = target_imag.copy()
    residual_norm = float(
        np.hypot(np.linalg.norm(residual_real), np.linalg.norm(residual_imag))
    )
    return alpha, beta, residual_real, residual_imag, residual_norm


def _orbit_recovery_from_fit(
    c_raw: Array,
    s_raw: Array,
    support: Sequence[int],
    alpha: Array,
    beta: Array,
    residual_norm: float,
) -> OrbitRecovery:
    support = tuple(sorted(int(x) for x in support))
    positive_support = tuple(x for x in support if x > 0)
    beta_by_x = {x: value for x, value in zip(positive_support, beta)}
    coefficients = {
        x: complex(real_value, beta_by_x.get(x, 0.0))
        for x, real_value in zip(support, alpha)
    }
    c_sel_raw = c_raw[:, support] if support else np.empty((c_raw.shape[0], 0))
    s_sel_raw = (
        s_raw[:, np.asarray(positive_support) - 1]
        if positive_support
        else np.empty((s_raw.shape[0], 0))
    )
    cond_real = float(np.linalg.cond(c_sel_raw)) if c_sel_raw.size else 1.0
    cond_imag = float(np.linalg.cond(s_sel_raw)) if s_sel_raw.size else 1.0
    return OrbitRecovery(
        coefficients=coefficients,
        support=support,
        residual_norm=float(residual_norm),
        condition_real=cond_real,
        condition_imag=cond_imag,
    )


def recover_orbit_coefficients_group_omp(
    moments: Array,
    gamma: float,
    max_offset: int,
    sparsity: int,
    *,
    row_equilibrate: bool | None = None,
    row_scaling: str = "dictionary",
    relative_tolerance: float = 1e-11,
) -> OrbitRecovery:
    """Recover reciprocal-orbit coefficients with a practical group OMP solver."""

    moments = np.asarray(moments, dtype=np.complex128)
    lag_count = len(moments)
    if sparsity <= 0:
        raise ValueError("sparsity must be positive")
    c, s = moment_dictionaries(gamma, lag_count, max_offset)
    target = moments.copy()
    if row_equilibrate is not None:
        row_scaling = "dictionary" if row_equilibrate else "none"
    c_work, s_work, target = _scale_moment_rows(
        c, s, target, gamma, row_scaling
    )

    target_real = target.real
    target_imag = target.imag
    residual_real = target_real.copy()
    residual_imag = target_imag.copy()
    support: list[int] = []
    alpha = np.empty(0, dtype=float)
    beta = np.empty(0, dtype=float)
    baseline = max(np.linalg.norm(target), np.finfo(float).eps)

    for _ in range(min(sparsity, max_offset + 1)):
        score = _normalized_correlations(c_work, residual_real) ** 2
        if max_offset:
            score[1:] += _normalized_correlations(s_work, residual_imag) ** 2
        if support:
            score[np.asarray(support, dtype=int)] = -np.inf
        selected = int(np.argmax(score))
        if not np.isfinite(score[selected]):
            break
        support.append(selected)
        support.sort()

        c_sel = c_work[:, support]
        alpha = np.linalg.lstsq(c_sel, target_real, rcond=None)[0]
        residual_real = target_real - c_sel @ alpha

        positive_support = [x for x in support if x > 0]
        if positive_support:
            s_sel = s_work[:, np.asarray(positive_support) - 1]
            beta = np.linalg.lstsq(s_sel, target_imag, rcond=None)[0]
            residual_imag = target_imag - s_sel @ beta
        else:
            beta = np.empty(0, dtype=float)
            residual_imag = target_imag.copy()
        if np.hypot(np.linalg.norm(residual_real), np.linalg.norm(residual_imag)) <= relative_tolerance * baseline:
            break

    residual = np.hypot(np.linalg.norm(residual_real), np.linalg.norm(residual_imag))
    return _orbit_recovery_from_fit(
        c, s, support, alpha, beta, float(residual)
    )


def recover_orbit_coefficients_beam(
    moments: Array,
    gamma: float,
    max_offset: int,
    sparsity: int,
    *,
    beam_width: int = 64,
    row_equilibrate: bool | None = None,
    row_scaling: str = "dictionary",
) -> OrbitRecovery:
    """Structured variable-projection recovery with a support beam search.

    For every candidate support, the real and imaginary weights are eliminated by
    least squares.  Keeping the best ``beam_width`` partial supports is more expensive
    than group OMP but substantially more reliable for the coherent minimum-lag
    dictionaries.  Setting ``beam_width`` at least as large as all partial supports
    reduces to exhaustive support enumeration.
    """

    moments = np.asarray(moments, dtype=np.complex128)
    if sparsity <= 0 or beam_width <= 0:
        raise ValueError("sparsity and beam_width must be positive")
    c_raw, s_raw = moment_dictionaries(gamma, len(moments), max_offset)
    if row_equilibrate is not None:
        row_scaling = "dictionary" if row_equilibrate else "none"
    c, s, target = _scale_moment_rows(
        c_raw, s_raw, moments.copy(), gamma, row_scaling
    )
    target_real, target_imag = target.real, target.imag

    beam: list[tuple[float, tuple[int, ...], Array, Array]] = [
        (float(np.linalg.norm(target)), tuple(), np.empty(0), np.empty(0))
    ]
    for _ in range(min(sparsity, max_offset + 1)):
        candidates: dict[tuple[int, ...], tuple[float, Array, Array]] = {}
        for _, support, _, _ in beam:
            start = support[-1] + 1 if support else 0
            for offset in range(start, max_offset + 1):
                child = support + (offset,)
                alpha, beta, _, _, residual = _fit_orbit_support(
                    c, s, target_real, target_imag, child
                )
                previous = candidates.get(child)
                if previous is None or residual < previous[0]:
                    candidates[child] = (residual, alpha, beta)
        ranked = sorted(
            (
                (residual, support, alpha, beta)
                for support, (residual, alpha, beta) in candidates.items()
            ),
            key=lambda item: item[0],
        )
        beam = ranked[:beam_width]
        if not beam:
            break

    if not beam:
        raise RuntimeError("beam search produced no support candidates")
    residual, support, alpha, beta = beam[0]
    return _orbit_recovery_from_fit(
        c_raw, s_raw, support, alpha, beta, residual
    )


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


@dataclass(frozen=True)
class EndToEndRecovery:
    result: TwoScanRecovery
    scan_recoveries: tuple[OrbitRecovery, OrbitRecovery]
    reflection_sums: tuple[Array, Array]


def recover_from_two_intensities(
    intensity1: Array,
    intensity2: Array,
    probe: Array,
    gamma: float,
    r1: int,
    r2: int,
    sparsity: int,
    max_offset: int,
    *,
    lag_count: int | None = None,
    solver: str = "beam",
    beam_width: int = 64,
    row_scaling: str = "dictionary",
) -> EndToEndRecovery:
    if lag_count is None:
        lag_count = 2 * sparsity
    moments1 = clean_lag_moments(intensity1, probe, gamma, lag_count)
    moments2 = clean_lag_moments(intensity2, probe, gamma, lag_count)
    if solver == "beam":
        recover = lambda values: recover_orbit_coefficients_beam(
            values,
            gamma,
            max_offset,
            sparsity,
            beam_width=beam_width,
            row_scaling=row_scaling,
        )
    elif solver == "omp":
        recover = lambda values: recover_orbit_coefficients_group_omp(
            values, gamma, max_offset, sparsity, row_scaling=row_scaling
        )
    else:
        raise ValueError(f"unknown solver {solver!r}")
    orbit1 = recover(moments1)
    orbit2 = recover(moments2)
    q1 = centered_reflection_sum_from_orbits(n=len(probe), gamma=gamma, coefficients=orbit1.coefficients)
    q2 = centered_reflection_sum_from_orbits(n=len(probe), gamma=gamma, coefficients=orbit2.coefficients)
    p1 = translate_centered_sum(q1, r1)
    p2 = translate_centered_sum(q2, r2)
    result = recover_from_two_reflection_sums(p1, p2, r1, r2, sparsity)
    return EndToEndRecovery(
        result=result,
        scan_recoveries=(orbit1, orbit2),
        reflection_sums=(p1, p2),
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


def support_metrics(truth: Array, estimate: Array, sparsity: int) -> dict[str, float]:
    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have equal shapes")
    true_support = set(np.flatnonzero(np.abs(truth) > 0).tolist())
    estimated_support = set(
        np.argpartition(np.abs(estimate), -sparsity)[-sparsity:].tolist()
    )
    intersection = len(true_support & estimated_support)
    precision = intersection / max(len(estimated_support), 1)
    recall = intersection / max(len(true_support), 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    denominator = max(float(np.linalg.norm(truth)), np.finfo(float).eps)
    nrmse = float(np.linalg.norm(estimate - truth) / denominator)
    return {
        "support_exact": float(true_support == estimated_support),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "nrmse": nrmse,
    }


def best_circular_location_error(
    true_positions: Sequence[int], estimated_positions: Sequence[int], n: int
) -> float:
    if len(true_positions) != len(estimated_positions):
        return float("inf")
    if not true_positions:
        return 0.0
    best = float("inf")
    for candidate in permutations(estimated_positions):
        error = sum(
            circular_distance(int(p), int(q), n) ** 2
            for p, q in zip(true_positions, candidate)
        )
        best = min(best, error)
    return float(np.sqrt(best / len(true_positions)))


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


def random_amplitudes(
    sparsity: int,
    rng: np.random.Generator,
    *,
    physical_phase: bool = True,
    minimum_magnitude: float = 0.0,
) -> tuple[complex, ...]:
    if minimum_magnitude < 0:
        raise ValueError("minimum_magnitude must be nonnegative")
    if physical_phase:
        values: list[complex] = []
        while len(values) < sparsity:
            phase = float(rng.uniform(-0.9 * np.pi, 0.9 * np.pi))
            value = complex(np.exp(1j * phase) - 1.0)
            if abs(value) >= minimum_magnitude:
                values.append(value)
        return tuple(values)
    lower = max(0.5, minimum_magnitude)
    if lower > 1.5:
        raise ValueError("minimum_magnitude cannot exceed 1.5 for generic amplitudes")
    magnitude = rng.uniform(lower, 1.5, size=sparsity)
    phase = rng.uniform(-np.pi, np.pi, size=sparsity)
    return tuple(complex(r * np.exp(1j * p)) for r, p in zip(magnitude, phase))


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
