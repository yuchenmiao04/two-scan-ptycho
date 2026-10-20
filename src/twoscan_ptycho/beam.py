"""Beam-search moment reconstruction used in the decoder ablation.

Its independent reflection-orbit search does not enforce the same support class
as joint raw-lag fitting. Report this ablation separately from the common-prior
five-method comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from .core import (
    Array, TwoScanRecovery, clean_lag_moments,
    centered_reflection_sum_from_orbits, translate_centered_sum,
    recover_from_two_reflection_sums,
)

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
