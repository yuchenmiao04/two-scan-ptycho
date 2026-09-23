"""Raw-lag and spectral reconstruction with explicit numerical scope.

Arrays use the centered spatial order of :mod:`twoscan_ptycho.core`.  The joint
dictionary has columns ``[Re(u[candidates]), Im(u[candidates])]`` and rows
``[Re(D_scan0[1:L+1]), Im(D_scan0[1:L+1]), Re(D_scan1...), ...]``.
Consequently independent real white detector noise gives equal variance to every
row within one frame, before any user-specified whitening.

Example (NumPy only)::

    from twoscan_ptycho.core import affine_object, diffraction_intensity
    from twoscan_ptycho.reconstruction import build_joint_dictionary, GlobalSupportSolver
    model = build_joint_dictionary(257, .025, (0, 1), range(-10, 11), 2)
    solver = GlobalSupportSolver(model, 2)  # SVD preparation, reused across trials
    z = affine_object(257, (-5, 4), (.3+.2j, -.4+.1j))
    frames = [diffraction_intensity(z, model.probes[i], r)
              for i, r in enumerate(model.scans)]
    result = solver.solve_intensities(frames)
    print(result.support, result.residual, result.status, result.timings)

``GlobalSupportSolver`` evaluates every legal support of size zero through the
supplied upper bound.  It solves physical raw-lag least squares using SVD, never
normal equations.  Its default objective is unregularized residual; explicit
TSVD/ridge settings and numerical rank losses are reported.  All support residuals
remain available for a floating-point residual-gap diagnostic.  This is not an
interval-arithmetic certificate and is not an assertion of beam-search success.

``recover_reciprocal_moments`` is the separate polynomial-size algebraic decoder
from the theory note.  It uses the supplied 2N positive moments only.  Finite
precision can cause it to fail, especially for nearly cancelling coefficients or
clustered transformed nodes; failures are returned, not replaced by truth-based
support choices or by the joint solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import gcd
from time import perf_counter
from typing import Sequence

import numpy as np
from numpy.polynomial import chebyshev as cheb

from .core import (
    centered_grid,
    gaussian_probe,
    position_to_index,
    recover_from_two_reflection_sums,
    translate_centered_sum,
    wrap_centered,
)


@dataclass(frozen=True)
class SpectralConfig:
    """Floating-point tolerances, not noise-derived statistical thresholds."""

    rank_rtol: float = 1e-11
    verify_rtol: float = 2e-7
    root_rtol: float = 2e-7
    merge_rtol: float = 2e-6
    channel_zero_rtol: float = 64 * np.finfo(float).eps
    solve_rcond: float | None = None
    max_condition: float = 1e14

    def __post_init__(self):
        for name in ("rank_rtol", "verify_rtol", "root_rtol", "merge_rtol", "channel_zero_rtol", "max_condition"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class SpectralRecovery:
    status: str
    message: str
    nodes: np.ndarray = field(default_factory=lambda: np.empty(0))
    coefficients: np.ndarray = field(default_factory=lambda: np.empty(0, complex))
    center_coefficient: float = 0.0
    reconstructed_moments: np.ndarray = field(default_factory=lambda: np.empty(0, complex))
    relative_residual: float = float("inf")
    mass_candidates: np.ndarray = field(default_factory=lambda: np.empty(0))
    candidate_count: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status.startswith("ok_")


def _rank(values: np.ndarray, rtol: float) -> int:
    singular = np.linalg.svd(values, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0
    return int(np.sum(singular > rtol * singular[0]))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), np.finfo(float).tiny))


def _second_kind_basis(count: int) -> np.ndarray:
    basis = np.zeros((count, count))
    if count:
        basis[0, 0] = 1
    if count > 1:
        basis[1, 1] = 2
    for degree in range(2, count):
        basis[degree, 1:] = 2 * basis[degree - 1, :-1]
        basis[degree] -= basis[degree - 2]
    return basis


def _real_roots(roots: np.ndarray, lower: float, config: SpectralConfig, *, strict=False):
    if not np.all(np.isfinite(roots)):
        return None, "nonfinite_roots"
    tolerance = config.root_rtol * np.maximum(1.0, np.abs(roots.real))
    if np.any(np.abs(roots.imag) > tolerance):
        return None, "nonreal_roots"
    nodes = np.sort(roots.real)
    if strict:
        if np.any(nodes <= lower):
            return None, "imaginary_node_not_above_one"
    elif np.any(nodes < lower - config.root_rtol * max(1.0, abs(lower))):
        return None, "node_outside_half_line"
    else:
        nodes = np.maximum(nodes, lower)
    if len(nodes) > 1 and np.any(np.diff(nodes) <= config.root_rtol * np.maximum(1.0, np.abs(nodes[:-1]))):
        return None, "unresolved_repeated_nodes"
    return nodes, "ok"


def _signed_prony(power: np.ndarray, bound: int, config: SpectralConfig):
    """Rank-adaptive signed Prony; all ranks come from the full Hankel block."""
    if not np.any(power):
        return np.empty(0), np.empty(0), "ok_zero", {}
    hankel = np.array([[power[i + j] for j in range(bound)] for i in range(bound)])
    rank = _rank(hankel, config.rank_rtol)
    info = {"rank": rank}
    if rank == 0:
        return None, None, "nonzero_moments_zero_rank", info
    leading = hankel[:rank, :rank]
    condition = float(np.linalg.cond(leading))
    info["leading_condition"] = condition
    if not np.isfinite(condition) or condition > config.max_condition:
        return None, None, "prony_leading_block_ill_conditioned", info
    try:
        coefficients = np.linalg.solve(leading, -power[rank : 2 * rank])
        roots = np.roots(np.r_[1.0, coefficients[::-1]])
        nodes, status = _real_roots(roots, 1.0, config, strict=True)
        if nodes is None:
            return None, None, status, info
        design = nodes[None, :] ** np.arange(len(power))[:, None]
        weights = np.linalg.lstsq(design, power, rcond=config.solve_rcond)[0]
        error = _relative_error(design @ weights, power)
        info["moment_residual"] = error
        if error > config.verify_rtol:
            return None, None, "prony_moment_verification_failed", info
        return nodes, weights, "ok", info
    except np.linalg.LinAlgError:
        return None, None, "prony_linear_algebra_failed", info


def _chebyshev_gram(moments: np.ndarray, size: int) -> np.ndarray:
    i, j = np.indices((size, size))
    return (moments[i + j] + moments[np.abs(i - j)]) / 2


def _real_completion(positive: np.ndarray, bound: int, config: SpectralConfig):
    """Complete the unobserved Chebyshev mass; never read measured lag zero."""
    if not np.any(positive):
        return [(np.empty(0), np.empty(0), 0.0, 0.0)], np.array([0.0]), []
    known = np.r_[0.0, positive]
    scale = np.r_[1.0, np.full(bound, np.sqrt(2.0))]
    gram0 = _chebyshev_gram(known, bound + 1)
    masses = -np.linalg.eigvalsh(gram0 * scale[:, None] * scale[None, :])
    accepted, failures = [], []
    for mass in masses:
        complete = known.copy()
        complete[0] = mass
        gram = _chebyshev_gram(complete, bound + 1)
        rank = _rank(gram, config.rank_rtol)
        if rank == 0 or rank > bound:
            failures.append("completion_invalid_rank")
            continue
        leading = gram[:rank, :rank]
        condition = float(np.linalg.cond(leading))
        if not np.isfinite(condition) or condition > config.max_condition:
            failures.append("completion_leading_block_ill_conditioned")
            continue
        try:
            polynomial = np.r_[np.linalg.solve(leading, -gram[:rank, rank]), 1.0]
            nodes, status = _real_roots(cheb.chebroots(polynomial), 1.0, config)
            if nodes is None:
                failures.append(status)
                continue
            design = cheb.chebvander(nodes, 2 * bound).T
            # The completed zeroth moment participates in this solve and is then
            # checked; no measured autocorrelation zero lag is used.
            weights = np.linalg.lstsq(design, complete, rcond=config.solve_rcond)[0]
            error = _relative_error((design @ weights)[1:], positive)
            if error > config.verify_rtol:
                failures.append("completion_moment_verification_failed")
                continue
            accepted.append((nodes, weights, float(mass), error))
        except np.linalg.LinAlgError:
            failures.append("completion_linear_algebra_failed")
    return accepted, masses, failures


def recover_reciprocal_moments(
    moments: Sequence[complex],
    sparsity_bound: int,
    *,
    config: SpectralConfig | None = None,
) -> SpectralRecovery:
    """Recover paired positive nodes from exactly the first ``2N`` moments.

    Returned ``nodes`` are ``t=cosh(y)>1`` and ``coefficients`` are the masses
    ``b`` at ``exp(y)``; the opposite-node mass is ``conj(b)``.  The real mass
    at node one is returned separately as ``center_coefficient``.
    """
    start = perf_counter()
    config = config or SpectralConfig()
    target = np.asarray(moments, dtype=complex)
    bound = int(sparsity_bound)
    if bound < 1 or target.shape != (2 * bound,) or not np.all(np.isfinite(target)):
        return SpectralRecovery("invalid_input", "Supply exactly 2N finite positive moments and N>=1.")
    magnitude = float(np.max(np.abs(target)))
    if magnitude == 0:
        return SpectralRecovery("ok_zero", "All supplied positive moments are zero.", reconstructed_moments=target.copy(), relative_residual=0.0, timings={"total": perf_counter() - start})
    normalized = target / magnitude
    real = normalized.real.copy()
    imag = normalized.imag.copy()
    diagnostics = {}
    for name, channel in (("real", real), ("imag", imag)):
        if np.max(np.abs(channel)) <= config.channel_zero_rtol:
            diagnostics[name + "_channel_treated_as_roundoff"] = bool(np.any(channel))
            channel[:] = 0
    try:
        phase_start = perf_counter()
        power = np.linalg.solve(_second_kind_basis(2 * bound), imag)
        inodes, iweights, istatus, idiagnostics = _signed_prony(power, bound, config)
        imaginary_seconds = perf_counter() - phase_start
        diagnostics["imaginary"] = idiagnostics
        if inodes is None:
            return SpectralRecovery("imaginary_" + istatus, "The imaginary signed-Prony stage failed its numerical checks.", timings={"total": perf_counter() - start}, diagnostics=diagnostics)
        phase_start = perf_counter()
        real_candidates, masses, failures = _real_completion(real, bound, config)
        real_seconds = perf_counter() - phase_start
        diagnostics["real_candidate_failures"] = failures
        if not real_candidates:
            return SpectralRecovery("no_valid_real_completion", "No spectral mass candidate passed the real half-line and moment checks.", mass_candidates=masses * magnitude, candidate_count=len(masses), timings={"total": perf_counter() - start}, diagnostics=diagnostics)
        valid = []
        for rnodes, rweights, mass, error in real_candidates:
            union = []
            center = 0.0
            for node, weight in zip(rnodes, rweights):
                if abs(node - 1.0) <= config.root_rtol:
                    center += float(weight)
                else:
                    union.append([float(node), float(weight) / 2, 0.0])
            for node, weight in zip(inodes, iweights):
                beta = float(weight / (2 * np.sqrt(node * node - 1)))
                candidates = [j for j, item in enumerate(union) if abs(item[0] - node) <= config.merge_rtol * max(1.0, abs(node))]
                if len(candidates) > 1:
                    union = None
                    break
                if candidates:
                    union[candidates[0]][2] += beta
                else:
                    union.append([float(node), 0.0, beta])
            if union is None or len(union) + int(abs(center) > config.channel_zero_rtol) > bound:
                continue
            union.sort(key=lambda item: item[0])
            nodes = np.array([item[0] for item in union])
            paired = np.array([complex(item[1], item[2]) for item in union])
            ell = np.arange(1, 2 * bound + 1)
            y = np.arccosh(nodes)
            prediction = np.full(2 * bound, center, complex)
            if len(nodes):
                prediction += np.exp(ell[:, None] * y) @ paired
                prediction += np.exp(-ell[:, None] * y) @ paired.conj()
            residual = _relative_error(prediction, normalized)
            if np.isfinite(residual) and residual <= config.verify_rtol:
                valid.append((residual, nodes, paired, center, prediction))
        if not valid:
            return SpectralRecovery("paired_reassembly_failed", "No merged reciprocal measure passed the orbit-count and full complex-moment checks.", mass_candidates=masses * magnitude, candidate_count=len(masses), timings={"total": perf_counter() - start}, diagnostics=diagnostics)
        valid.sort(key=lambda item: item[0])
        best = valid[0]
        for other in valid[1:]:
            same_nodes = len(other[1]) == len(best[1]) and np.allclose(other[1], best[1], rtol=config.merge_rtol, atol=config.merge_rtol)
            if not same_nodes or not np.allclose(other[2], best[2], rtol=10 * config.verify_rtol, atol=10 * config.verify_rtol) or abs(other[3] - best[3]) > 10 * config.verify_rtol:
                return SpectralRecovery("ambiguous_numerical_completions", "Distinct floating-point completions passed the configured tolerance; no unique output is asserted.", mass_candidates=masses * magnitude, candidate_count=len(masses), timings={"total": perf_counter() - start}, diagnostics=diagnostics)
        return SpectralRecovery(
            "ok_spectral", "Reciprocal measure passed all configured numerical checks.",
            nodes=best[1], coefficients=best[2] * magnitude,
            center_coefficient=float(best[3] * magnitude),
            reconstructed_moments=best[4] * magnitude, relative_residual=best[0],
            mass_candidates=masses * magnitude, candidate_count=len(masses),
            timings={"imaginary_prony": imaginary_seconds, "real_completion": real_seconds, "total": perf_counter() - start},
            diagnostics=diagnostics,
        )
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        return SpectralRecovery("spectral_linear_algebra_failed", str(exc), timings={"total": perf_counter() - start}, diagnostics=diagnostics)


@dataclass
class JointDictionary:
    n: int
    gamma: float
    scans: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    sparsity_bound: int
    lag_count: int
    minimum_separation: int
    probes: np.ndarray
    matrix: np.ndarray
    background_intensities: np.ndarray
    margin_valid: bool
    build_seconds: float

    @property
    def candidates(self) -> tuple[int, ...]:
        return self.candidate_positions

    @property
    def clean_lag_guarantee(self) -> bool:
        return self.minimum_separation >= self.lag_count


def build_joint_dictionary(
    n: int,
    gamma: float,
    scans: Sequence[int],
    candidate_positions: Sequence[int],
    sparsity_bound: int,
    *,
    lag_count: int | None = None,
    probe: np.ndarray | None = None,
    minimum_separation: int | None = None,
    enforce_margin: bool = True,
) -> JointDictionary:
    """Exact raw cross-term dictionary, also valid for known complex probes.

    Candidate sites and scans are integer coordinates. Invalid margins raise by
    default; ``enforce_margin=False`` permits an explicitly out-of-theory test.
    Spacing remains a support constraint in ``GlobalSupportSolver``.
    """
    start = perf_counter()
    grid = centered_grid(n)
    bound = int(sparsity_bound)
    lag_count = 2 * bound if lag_count is None else int(lag_count)
    if bound < 1 or not 1 <= lag_count <= n // 2:
        raise ValueError("Require N>=1 and 1<=lag_count<=M on an odd grid.")
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and positive")
    if any(int(r) != r for r in scans) or any(int(t) != t for t in candidate_positions):
        raise ValueError("Dictionary scans and candidate sites must be integers.")
    scans = tuple(int(wrap_centered(int(r), n)) for r in scans)
    if not scans:
        raise ValueError("At least one scan is required.")
    positions = tuple(sorted(int(wrap_centered(int(t), n)) for t in candidate_positions))
    if len(set(positions)) != len(positions) or not positions:
        raise ValueError("Candidate sites must be nonempty and distinct modulo n.")
    separation = lag_count if minimum_separation is None else int(minimum_separation)
    if separation < 0:
        raise ValueError("minimum_separation must be nonnegative")
    offsets = np.array([wrap_centered(np.asarray(positions) - r, n) for r in scans])
    margin_valid = bool(np.all(np.abs(offsets) <= n // 2 - lag_count))
    if enforce_margin and not margin_valid:
        raise ValueError("Candidate grid violates a scan's no-wrap margin.")
    if probe is None:
        probes = np.broadcast_to(gaussian_probe(n, gamma), (len(scans), n)).copy()
    else:
        probes = np.asarray(probe, dtype=complex)
        if probes.shape == (n,):
            probes = np.broadcast_to(probes, (len(scans), n)).copy()
        if probes.shape != (len(scans), n) or not np.all(np.isfinite(probes)):
            raise ValueError("probe must have shape (n,) or (number_of_scans,n) and be finite")
    lags = np.arange(1, lag_count + 1)[:, None]
    rows = []
    for wave, offset in zip(probes, offsets):
        left = wave[position_to_index(offset, n)][None, :] * np.conj(wave[position_to_index(offset[None, :] - lags, n)])
        right = wave[position_to_index(offset[None, :] + lags, n)] * np.conj(wave[position_to_index(offset, n)])[None, :]
        real_columns = left + right
        imag_columns = 1j * (left - right)
        rows.extend((np.column_stack((real_columns.real, imag_columns.real)), np.column_stack((real_columns.imag, imag_columns.imag))))
    return JointDictionary(
        n=n, gamma=float(gamma), scans=scans, candidate_positions=positions,
        sparsity_bound=bound, lag_count=lag_count, minimum_separation=separation,
        probes=probes, matrix=np.vstack(rows),
        background_intensities=np.abs(np.fft.fft(probes, axis=1)) ** 2,
        margin_valid=margin_valid, build_seconds=perf_counter() - start,
    )


def extract_raw_data(
    intensities: np.ndarray,
    dictionary: JointDictionary,
    *,
    background_intensities: np.ndarray | None = None,
) -> np.ndarray:
    """IFFT/background subtraction in exactly the dictionary's row order."""
    frames = np.asarray(intensities)
    shape = (len(dictionary.scans), dictionary.n)
    if frames.ndim == 1 and len(dictionary.scans) == 1:
        frames = frames[None, :]
    if frames.shape != shape or not np.all(np.isfinite(frames)) or np.iscomplexobj(frames) and np.any(frames.imag):
        raise ValueError(f"intensities must be finite real values of shape {shape}")
    background = dictionary.background_intensities if background_intensities is None else np.asarray(background_intensities)
    if background.shape == (dictionary.n,):
        background = np.broadcast_to(background, shape)
    if background.shape != shape or not np.all(np.isfinite(background)):
        raise ValueError("background_intensities has incompatible shape or nonfinite values")
    difference = np.fft.ifft(frames - background, axis=1)[:, 1 : dictionary.lag_count + 1]
    return np.concatenate([part for frame in difference for part in (frame.real, frame.imag)])


@dataclass
class JointRecovery:
    u: np.ndarray
    support: tuple[int, ...]
    coefficients: np.ndarray
    residual: float
    status: str
    timings: dict[str, float]
    candidate_count: int
    all_supports: tuple[tuple[int, ...], ...]
    all_residuals: np.ndarray
    objective: float
    minimum_residual: float
    certificate: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def residual_norm(self) -> float:
        return self.residual


@dataclass
class AlgebraicRecovery:
    u: np.ndarray
    support: tuple[int, ...]
    status: str
    message: str
    residual: float
    scan_recoveries: tuple[SpectralRecovery, ...]
    timings: dict[str, float]
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status.startswith("ok_")


def recover_two_scan_spectral(
    intensities: np.ndarray,
    probe: np.ndarray,
    gamma: float,
    scans: Sequence[int],
    sparsity_bound: int,
    *,
    config: SpectralConfig | None = None,
    grid_tolerance: float = 1e-3,
    coefficient_zero_rtol: float = 1e-7,
) -> AlgebraicRecovery:
    """Finite-grid wrapper for two exact algebraic reflection reconstructions.

    This wrapper checks the real Gaussian, node/grid agreement, orbit order,
    recovered spacing and the final raw-lag fit. It performs no candidate-support
    search and no fallback to global LS. Roundoff-level moment channels and orbit
    coefficients are explicitly reported in diagnostics.
    """
    start = perf_counter()
    config = config or SpectralConfig()
    frames = np.asarray(intensities)
    wave = np.asarray(probe)
    n = len(wave)
    empty = np.zeros(n, complex)
    recoveries = []
    diagnostics = {}

    def fail(status, message):
        return AlgebraicRecovery(empty.copy(), (), status, message, float("inf"), tuple(recoveries), {"total_online": perf_counter() - start}, diagnostics)

    if frames.shape != (2, n) or len(scans) != 2 or sparsity_bound < 1 or n % 2 != 1:
        return fail("invalid_input", "Require two frames on an odd grid and a positive sparsity bound.")
    if gamma <= 0 or not np.all(np.isfinite(frames)) or not np.all(np.isfinite(wave)):
        return fail("invalid_input", "Nonfinite data or invalid Gaussian width.")
    if not np.allclose(wave, gaussian_probe(n, gamma), rtol=1e-13, atol=1e-15):
        return fail("unsupported_probe", "This wrapper implements the real centered Gaussian theorem only.")
    if any(int(r) != r for r in scans):
        return fail("invalid_scan_grid", "Scan positions must be integers.")
    scans = tuple(int(r) for r in scans)
    displacement = (2 * (scans[1] - scans[0])) % n
    if displacement == 0 or n // gcd(n, displacement) <= 2 * sparsity_bound:
        return fail("orbit_order_condition_failed", "The translation orbit order does not exceed 2N.")
    lags = np.arange(1, 2 * sparsity_bound + 1)
    if len(lags) > n // 2:
        return fail("insufficient_grid_margin", "There are not 2N distinct positive lags.")
    background = np.abs(np.fft.fft(wave)) ** 2
    raw = np.fft.ifft(frames - background, axis=1)[:, lags]
    scales = np.exp(gamma * lags * lags)
    if not np.all(np.isfinite(scales)):
        return fail("moment_scaling_overflow", "Moment deweighting overflowed.")
    reflected = []
    floor = 64 * np.finfo(float).eps * max(float(np.linalg.norm(frames)), float(np.linalg.norm(background))) / np.sqrt(n)
    diagnostics["raw_roundoff_floor"] = floor
    diagnostics["zeroed_scan_moments"] = []
    for scan_index, scan in enumerate(scans):
        moments = raw[scan_index] * scales
        if np.linalg.norm(raw[scan_index]) <= floor:
            moments = np.zeros_like(moments)
            diagnostics["zeroed_scan_moments"].append(scan_index)
        recovered = recover_reciprocal_moments(moments, sparsity_bound, config=config)
        recoveries.append(recovered)
        if not recovered.success:
            return fail("scan_" + str(scan_index) + "_" + recovered.status, recovered.message)
        coordinates = np.arccosh(recovered.nodes) / (2 * gamma)
        snapped = np.rint(coordinates).astype(int)
        if np.any(np.abs(coordinates - snapped) > grid_tolerance) or np.any(snapped <= 0) or len(set(snapped.tolist())) != len(snapped):
            diagnostics["unsnapped_offsets"] = coordinates.tolist()
            return fail("spectral_nodes_off_grid", "Recovered spectral nodes cannot be assigned unambiguously to the integer grid.")
        if np.any(snapped > n // 2 - len(lags)):
            return fail("recovered_margin_failed", "A recovered reflected offset violates the Gaussian no-wrap margin.")
        q = np.zeros(n, complex)
        q[position_to_index(0, n)] = recovered.center_coefficient
        for x, mass in zip(snapped, recovered.coefficients):
            amplitude = mass * np.exp(2 * gamma * x * x)
            q[position_to_index(int(x), n)] = amplitude
            q[position_to_index(-int(x), n)] = amplitude.conjugate()
        reflected.append(translate_centered_sum(q, scan))
    # Passing n disables the old helper's top-N truncation. The upper-bound
    # formulation must allow zero objects and fewer than N active locations.
    integrated = recover_from_two_reflection_sums(reflected[0], reflected[1], scans[0], scans[1], n)
    u = integrated.perturbation.copy()
    threshold = coefficient_zero_rtol * float(np.max(np.abs(u)))
    u[np.abs(u) <= threshold] = 0
    support = tuple(int(x) for x in centered_grid(n)[np.abs(u) > 0])
    diagnostics["coefficient_roundoff_threshold"] = threshold
    diagnostics["unthresholded_orbit_residual"] = integrated.difference_residual
    if len(support) > sparsity_bound:
        return fail("orbit_sparsity_failed", "Integrated reflection sums exceed the specified sparsity upper bound.")
    if any(abs(wrap_centered(x - r, n)) > n // 2 - len(lags) for x in support for r in scans):
        return fail("object_margin_failed", "The integrated object violates a scan margin.")
    if any(abs(wrap_centered(a - b, n)) <= len(lags) for a, b in combinations(support, 2)):
        return fail("object_separation_failed", "The integrated object violates the required strict separation.")
    if support:
        model = build_joint_dictionary(n, gamma, scans, support, sparsity_bound, probe=wave)
        coefficients = u[position_to_index(np.asarray(support), n)]
        predicted = model.matrix @ np.r_[coefficients.real, coefficients.imag]
    else:
        predicted = np.zeros(4 * len(lags))
    target = np.concatenate([part for frame in raw for part in (frame.real, frame.imag)])
    residual = float(np.linalg.norm(predicted - target))
    if residual > config.verify_rtol * np.linalg.norm(target) + 4 * floor:
        diagnostics["final_raw_residual"] = residual
        return fail("final_raw_moment_verification_failed", "The integrated object does not fit the original raw clean lags within tolerance.")
    return AlgebraicRecovery(u, support, "ok_spectral_object", "Polynomial reconstruction, integer-grid checks and raw-data verification passed.", residual, tuple(recoveries), {"total_online": perf_counter() - start}, diagnostics)


class GlobalSupportSolver:
    """Reusable exhaustive legal-support solver for small candidate grids.

    No truth is accepted by this API. Each fixed-support fit is a cached SVD
    pseudoinverse. Column normalization is algebraically undone for ordinary LS;
    ridge uses physical coefficients and the unnormalized matrix.
    """

    def __init__(
        self,
        dictionary: JointDictionary,
        sparsity_bound: int,
        *,
        rcond: float | None = None,
        ridge: float = 0.0,
        max_supports: int = 100_000,
    ):
        start = perf_counter()
        self.dictionary = dictionary
        self.sparsity_bound = int(sparsity_bound)
        if self.sparsity_bound < 0 or self.sparsity_bound > dictionary.sparsity_bound:
            raise ValueError("Solver bound must lie between zero and the dictionary's sparsity bound.")
        if ridge < 0 or not np.isfinite(ridge) or rcond is not None and (rcond < 0 or not np.isfinite(rcond)):
            raise ValueError("ridge and rcond must be finite and nonnegative")
        self.ridge, self.rcond = float(ridge), rcond
        self.groups = []
        supports = []
        positions = np.asarray(dictionary.candidate_positions)
        k = len(positions)
        delta = np.abs(positions[:, None] - positions[None, :])
        distance = np.minimum(delta, dictionary.n - delta)
        legal_pair = distance > dictionary.minimum_separation
        cursor = 0
        self.rank_dropped_count = 0
        for size in range(self.sparsity_bound + 1):
            choices = []
            for support in combinations(range(k), size):
                if all(legal_pair[i, j] for i, j in combinations(support, 2)):
                    choices.append(support)
                    if cursor + len(choices) > max_supports:
                        raise ValueError("Legal-support count exceeds max_supports; global enumeration was not truncated.")
            if not choices:
                continue
            indices = np.asarray(choices, dtype=int).reshape(len(choices), size)
            columns = np.concatenate((indices, indices + k), axis=1)
            design = np.transpose(dictionary.matrix[:, columns], (1, 0, 2))
            if size:
                norms = np.linalg.norm(design, axis=1)
                scales = np.ones_like(norms) if ridge else np.where(norms > 0, norms, 1.0)
                normalized = design / scales[:, None, :]
                uu, singular, vv = np.linalg.svd(normalized, full_matrices=False)
                cutoff = (np.finfo(float).eps * max(normalized.shape[1:]) if rcond is None else rcond)
                keep = singular > cutoff * singular[:, :1]
                self.rank_dropped_count += int(np.sum(np.any(~keep, axis=1)))
                factors = np.zeros_like(singular)
                if ridge:
                    factors[keep] = singular[keep] / (singular[keep] ** 2 + ridge)
                else:
                    np.divide(1.0, singular, out=factors, where=keep)
                pinv = np.einsum("cjp,cp,crp->cjr", vv.transpose(0, 2, 1), factors, uu, optimize=True)
                pinv /= scales[:, :, None]
                ranks = np.sum(keep, axis=1)
            else:
                pinv = np.empty((len(choices), 0, dictionary.matrix.shape[0]))
                ranks = np.zeros(len(choices), dtype=int)
            end = cursor + len(choices)
            self.groups.append({"size": size, "indices": indices, "design": design, "pinv": pinv, "ranks": ranks, "slice": slice(cursor, end)})
            supports.extend(tuple(int(positions[j]) for j in support) for support in choices)
            cursor = end
        self.all_supports = tuple(supports)
        self.candidate_count = cursor
        self.sizes = np.array([len(support) for support in supports])
        self.precompute_seconds = perf_counter() - start

    def solve(
        self,
        data: np.ndarray,
        *,
        noise_bound: float | None = None,
        selection: str = "residual",
        tie_rtol: float = 64 * np.finfo(float).eps,
        tie_atol: float = 0.0,
    ) -> JointRecovery:
        """Evaluate all supports, retaining their physical residual norms.

        ``selection='discrepancy'`` chooses the smallest support fitting the given
        noise bound, then the smallest residual within that size. It is explicitly
        different from the global minimum-residual objective. No true sparsity is
        inferred from a ground-truth object.
        """
        start = perf_counter()
        data = np.asarray(data, dtype=float)
        if data.shape != (self.dictionary.matrix.shape[0],) or not np.all(np.isfinite(data)):
            raise ValueError("data must be a finite vector in dictionary row order")
        if selection not in ("residual", "discrepancy"):
            raise ValueError("selection must be 'residual' or 'discrepancy'")
        if noise_bound is not None and (noise_bound < 0 or not np.isfinite(noise_bound)):
            raise ValueError("noise_bound must be finite and nonnegative")
        if selection == "discrepancy" and noise_bound is None:
            raise ValueError("discrepancy selection requires noise_bound")
        residuals = np.empty(self.candidate_count)
        objectives = np.empty_like(residuals)
        fitted = []
        for group in self.groups:
            coefficients = np.einsum("cpr,r->cp", group["pinv"], data, optimize=True)
            prediction = np.einsum("crp,cp->cr", group["design"], coefficients, optimize=True)
            errors = prediction - data[None, :]
            squared = np.einsum("cr,cr->c", errors, errors)
            residuals[group["slice"]] = np.sqrt(squared)
            objectives[group["slice"]] = squared + self.ridge * np.einsum("cp,cp->c", coefficients, coefficients)
            fitted.append(coefficients)
        minimum = float(np.min(residuals))
        tolerance = tie_atol + tie_rtol * float(np.linalg.norm(data))
        status = "ok_global_ls"
        if self.ridge:
            status = "ok_global_ridge"
        elif self.rcond is not None:
            status = "ok_global_tsvd_configured"
        if self.rank_dropped_count:
            status += "_rank_truncated"
        if selection == "discrepancy":
            eligible = np.flatnonzero(residuals <= noise_bound + tolerance)
            if len(eligible):
                eligible = eligible[self.sizes[eligible] == np.min(self.sizes[eligible])]
                selected = int(eligible[np.argmin(objectives[eligible])])
                status = "ok_discrepancy" + ("_regularized" if self.ridge or self.rcond is not None else "")
                if self.rank_dropped_count:
                    status += "_rank_truncated"
            else:
                selected = int(np.argmin(objectives))
                status = "noise_bound_infeasible"
        else:
            if self.ridge:
                eligible = np.flatnonzero(objectives <= np.min(objectives) + tolerance * (2 * np.linalg.norm(data) + tolerance))
            else:
                eligible = np.flatnonzero(residuals <= minimum + tolerance)
            eligible = eligible[self.sizes[eligible] == np.min(self.sizes[eligible])]
            selected = int(eligible[np.argmin(objectives[eligible])])
        selected_coefficients = None
        for group, coefficients in zip(self.groups, fitted):
            if group["slice"].start <= selected < group["slice"].stop:
                selected_coefficients = coefficients[selected - group["slice"].start]
                break
        support = self.all_supports[selected]
        size = len(support)
        values = selected_coefficients[:size] + 1j * selected_coefficients[size:]
        u = np.zeros(self.dictionary.n, complex)
        u[position_to_index(np.asarray(support, dtype=int), self.dictionary.n)] = values
        certificate = {"kind": "floating_point_residual_gap", "is_formal_interval_certificate": False, "available": False}
        if noise_bound is not None:
            unsupported = self.ridge > 0 or self.rank_dropped_count > 0
            competing = [i for i, item in enumerate(self.all_supports) if not set(support).issubset(item)]
            runner = float(np.min(residuals[competing])) if competing else float("inf")
            certificate.update({
                "available": not unsupported,
                "reason": "regularized_or_rank_truncated_fits" if unsupported else "all_legal_supports_evaluated",
                "noise_bound": float(noise_bound), "feasible_support_count": int(np.sum(residuals <= noise_bound + tolerance)),
                "minimum_excluding_support_residual": runner,
                "gap_above_noise_bound": runner - noise_bound,
                "selected_support_required_by_residuals": bool(not unsupported and residuals[selected] <= noise_bound + tolerance and runner > noise_bound + tolerance),
                "selected_has_full_bound_size": size == self.sparsity_bound,
            })
        elapsed = perf_counter() - start
        return JointRecovery(
            u=u, support=support, coefficients=values,
            residual=float(residuals[selected]), status=status,
            timings={"solve": elapsed, "total_online": elapsed, "dictionary": self.dictionary.build_seconds, "precompute": self.precompute_seconds, "total_cold": elapsed + self.dictionary.build_seconds + self.precompute_seconds},
            candidate_count=self.candidate_count, all_supports=self.all_supports,
            all_residuals=residuals, objective=float(objectives[selected]), minimum_residual=minimum,
            certificate=certificate,
            diagnostics={"selection": selection, "ridge": self.ridge, "rcond": self.rcond, "rank_dropped_support_count": self.rank_dropped_count, "residual_selection_slack": float(residuals[selected] - minimum), "margin_valid": self.dictionary.margin_valid, "clean_lag_spacing_guarantee": self.dictionary.clean_lag_guarantee, "row_order": "per scan: real positive lags, imaginary positive lags"},
        )

    def solve_intensities(
        self,
        intensities: np.ndarray,
        *,
        background_intensities: np.ndarray | None = None,
        **kwargs,
    ) -> JointRecovery:
        start = perf_counter()
        data = extract_raw_data(intensities, self.dictionary, background_intensities=background_intensities)
        extraction = perf_counter() - start
        result = self.solve(data, **kwargs)
        result.timings["extraction"] = extraction
        result.timings["total_online"] += extraction
        result.timings["total_cold"] += extraction
        return result
