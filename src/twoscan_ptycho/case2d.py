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


def canonical_reflection(point: Point) -> Point:
    point = (int(point[0]), int(point[1]))
    reflected = (-point[0], -point[1])
    return min(point, reflected)


def candidate_reflection_orbits(offset_bound: int) -> tuple[Point, ...]:
    result = {
        canonical_reflection((p1, p2))
        for p1 in range(-offset_bound, offset_bound + 1)
        for p2 in range(-offset_bound, offset_bound + 1)
        if (p1, p2) != (0, 0)
    }
    return tuple(sorted(result))


def _group_dictionaries(
    gamma: float, lag_count: int, orbits: Sequence[Point]
) -> tuple[Array, Array, Array]:
    lags = positive_lag_box(lag_count).astype(float)
    m = len(lags)
    real_dictionary = np.zeros((2 * m, len(orbits)), dtype=float)
    imag_dictionary = np.zeros_like(real_dictionary)
    for column, point in enumerate(orbits):
        x = np.asarray(point, dtype=float)
        weight = np.exp(-2.0 * gamma * float(x @ x))
        exponent = 2.0 * gamma * (lags @ x)
        real_dictionary[:m, column] = 2.0 * weight * np.cosh(exponent)
        imag_dictionary[m:, column] = 2.0 * weight * np.sinh(exponent)
    row_weights = np.exp(-gamma * np.sum(lags * lags, axis=1))
    row_weights = np.concatenate((row_weights, row_weights))
    return (
        real_dictionary * row_weights[:, None],
        imag_dictionary * row_weights[:, None],
        row_weights,
    )


@dataclass(frozen=True)
class ReflectionRecovery2D:
    orbits: tuple[Point, ...]
    coefficients: tuple[complex, ...]
    residual_norm: float


def recover_reflection_sum_2d(
    moments: Array,
    gamma: float,
    lag_count: int,
    sparsity: int,
    orbits: Sequence[Point],
) -> ReflectionRecovery2D:
    d_real, d_imag, weights = _group_dictionaries(gamma, lag_count, orbits)
    target = np.concatenate((np.real(moments), np.imag(moments))) * weights
    real_norms = np.maximum(np.linalg.norm(d_real, axis=0), 1e-15)
    imag_norms = np.maximum(np.linalg.norm(d_imag, axis=0), 1e-15)
    selected: list[int] = []
    residual = target.copy()
    coefficients = np.empty(0, dtype=float)
    for _ in range(sparsity):
        real_scores = (d_real.T @ residual) / real_norms
        imag_scores = (d_imag.T @ residual) / imag_norms
        scores = np.hypot(real_scores, imag_scores)
        if selected:
            scores[np.asarray(selected, dtype=int)] = -np.inf
        selected.append(int(np.argmax(scores)))
        design = np.column_stack(
            (d_real[:, selected], d_imag[:, selected])
        )
        coefficients = np.linalg.lstsq(design, target, rcond=1e-12)[0]
        residual = target - design @ coefficients
    count = len(selected)
    complex_coefficients = coefficients[:count] + 1j * coefficients[count:]
    return ReflectionRecovery2D(
        orbits=tuple(orbits[index] for index in selected),
        coefficients=tuple(complex(value) for value in complex_coefficients),
        residual_norm=float(np.linalg.norm(residual)),
    )


def recover_reflection_sum_esprit_2d(
    moments: Array,
    gamma: float,
    lag_count: int,
    sparsity: int,
    offset_bound: int,
) -> ReflectionRecovery2D:
    """Recover a reciprocal node set by a paired matrix-pencil construction.

    The first coordinate is obtained from the shift-invariant column space of the
    moment matrix.  After snapping those nodes to the known integer grid, their
    coefficient sequences across the second lag coordinate reveal the paired second
    coordinates.  The case-study sampler avoids repeated absolute first coordinates,
    which is the standard simple-spectrum condition for this implementation.
    """

    rank = 2 * int(sparsity)
    matrix = np.asarray(moments, dtype=np.complex128).reshape(
        lag_count, lag_count
    )
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    subspace = u[:, :rank]
    pencil = np.linalg.lstsq(subspace[:-1], subspace[1:], rcond=1e-12)[0]
    first_nodes = np.linalg.eigvals(pencil)
    first_nodes = np.maximum(np.abs(first_nodes), 1e-12)
    first_coordinates = np.rint(np.log(first_nodes) / (2.0 * gamma)).astype(int)
    first_coordinates = np.clip(first_coordinates, -offset_bound, offset_bound)
    if len(set(int(value) for value in first_coordinates)) != rank:
        return ReflectionRecovery2D(tuple(), tuple(), float("inf"))

    orders = np.arange(1, lag_count + 1, dtype=float)[:, None]
    snapped_first_nodes = np.exp(
        2.0 * gamma * first_coordinates.astype(float)
    )
    vandermonde = snapped_first_nodes[None, :] ** orders
    coefficient_sequences = np.linalg.lstsq(
        vandermonde, matrix, rcond=1e-12
    )[0]
    second_nodes = []
    for sequence in coefficient_sequences:
        denominator = float(np.vdot(sequence[:-1], sequence[:-1]).real)
        if denominator <= 1e-24:
            second_nodes.append(np.nan)
        else:
            second_nodes.append(np.vdot(sequence[:-1], sequence[1:]) / denominator)
    second_nodes = np.asarray(second_nodes)
    if not np.all(np.isfinite(second_nodes)):
        return ReflectionRecovery2D(tuple(), tuple(), float("inf"))
    second_coordinates = np.rint(
        np.log(np.maximum(np.abs(second_nodes), 1e-12)) / (2.0 * gamma)
    ).astype(int)
    second_coordinates = np.clip(second_coordinates, -offset_bound, offset_bound)

    nodes = tuple(
        (int(first), int(second))
        for first, second in zip(first_coordinates, second_coordinates)
    )
    if len(set(nodes)) != rank:
        return ReflectionRecovery2D(tuple(), tuple(), float("inf"))

    lags = positive_lag_box(lag_count).astype(float)
    design = np.column_stack(
        [
            np.exp(
                2.0
                * gamma
                * (lags @ np.asarray(point, dtype=float))
            )
            for point in nodes
        ]
    )
    node_weights = np.linalg.lstsq(
        design, np.asarray(moments).reshape(-1), rcond=1e-12
    )[0]
    orbit_values: dict[Point, list[complex]] = {}
    for point, value in zip(nodes, node_weights):
        orbit = canonical_reflection(point)
        probe_weight = np.exp(-2.0 * gamma * float(np.dot(point, point)))
        coefficient = complex(value / probe_weight)
        if point != orbit:
            coefficient = np.conj(coefficient)
        orbit_values.setdefault(orbit, []).append(coefficient)
    collapsed = [
        (orbit, complex(np.mean(values)))
        for orbit, values in orbit_values.items()
    ]
    collapsed.sort(key=lambda item: abs(item[1]), reverse=True)
    collapsed = collapsed[:sparsity]
    fitted = design @ node_weights
    return ReflectionRecovery2D(
        orbits=tuple(item[0] for item in collapsed),
        coefficients=tuple(item[1] for item in collapsed),
        residual_norm=float(
            np.linalg.norm(fitted - np.asarray(moments).reshape(-1))
        ),
    )


def _offset(position: Point, scan: Point, n: int) -> Point:
    return (
        int(wrap_centered(position[0] - scan[0], n)),
        int(wrap_centered(position[1] - scan[1], n)),
    )


def match_reflection_supports(
    recoveries: Sequence[ReflectionRecovery2D],
    scans: Sequence[Point],
    sparsity: int,
    n: int,
    object_bound: int,
) -> tuple[Point, ...]:
    magnitude_maps = [
        {orbit: abs(value) for orbit, value in zip(result.orbits, result.coefficients)}
        for result in recoveries
    ]
    scored: list[tuple[float, Point]] = []
    for p1 in range(-object_bound, object_bound + 1):
        for p2 in range(-object_bound, object_bound + 1):
            point = (p1, p2)
            magnitudes = []
            for scan, values in zip(scans, magnitude_maps):
                orbit = canonical_reflection(_offset(point, scan, n))
                magnitudes.append(values.get(orbit, 0.0))
            scored.append((float(np.prod(magnitudes)), point))
    scored.sort(key=lambda item: item[0], reverse=True)
    return tuple(point for _, point in scored[:sparsity])


def fit_global_amplitudes(
    moments_by_scan: Sequence[Array],
    positions: Sequence[Point],
    scans: Sequence[Point],
    gamma: float,
    lag_count: int,
    n: int,
    *,
    physical_phase_projection: bool = True,
) -> tuple[complex, ...]:
    lags = positive_lag_box(lag_count).astype(float)
    row_weights = np.exp(-gamma * np.sum(lags * lags, axis=1))
    target_blocks = []
    real_columns: list[list[Array]] = [[] for _ in positions]
    imag_columns: list[list[Array]] = [[] for _ in positions]
    for moments, scan in zip(moments_by_scan, scans):
        target_blocks.extend((np.real(moments) * row_weights, np.imag(moments) * row_weights))
        for index, position in enumerate(positions):
            x = np.asarray(_offset(position, scan, n), dtype=float)
            weight = np.exp(-2.0 * gamma * float(x @ x))
            exponent = 2.0 * gamma * (lags @ x)
            c = 2.0 * weight * np.cosh(exponent) * row_weights
            s = 2.0 * weight * np.sinh(exponent) * row_weights
            real_columns[index].extend((c, np.zeros_like(c)))
            imag_columns[index].extend((np.zeros_like(s), s))
    target = np.concatenate(target_blocks)
    d_real = np.column_stack([np.concatenate(parts) for parts in real_columns])
    d_imag = np.column_stack([np.concatenate(parts) for parts in imag_columns])
    design = np.column_stack((d_real, d_imag))
    coefficients = np.linalg.lstsq(design, target, rcond=1e-12)[0]
    count = len(positions)
    amplitudes = coefficients[:count] + 1j * coefficients[count:]
    if physical_phase_projection:
        amplitudes = np.exp(1j * np.angle(1.0 + amplitudes)) - 1.0
    return tuple(complex(value) for value in amplitudes)


@dataclass(frozen=True)
class CaseRecovery2D:
    positions: tuple[Point, ...]
    amplitudes: tuple[complex, ...]
    reflection_recoveries: tuple[ReflectionRecovery2D, ...]


def recover_case_2d(
    intensities: Sequence[Array],
    scans: Sequence[Point],
    assumed_gamma: float,
    lag_count: int,
    sparsity: int,
    n: int,
    offset_bound: int,
    object_bound: int,
) -> CaseRecovery2D:
    probe = gaussian_probe_2d(n, assumed_gamma)
    moments = [
        clean_lag_moments_2d(intensity, probe, assumed_gamma, lag_count)
        for intensity in intensities
    ]
    reflected = tuple(
        recover_reflection_sum_esprit_2d(
            value, assumed_gamma, lag_count, sparsity, offset_bound
        )
        for value in moments
    )
    positions = match_reflection_supports(
        reflected, scans, sparsity, n, object_bound
    )
    amplitudes = fit_global_amplitudes(
        moments, positions, scans, assumed_gamma, lag_count, n
    )
    return CaseRecovery2D(positions, amplitudes, reflected)


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


def profiled_support_score_2d(
    target: Array,
    model: CalibratedSignatureModel2D,
    recovered_positions: Sequence[Point],
) -> Array:
    """Profile the data-fit objective over each possible support replacement.

    For every candidate grid point, one recovered atom is replaced by that point and
    the nonnegative severities are refitted.  The best replacement over all support
    slots is reported as a decibel fit score.  Unlike a support-marker schematic, this
    map retains the noise texture, sidelobes, and localization basins of the actual
    Poisson trial.
    """

    point_to_index = {point: index for index, point in enumerate(model.points)}
    selected = [point_to_index[tuple(point)] for point in recovered_positions]
    target = np.asarray(target, dtype=float)
    target_energy = float(target @ target)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        full_cross = model.dictionary.T @ target
    if not np.all(np.isfinite(full_cross)):
        raise FloatingPointError("non-finite profiled cross-correlation")
    residuals = np.full(len(model.points), np.inf, dtype=float)
    for slot in range(len(selected)):
        for candidate in range(len(model.points)):
            trial = selected.copy()
            trial[slot] = candidate
            if len(set(trial)) != len(trial):
                continue
            index = np.asarray(trial, dtype=int)
            _, residual = _nnls_from_gram(
                model.gram[np.ix_(index, index)],
                full_cross[index],
                target_energy,
            )
            residuals[candidate] = min(residuals[candidate], residual)
    relative_power = residuals / max(target_energy, 1e-30)
    scores = -10.0 * np.log10(np.maximum(relative_power, 1e-15))
    side = int(round(np.sqrt(len(model.points))))
    if side * side != len(model.points):
        raise ValueError("profiled support map requires a square candidate grid")
    return scores.reshape(side, side)


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
