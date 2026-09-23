"""Model-adapted full-frame comparison methods.

These are specified variants, not verbatim reproductions of published algorithms:
``pie`` uses sequential Fourier-amplitude projections and known-probe PIE updates;
``htaf`` uses an amplitude-loss gradient, backtracking, and hard support projection;
``htwf`` uses the analogous full intensity-squared loss and its own gradient;
``full_linear`` uses affine intensity linearization, quadratic-term correction,
and fixed-support amplitude Gauss--Newton debiasing. All allow arbitrary complex
defect amplitudes and enforce the same candidate grid and separation constraint.
There is no phase-only projection, amplitude cap, or access to ground truth.

Every reported runtime includes dictionary construction, support preparation,
initialization, all restarts, and optimization. Small legal-support projection
tables and linearization factorizations are reused within one call only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter
from typing import Sequence

import numpy as np

from .core import position_to_index, wrap_centered


@dataclass(frozen=True)
class BaselineConfig:
    iterations: int = 400
    restarts: int = 8
    tolerance: float = 1e-7
    patience: int = 8
    learning_rate: float = 0.5
    threshold_after: int = 0
    pie_relaxation: float = 0.8
    initialization_scale: float = 1.0
    full_linear_steps: int = 3
    backtracking_steps: int = 16
    max_supports: int = 100_000


@dataclass
class BaselineResult:
    u: np.ndarray
    support: tuple[int, ...]
    status: str
    iterations: int
    runtime_seconds: float
    diagnostics: dict = field(default_factory=dict)


def _forward_design(probes, scans, candidates):
    """Flattened affine field ``reference + design @ coefficients``.

    Spatial arrays use centered order, exactly as :func:`core.exit_wave`.
    This also supports a distinct calibrated complex probe in each frame.
    """
    probes = np.asarray(probes, dtype=complex)
    if probes.ndim == 1:
        probes = np.broadcast_to(probes, (len(scans), len(probes)))
    n = probes.shape[-1]
    blocks = []
    for probe, scan in zip(probes, scans):
        offsets = position_to_index(np.asarray(candidates) - int(scan), n)
        atoms = np.zeros((len(candidates), n), complex)
        atoms[np.arange(len(candidates)), offsets] = probe[offsets]
        blocks.append(np.fft.fft(atoms, axis=1).T)
    return np.vstack(blocks), np.fft.fft(probes, axis=1).reshape(-1)


def _amplitude_loss_gradient(coefficients, design, reference, amplitudes):
    """Half mean amplitude error and its real-Euclidean complex gradient.

    For complex direction ``d``, the directional derivative is
    ``real(vdot(gradient, d))``. The chosen phase at a zero field is one;
    the amplitude loss is nonsmooth there. Rows of coefficients are restarts.
    """
    coefficients = np.atleast_2d(np.asarray(coefficients, complex))
    field = np.einsum("rk,mk->rm", coefficients, design, optimize=False) + reference[None, :]
    magnitude = np.abs(field)
    residual = magnitude - amplitudes[None, :]
    phase = np.ones_like(field)
    np.divide(field, magnitude, out=phase, where=magnitude > 0)
    gradient = np.einsum("rm,mk->rk", residual * phase, design.conj(), optimize=False) / len(reference)
    return 0.5 * np.mean(residual**2, axis=1), gradient


def _loss(coefficients, design, reference, amplitudes):
    field = np.einsum("rk,mk->rm", np.atleast_2d(coefficients), design, optimize=False) + reference[None, :]
    return 0.5 * np.mean((np.abs(field) - amplitudes[None, :]) ** 2, axis=1)


def _intensity_loss_gradient(coefficients, design, reference, intensities):
    """Half mean squared intensity error and real-Euclidean complex gradient.

    The gradient is ``2 D^H[(|F|^2-Y)F]/m``: the half in the loss cancels
    the squared-residual derivative, while differentiating |F|^2 supplies two.
    """
    coefficients = np.atleast_2d(np.asarray(coefficients, complex))
    field = np.einsum("rk,mk->rm", coefficients, design, optimize=False) + reference[None, :]
    residual = np.abs(field)**2 - intensities[None, :]
    gradient = 2 * np.einsum("rm,mk->rk", residual * field, design.conj(), optimize=False) / len(reference)
    return 0.5 * np.mean(residual**2, axis=1), gradient


def _intensity_loss(coefficients, design, reference, intensities):
    field = np.einsum("rk,mk->rm", np.atleast_2d(coefficients), design, optimize=False) + reference[None, :]
    return 0.5 * np.mean((np.abs(field)**2 - intensities[None, :])**2, axis=1)


class _SupportProjection:
    """Exact Euclidean projection onto all legal supports of size <= N."""

    def __init__(self, n, candidates, sparsity, minimum_separation, max_supports):
        candidates = np.asarray(candidates, int)
        distance = np.abs(candidates[:, None] - candidates[None, :])
        legal = np.minimum(distance, n - distance) > minimum_separation
        self.supports = []
        for size in range(sparsity + 1):
            for support in combinations(range(len(candidates)), size):
                if all(legal[i, j] for i, j in combinations(support, 2)):
                    self.supports.append(support)
                    if len(self.supports) > max_supports:
                        raise ValueError("legal support count exceeds max_supports")
        self.masks = np.zeros((len(self.supports), len(candidates)), float)
        for row, support in enumerate(self.supports):
            self.masks[row, list(support)] = 1.0

    def project(self, coefficients):
        coefficients = np.atleast_2d(coefficients)
        score = np.einsum("rk,ck->rc", np.abs(coefficients) ** 2, self.masks, optimize=False)
        chosen = np.argmax(score, axis=1)
        return coefficients * self.masks[chosen]


def _validate(intensities, probe, scans, candidates, sparsity, separation, config):
    values = np.asarray(intensities)
    if np.iscomplexobj(values) and np.any(values.imag):
        raise ValueError("intensities must be real")
    values = np.asarray(values.real, float)
    scans = tuple(int(r) for r in scans)
    probes = np.asarray(probe, complex)
    if probes.ndim not in (1, 2) or not scans:
        raise ValueError("probe must have one spatial axis and optional frame axis")
    n = probes.shape[-1]
    if probes.ndim == 1:
        probes = np.broadcast_to(probes, (len(scans), n)).copy()
    if probes.shape != (len(scans), n) or values.shape != probes.shape:
        raise ValueError("intensities/probes must have shape (number of frames, n)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(probes)):
        raise ValueError("measurements and probe must be finite")
    candidates = tuple(sorted(int(wrap_centered(t, n)) for t in candidates))
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("candidate positions must be nonempty and distinct")
    if not 0 <= sparsity <= len(candidates) or separation < 0:
        raise ValueError("invalid sparsity or minimum_separation")
    if min(config.iterations, config.restarts, config.patience,
           config.full_linear_steps, config.backtracking_steps, config.max_supports) < 1:
        raise ValueError("iteration/restart/count parameters must be positive")
    if not 0 <= config.threshold_after < config.iterations:
        raise ValueError("threshold_after must be smaller than iterations")
    parameters = (config.tolerance, config.learning_rate, config.pie_relaxation,
                  config.initialization_scale)
    if not all(np.isfinite(x) and x > 0 for x in parameters):
        raise ValueError("tolerances and step/initialization scales must be positive")
    if np.max(np.abs(probes)) == 0:
        raise ValueError("the probe cannot vanish everywhere")
    return values, probes, scans, candidates


def _initializations(design, reference, amplitudes, projection, indices,
                     n, sparsity, seed, config, initial_u):
    rng = np.random.default_rng(seed)
    contrast = np.linalg.norm(amplitudes - np.abs(reference))
    power = max(float(np.mean(np.abs(design) ** 2)), np.finfo(float).tiny)
    scale = contrast / np.sqrt(len(reference) * max(sparsity, 1) * power)
    scale = config.initialization_scale * max(float(scale), 0.1)
    coefficients = scale * (rng.normal(size=(config.restarts, len(indices)))
                            + 1j * rng.normal(size=(config.restarts, len(indices)))) / np.sqrt(2)
    coefficients = projection.project(coefficients)
    coefficients[0] = 0.0
    if initial_u is not None:
        initial = np.asarray(initial_u, complex)
        if initial.shape == (n,):
            initial = initial[None, :]
        if initial.ndim != 2 or initial.shape[1] != n or len(initial) > config.restarts:
            raise ValueError("initial_u must have shape (n,) or (at most restarts, n)")
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_u must be finite")
        coefficients[:len(initial)] = projection.project(initial[:, indices])
    return coefficients


def _iterative(method, coefficients, design, reference, amplitudes, intensities, probes,
               scans, indices, projection, config):
    n = probes.shape[-1]
    measured = amplitudes.reshape(len(scans), n)
    target = intensities if method == "htwf" else amplitudes
    objective_loss = _intensity_loss if method == "htwf" else _loss
    objective_gradient = _intensity_loss_gradient if method == "htwf" else _amplitude_loss_gradient
    best = projection.project(coefficients)
    best_loss = objective_loss(best, design, reference, target)
    active = np.ones(len(coefficients), bool)
    stale = np.zeros(len(coefficients), int)
    counts = np.zeros(len(coefficients), int)
    field_power = max(float(np.mean(np.abs(design) ** 2, axis=0).max()),
                      np.finfo(float).tiny)
    base_step = config.learning_rate / field_power
    baseline = np.abs(reference)
    if method == "htwf":
        # A data-only intensity scale sets the initial step; monotone
        # backtracking controls its actual use at every restart and iteration.
        intensity_scale = float(np.mean(np.abs(reference)**2) + np.mean(np.maximum(intensities, 0)))
        base_step /= max(2 * intensity_scale, np.finfo(float).tiny)
        baseline = baseline**2
    contrast = max(float(np.linalg.norm(target - baseline)),
                   np.finfo(float).eps * float(np.linalg.norm(target)),
                   np.finfo(float).tiny)
    history = [float(np.sqrt(2 * len(reference) * best_loss.min()) / contrast)]
    for iteration in range(config.iterations):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        current = coefficients[rows].copy()
        sparse_step = iteration >= config.threshold_after
        if iteration == config.threshold_after:
            current = projection.project(current)
            stale[rows] = 0
        if method == "pie":
            full = np.zeros((len(rows), n), complex)
            full[:, indices] = current
            for probe, scan, frame_amplitude in zip(probes, scans, measured):
                shifted = np.roll(1 + full, -scan, axis=1)
                exit_field = shifted * probe[None, :]
                fourier = np.fft.fft(exit_field, axis=1)
                magnitude = np.abs(fourier)
                phase = np.ones_like(fourier)
                np.divide(fourier, magnitude, out=phase, where=magnitude > 0)
                projected = np.fft.ifft(frame_amplitude[None, :] * phase, axis=1)
                denominator = max(float(np.max(np.abs(probe) ** 2)), np.finfo(float).tiny)
                correction = config.pie_relaxation * probe.conj()[None, :] * (projected - exit_field) / denominator
                full = np.roll(shifted + correction, scan, axis=1) - 1
                outside = np.ones(n, bool)
                outside[indices] = False
                full[:, outside] = 0
            current = full[:, indices]
            if sparse_step:
                current = projection.project(current)
        else:
            old_loss, gradient = objective_gradient(current, design, reference, target)
            unresolved = np.ones(len(rows), bool)
            accepted = current.copy()
            step = base_step
            for _ in range(config.backtracking_steps):
                pending = np.flatnonzero(unresolved)
                if not len(pending):
                    break
                trial = current[pending] - step * gradient[pending]
                if sparse_step:
                    trial = projection.project(trial)
                candidate_loss = objective_loss(trial, design, reference, target)
                take = candidate_loss <= old_loss[pending]
                accepted[pending[take]] = trial[take]
                unresolved[pending[take]] = False
                step *= 0.5
            current = accepted
        coefficients[rows] = current
        feasible = projection.project(current)
        value = objective_loss(feasible, design, reference, target)
        previous = best_loss[rows].copy()
        improved = value < previous
        best[rows[improved]] = feasible[improved]
        best_loss[rows[improved]] = value[improved]
        counts[rows] += 1
        relative = (previous - best_loss[rows]) / np.maximum(previous, np.finfo(float).tiny)
        stale[rows] = np.where(relative > config.tolerance, 0, stale[rows] + 1)
        residual = np.sqrt(2 * len(reference) * best_loss[rows])
        if sparse_step:
            active[rows] = (stale[rows] < config.patience) & (residual > config.tolerance * contrast)
        history.append(float(np.sqrt(2 * len(reference) * best_loss.min()) / contrast))
    winner = int(np.argmin(best_loss))
    return best[winner], int(counts[winner]), {
        "restart_losses": best_loss.tolist(), "restart_iterations": counts.tolist(),
        "best_restart": winner, "loss_history": history,
        "objective_half_mean_loss": float(best_loss[winner]),
        "iteration_limit_reached": bool(np.any(active)),
        "initialization": "zero plus general-complex projected random starts",
    }


def _linear_fits(matrix, projection):
    """Cache SVD pseudoinverses of all legal affine-intensity submodels."""
    k = matrix.shape[1] // 2
    groups = []
    rank_dropped = 0
    for size in sorted(set(map(len, projection.supports))):
        supports = [s for s in projection.supports if len(s) == size]
        indices = np.asarray(supports, int).reshape(len(supports), size)
        columns = np.concatenate((indices, indices + k), axis=1)
        blocks = np.transpose(matrix[:, columns], (1, 0, 2))
        if size:
            scale = np.linalg.norm(blocks, axis=1)
            scale = np.where(scale > 0, scale, 1.0)
            normalized = blocks / scale[:, None, :]
            uu, ss, vv = np.linalg.svd(normalized, full_matrices=False)
            keep = ss > np.finfo(float).eps * max(normalized.shape[1:]) * ss[:, :1]
            rank_dropped += int(np.sum(np.any(~keep, axis=1)))
            inverse = np.zeros_like(ss)
            np.divide(1, ss, out=inverse, where=keep)
            pinv = np.einsum("cjp,cp,crp->cjr", vv.transpose(0, 2, 1), inverse, uu, optimize=True)
            pinv /= scale[:, :, None]
        else:
            pinv = np.empty((1, 0, matrix.shape[0]))
        groups.append((indices, pinv))
    return groups, rank_dropped


def _gauss_newton(initial, design, reference, amplitudes, config, budget):
    support = np.flatnonzero(np.abs(initial) > 0)
    if not len(support):
        return initial.copy(), 0, float(_loss(initial, design, reference, amplitudes)[0])
    local = design[:, support]
    c = initial[support].copy()
    stale = 0
    completed = 0
    for iteration in range(budget):
        field = reference + np.einsum("mk,k->m", local, c, optimize=False)
        magnitude = np.abs(field)
        residual = magnitude - amplitudes
        objective = float(np.sum(residual**2))
        phase = np.ones_like(field)
        np.divide(field, magnitude, out=phase, where=magnitude > 0)
        jac = phase.conj()[:, None] * local
        jacobian = np.column_stack((jac.real, -jac.imag))
        delta_real = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        delta = delta_real[:len(support)] + 1j * delta_real[len(support):]
        accepted = False
        step = 1.0
        for _ in range(config.backtracking_steps):
            trial = c + step * delta
            error = np.abs(reference + np.einsum("mk,k->m", local, trial, optimize=False)) - amplitudes
            candidate = float(np.sum(error**2))
            if candidate <= objective:
                c = trial
                accepted = True
                break
            step *= 0.5
        completed = iteration + 1
        relative = (objective - candidate) / max(objective, np.finfo(float).tiny) if accepted else 0.0
        stale = 0 if relative > config.tolerance else stale + 1
        if stale >= config.patience or np.linalg.norm(step * delta) <= config.tolerance * max(1.0, float(np.linalg.norm(c))):
            break
    result = np.zeros_like(initial)
    result[support] = c
    return result, completed, float(_loss(result, design, reference, amplitudes)[0])


def _full_linear(values, design, reference, amplitudes, projection, starts,
                 config, groups, rank_dropped):
    k = design.shape[1]
    best = starts[int(np.argmin(_loss(starts, design, reference, amplitudes)))].copy()
    best_loss = float(_loss(best, design, reference, amplitudes)[0])
    counts, history = [], [best_loss]
    rounds = min(config.full_linear_steps, config.iterations)
    budget = max(1, config.iterations // rounds)
    for _ in range(rounds):
        defect_field = np.einsum("mk,k->m", design, best, optimize=False)
        target = values.reshape(-1) - np.abs(reference) ** 2 - np.abs(defect_field) ** 2
        candidates = []
        for indices, pinv in groups:
            size = indices.shape[1]
            real = np.einsum("cpm,m->cp", pinv, target, optimize=True)
            coefficients = np.zeros((len(indices), k), complex)
            if size:
                coefficients[np.arange(len(indices))[:, None], indices] = real[:, :size] + 1j * real[:, size:]
            candidates.append(coefficients)
        candidates = np.vstack(candidates + [starts, best[None, :]])
        losses = _loss(candidates, design, reference, amplitudes)
        selected, seen = [], set()
        for row in np.argsort(losses, kind="stable"):
            # Different initial coefficients on the same support can have distinct
            # local minima; retain them, but avoid exact duplicate starts.
            key = candidates[row].tobytes()
            if key not in seen:
                selected.append(int(row))
                seen.add(key)
            if len(selected) == config.restarts:
                break
        for row in selected:
            fitted, iterations, loss = _gauss_newton(candidates[row], design, reference, amplitudes, config, budget)
            counts.append(iterations)
            if loss < best_loss:
                best, best_loss = fitted, loss
        history.append(best_loss)
    return best, int(sum(counts)), {
        "linearization_rounds": rounds, "gauss_newton_iterations": counts,
        "linear_rank_truncated_supports": rank_dropped,
        "objective_half_mean_loss": best_loss,
        "loss_history": history, "iteration_count_kind": "sum across debias candidates",
        "initialization": "full-intensity affine fits and shared general-complex starts",
    }


def recover_baseline(method: str, intensities: np.ndarray, probe: np.ndarray,
                     scans: Sequence[int], candidate_positions: Sequence[int],
                     sparsity: int, minimum_separation: int, *, seed: int = 0,
                     config: BaselineConfig | None = None,
                     initial_u: np.ndarray | None = None) -> BaselineResult:
    """Recover from complete frames using explicitly adapted sparse baselines.

    Separation is strict: two selected cyclic positions must be farther apart
    than ``minimum_separation``. The amplitude target is sqrt(max(intensity, 0));
    original signed observations remain available to the intensity linearization.
    ``initial_u`` is an optional caller-provided initialization, not an oracle.
    For iterative methods ``iterations`` reports the selected restart's work;
    full_linear reports the sum of its debias iterations, also itemized in diagnostics.
    """
    start = perf_counter()
    if method not in ("pie", "htwf", "htaf", "full_linear"):
        raise ValueError("method must be pie, htwf, htaf, or full_linear")
    config = config or BaselineConfig()
    values, probes, scans, candidates = _validate(intensities, probe, scans,
        candidate_positions, int(sparsity), int(minimum_separation), config)
    n = probes.shape[-1]
    indices = position_to_index(np.asarray(candidates), n)
    projection = _SupportProjection(n, candidates, int(sparsity),
                                    int(minimum_separation), config.max_supports)
    design, reference = _forward_design(probes, scans, candidates)
    amplitudes = np.sqrt(np.maximum(values, 0)).reshape(-1)
    starts = _initializations(design, reference, amplitudes, projection, indices,
                             n, int(sparsity), seed, config, initial_u)
    linear_svd_setup = 0.0
    if method == "full_linear":
        linear_start = perf_counter()
        cross = reference.conj()[:, None] * design
        matrix = np.column_stack((2 * cross.real, -2 * cross.imag))
        linear_groups, rank_dropped = _linear_fits(matrix, projection)
        linear_svd_setup = perf_counter() - linear_start
    setup = perf_counter() - start
    if method == "full_linear":
        coefficients, iterations, diagnostics = _full_linear(values, design, reference,
            amplitudes, projection, starts, config, linear_groups, rank_dropped)
    else:
        coefficients, iterations, diagnostics = _iterative(method, starts, design,
            reference, amplitudes, values.reshape(-1), probes, scans, indices, projection, config)
    u = np.zeros(n, complex)
    u[indices] = coefficients
    support = tuple(candidates[j] for j in np.flatnonzero(np.abs(coefficients) > 0))
    residual = float(np.sqrt(2 * len(reference) * _loss(coefficients, design, reference, amplitudes)[0]))
    intensity_residual = float(np.sqrt(2 * len(reference) * _intensity_loss(coefficients, design, reference, values.reshape(-1))[0]))
    diagnostics.update({"method_variant": {
        "pie": "known-probe PIE-style amplitude projections with legal sparse object projection",
        "htaf": "hard-thresholded amplitude-loss gradient with backtracking",
        "htwf": "hard-thresholded intensity-loss Wirtinger-style gradient with backtracking",
        "full_linear": "full-intensity affine fits plus quadratic correction and amplitude Gauss-Newton debiasing",
    }[method], "amplitude_residual": residual, "setup_seconds": setup,
        "intensity_residual": intensity_residual,
        "selection_objective": "intensity_squared_error" if method == "htwf" else "amplitude_squared_error",
        "linear_svd_setup_seconds": linear_svd_setup,
        "legal_support_count": len(projection.supports), "seed": int(seed),
        "restarts": config.restarts, "minimum_separation": int(minimum_separation),
        "negative_detector_entries": int(np.sum(values < 0)),
        "candidate_positions": list(candidates), "general_complex_coefficients": True,
        "caller_supplied_initialization": initial_u is not None,
        "timing_includes_setup_and_all_restarts": True})
    status = "ok_iteration_limit" if diagnostics.get("iteration_limit_reached", False) else "ok_finished"
    if not np.all(np.isfinite(u)) or not np.isfinite(residual):
        status = "failed_nonfinite"
    runtime = perf_counter() - start
    diagnostics["optimization_seconds"] = runtime - setup
    return BaselineResult(u, support, status, iterations, runtime, diagnostics)
