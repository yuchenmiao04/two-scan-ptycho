#!/usr/bin/env python3
"""Noise Monte Carlo for the two-scan structured recovery in Fig. 2(a)."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from twoscan_ptycho.core import (
    add_intensity_awgn,
    affine_object,
    diffraction_intensity,
    gaussian_probe,
    random_amplitudes,
    recover_from_two_intensities,
    sample_separated_positions,
    sparse_perturbation,
    support_metrics,
)


def trial_configuration(
    n_atoms: int, *, lag_multiplier: int = 2, gamma_scale: float = 5.0
) -> tuple[int, int, float]:
    lag_count = lag_multiplier * n_atoms
    location_bound = max(
        10, int(np.ceil((n_atoms - 1) * (lag_count + 2) / 2.0)) + 5
    )
    gamma = gamma_scale / (location_bound * location_bound)
    return lag_count, location_bound, gamma


def run_trial(
    n_atoms: int,
    snr_db: float | None,
    trial: int,
    rng: np.random.Generator,
    beam_width: int,
    solver: str,
    row_scaling: str,
    minimum_amplitude: float,
    lag_multiplier: int,
    gamma_scale: float,
) -> dict[str, object]:
    n = 257
    scans = (0, 1)
    lag_count, location_bound, gamma = trial_configuration(
        n_atoms, lag_multiplier=lag_multiplier, gamma_scale=gamma_scale
    )
    positions = sample_separated_positions(
        n,
        n_atoms,
        minimum_distance=lag_count,
        scans=scans,
        margin=lag_count,
        rng=rng,
        location_bound=location_bound,
    )
    amplitudes = random_amplitudes(
        n_atoms,
        rng,
        physical_phase=True,
        minimum_magnitude=minimum_amplitude,
    )
    probe = gaussian_probe(n, gamma)
    z = affine_object(n, positions, amplitudes)
    intensities = [diffraction_intensity(z, probe, scan) for scan in scans]
    clean_intensities = [values.copy() for values in intensities]
    effective_snr = float("inf")
    if snr_db is not None:
        intensities = [add_intensity_awgn(y, snr_db, rng) for y in intensities]
        baseline = np.abs(np.fft.fft(probe)) ** 2
        signal_norm = np.sqrt(
            sum(np.linalg.norm(y - baseline) ** 2 for y in clean_intensities)
        )
        noise_norm = np.sqrt(
            sum(
                np.linalg.norm(noisy - clean) ** 2
                for noisy, clean in zip(intensities, clean_intensities)
            )
        )
        effective_snr = float(
            20.0
            * np.log10(
                max(signal_norm, np.finfo(float).eps)
                / max(noise_norm, np.finfo(float).eps)
            )
        )
    max_offset = location_bound + 2
    recovered = recover_from_two_intensities(
        intensities[0],
        intensities[1],
        probe,
        gamma,
        scans[0],
        scans[1],
        n_atoms,
        max_offset,
        lag_count=lag_count,
        solver=solver,
        beam_width=beam_width,
        row_scaling=row_scaling,
    )
    metrics = support_metrics(
        sparse_perturbation(n, positions, amplitudes),
        recovered.result.perturbation,
        n_atoms,
    )
    return {
        "n_atoms": n_atoms,
        "snr_db": "inf" if snr_db is None else snr_db,
        "trial": trial,
        "n": n,
        "lag_count": lag_count,
        "location_bound": location_bound,
        "gamma": gamma,
        "solver": solver,
        "row_scaling": row_scaling,
        "minimum_amplitude": minimum_amplitude,
        "effective_signal_snr_db": effective_snr,
        "positions": ";".join(str(x) for x in positions),
        **metrics,
        "scan1_condition_real": recovered.scan_recoveries[0].condition_real,
        "scan1_condition_imag": recovered.scan_recoveries[0].condition_imag,
        "scan2_condition_real": recovered.scan_recoveries[1].condition_real,
        "scan2_condition_imag": recovered.scan_recoveries[1].condition_imag,
        "scan1_residual": recovered.scan_recoveries[0].residual_norm,
        "scan2_residual": recovered.scan_recoveries[1].residual_norm,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted(
        set((int(row["n_atoms"]), str(row["snr_db"]), str(row["solver"])) for row in rows)
    )
    for n_atoms, snr_db, solver in keys:
        group = [
            row
            for row in rows
            if int(row["n_atoms"]) == n_atoms
            and str(row["snr_db"]) == snr_db
            and str(row["solver"]) == solver
        ]
        summaries.append(
            {
                "n_atoms": n_atoms,
                "snr_db": snr_db,
                "solver": solver,
                "trials": len(group),
                "support_success_rate": float(
                    np.mean([float(row["support_exact"]) for row in group])
                ),
                "support_f1_mean": float(
                    np.mean([float(row["support_f1"]) for row in group])
                ),
                "effective_signal_snr_db_mean": float(
                    np.mean(
                        [float(row["effective_signal_snr_db"]) for row in group]
                    )
                ),
                "nrmse_median": float(
                    np.median([float(row["nrmse"]) for row in group])
                ),
                "nrmse_q90": float(
                    np.quantile([float(row["nrmse"]) for row in group], 0.9)
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--max-atoms", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=2048)
    parser.add_argument(
        "--row-scaling",
        choices=("noise", "dictionary", "none"),
        default="noise",
    )
    parser.add_argument("--minimum-amplitude", type=float, default=0.75)
    parser.add_argument("--lag-multiplier", type=int, default=2)
    parser.add_argument("--gamma-scale", type=float, default=5.0)
    parser.add_argument("--solvers", nargs="+", choices=("beam", "omp"), default=("beam",))
    parser.add_argument(
        "--snr-db",
        nargs="+",
        default=("inf", "120", "110", "100", "90", "80", "70", "60"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/noise")
    )
    args = parser.parse_args()
    snr_values = [None if value.lower() == "inf" else float(value) for value in args.snr_db]
    rng = np.random.default_rng(args.seed)
    rows = [
        run_trial(
            n_atoms,
            snr_db,
            trial,
            rng,
            args.beam_width,
            solver,
            args.row_scaling,
            args.minimum_amplitude,
            args.lag_multiplier,
            args.gamma_scale,
        )
        for n_atoms in range(1, args.max_atoms + 1)
        for snr_db in snr_values
        for solver in args.solvers
        for trial in range(args.trials)
    ]
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "trials.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "configuration": {
            "trials": args.trials,
            "max_atoms": args.max_atoms,
            "beam_width": args.beam_width,
            "row_scaling": args.row_scaling,
            "minimum_amplitude": args.minimum_amplitude,
            "lag_multiplier": args.lag_multiplier,
            "gamma_scale": args.gamma_scale,
            "solvers": list(args.solvers),
            "snr_db": args.snr_db,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "summary": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"wrote {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
