#!/usr/bin/env python3
"""Paired comparison of the clean-lag decoder with full-data iterative baselines."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from twoscan_ptycho.full_data import recover_full_data_linearized
from twoscan_ptycho.iterative import (
    fienup_error_reduction,
    hard_thresholded_wirtinger_flow,
)
from .run_noise import trial_configuration
from twoscan_ptycho.core import (
    affine_object,
    diffraction_intensity,
    gaussian_probe,
    random_amplitudes,
    recover_from_two_intensities,
    sample_separated_positions,
    sparse_perturbation,
    support_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--snr-db", nargs="+", default=("60", "70", "80", "90", "100", "110", "120"))
    parser.add_argument("--atoms", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--minimum-amplitude", type=float, default=0.75)
    parser.add_argument("--pie-iterations", type=int, default=800)
    parser.add_argument("--pie-restarts", type=int, default=32)
    parser.add_argument("--wf-iterations", type=int, default=1200)
    parser.add_argument("--wf-restarts", type=int, default=32)
    parser.add_argument("--wf-learning-rate", type=float, default=0.05)
    parser.add_argument("--wf-threshold-after", type=int, default=450)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison"),
    )
    args = parser.parse_args()
    snr_values = [None if value.lower() == "inf" else float(value) for value in args.snr_db]
    rng = np.random.default_rng(args.seed)
    n = 257
    scans = (0, 1)
    lag_count, location_bound, gamma = trial_configuration(args.atoms)
    candidates = tuple(range(-location_bound, location_bound + 1))
    probe = gaussian_probe(n, gamma)
    rows: list[dict[str, object]] = []

    instances: list[dict[str, object]] = []
    for _ in range(args.trials):
        positions = sample_separated_positions(
            n,
            args.atoms,
            minimum_distance=lag_count,
            scans=scans,
            margin=lag_count,
            rng=rng,
            location_bound=location_bound,
        )
        amplitudes = random_amplitudes(
            args.atoms,
            rng,
            physical_phase=True,
            minimum_magnitude=args.minimum_amplitude,
        )
        z = affine_object(n, positions, amplitudes)
        clean = np.stack(
            [diffraction_intensity(z, probe, scan) for scan in scans]
        )
        instances.append(
            {
                "positions": positions,
                "truth": sparse_perturbation(n, positions, amplitudes),
                "clean": clean,
                "noise_direction": rng.standard_normal(clean.shape),
            }
        )

    for snr_db in snr_values:
        for trial, instance in enumerate(instances):
            positions = instance["positions"]
            truth = np.asarray(instance["truth"])
            clean = np.asarray(instance["clean"])
            noisy = clean.copy()
            if snr_db is not None:
                rms = np.linalg.norm(clean, axis=1) / np.sqrt(n)
                sigma = rms * 10.0 ** (-snr_db / 20.0)
                noisy = np.maximum(
                    clean + sigma[:, None] * np.asarray(instance["noise_direction"]),
                    0.0,
                )

            recoveries: list[tuple[str, object, float]] = []
            started = time.perf_counter()
            proposed = recover_from_two_intensities(
                noisy[0],
                noisy[1],
                probe,
                gamma,
                scans[0],
                scans[1],
                args.atoms,
                location_bound + 2,
                lag_count=lag_count,
                solver="beam",
                beam_width=args.beam_width,
                row_scaling="noise",
            )
            recoveries.append(("proposed", proposed.result, time.perf_counter() - started))

            started = time.perf_counter()
            full_data = recover_full_data_linearized(
                noisy,
                probe,
                scans,
                args.atoms,
                candidates,
                beam_width=args.beam_width,
                debias_iterations=4,
                physical_phase_projection=True,
            )
            recoveries.append(("full-linear", full_data, time.perf_counter() - started))

            started = time.perf_counter()
            pie = fienup_error_reduction(
                noisy,
                probe,
                scans,
                args.atoms,
                candidates,
                iterations=args.pie_iterations,
                restarts=args.pie_restarts,
                relaxation=0.8,
                seed=args.seed + 100_000 + 1000 * trial,
            )
            recoveries.append(("PIE", pie, time.perf_counter() - started))

            started = time.perf_counter()
            wf = hard_thresholded_wirtinger_flow(
                noisy,
                probe,
                scans,
                args.atoms,
                candidates,
                iterations=args.wf_iterations,
                restarts=args.wf_restarts,
                learning_rate=args.wf_learning_rate,
                hard_threshold_after=args.wf_threshold_after,
                seed=args.seed + 200_000 + 1000 * trial,
            )
            recoveries.append(("HT-WF", wf, time.perf_counter() - started))

            for method, recovery, runtime in recoveries:
                metrics = support_metrics(
                    truth, np.asarray(recovery.perturbation), args.atoms
                )
                rows.append(
                    {
                        "method": method,
                        "n_atoms": args.atoms,
                        "snr_db": "inf" if snr_db is None else snr_db,
                        "trial": trial,
                        "positions": ";".join(str(value) for value in positions),
                        **metrics,
                        "runtime_seconds": runtime,
                    }
                )
            rates = {
                method: int(rows[-4 + index]["support_exact"])
                for index, method in enumerate(("proposed", "full-linear", "PIE", "HT-WF"))
            }
            print(
                json.dumps(
                    {
                        "snr_db": "inf" if snr_db is None else snr_db,
                        "trial": trial,
                        "support_exact": rates,
                    }
                ),
                flush=True,
            )

    summaries: list[dict[str, object]] = []
    for method in ("proposed", "full-linear", "PIE", "HT-WF"):
        for snr_db in snr_values:
            label = "inf" if snr_db is None else str(float(snr_db))
            group = [
                row
                for row in rows
                if row["method"] == method and str(row["snr_db"]) == label
            ]
            summaries.append(
                {
                    "method": method,
                    "n_atoms": args.atoms,
                    "snr_db": label,
                    "trials": len(group),
                    "support_success_rate": float(
                        np.mean([float(row["support_exact"]) for row in group])
                    ),
                    "support_f1_mean": float(
                        np.mean([float(row["support_f1"]) for row in group])
                    ),
                    "nrmse_median": float(
                        np.median([float(row["nrmse"]) for row in group])
                    ),
                    "runtime_median_seconds": float(
                        np.median([float(row["runtime_seconds"]) for row in group])
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "model": {
            "n": n,
            "scans": scans,
            "lag_count": lag_count,
            "location_bound": location_bound,
            "gamma": gamma,
        },
        "summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("EXPERIMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
