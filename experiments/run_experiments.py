"""Paired, resumable experiments with no ground-truth inputs to solvers.

Run tuning first, then pass its frozen.json to the formal suites. Seeds for
tuning and evaluation are disjoint. One checkpoint contains all methods on the
same object and noisy frames, so a failed method is not replaced by a new trial.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from twoscan_ptycho.core import (
    affine_object, centered_grid, diffraction_intensity, gaussian_probe,
    position_to_index, sample_separated_positions,
    sparse_perturbation,
)
from twoscan_ptycho.reconstruction import build_joint_dictionary, GlobalSupportSolver, extract_raw_data

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("joint", "full_linear", "pie", "htwf", "htaf")
SEEDS = {"tuning": 19692001, "formal": 20260920, "timing": 19371011}


def stable_seed(*items):
    raw = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist()), encoding="utf-8")
    temp.replace(path)


def source_hashes():
    paths = [Path(__file__), *sorted((ROOT / "src/twoscan_ptycho").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_frozen_sources(frozen):
    """Reject missing, added or changed solver files after settings are frozen."""
    recorded = frozen.get("distribution_source_hashes")
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError("Frozen settings require nonempty distribution_source_hashes")
    expected = {path: digest for path, digest in recorded.items() if path.startswith("src/")}
    current = {path: digest for path, digest in source_hashes().items() if path.startswith("src/")}
    if expected.keys() != current.keys():
        raise RuntimeError("Frozen solver source set differs; retune before evaluation")
    for path, digest in current.items():
        if expected[path] != digest:
            raise RuntimeError(f"Solver changed after calibration: {path}. Retune before evaluation.")


def hardware():
    info = {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
            "cpu_count": os.cpu_count(), "blas_threads": 1}
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.physicalcpu", "hw.memsize"):
            info[key] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    return info


def configuration(sparsity):
    bound = max(10, math.ceil((sparsity - 1) * (2 * sparsity + 2) / 2) + 5)
    return bound, 5.0 / bound**2


@lru_cache(maxsize=80)
def joint_solver(n, gamma, scans, candidates, upper, separation, margin):
    model = build_joint_dictionary(n, gamma, scans, candidates, upper,
                                   minimum_separation=separation, enforce_margin=margin)
    return GlobalSupportSolver(model, upper)


def make_instance(job):
    sparsity = job.get("sparsity", 2)
    bound, gamma = configuration(sparsity)
    n, scans = 257, (0, 1)
    stress, value = job.get("stress", "none"), job.get("value", 0)
    if stress not in ("none", "cluster", "background"):
        raise ValueError(f"Unsupported experimental condition: {stress}")
    upper = sparsity
    spacing = 2 * sparsity
    candidates = tuple(range(-bound, bound + 1))
    rng = np.random.default_rng(stable_seed(job["namespace"], "object", sparsity, job["trial"], stress))
    positions = sample_separated_positions(n, sparsity, spacing, scans, 2*sparsity,
                                           rng, location_bound=min(bound, n//2-2*sparsity-1))
    if stress == "cluster":
        distance = int(value)
        positions = (-distance // 2, -distance // 2 + distance)
        spacing = 0   # Allow the true clustered support; clean-lag identity may fail.
    amplitudes = rng.uniform(.5, 1.5, sparsity) * np.exp(1j*rng.uniform(-np.pi, np.pi, sparsity))
    probe = gaussian_probe(n, gamma)
    z = affine_object(n, positions, amplitudes)
    if stress == "background":
        q = centered_grid(n)
        z += float(value) * (np.cos(2*np.pi*q/n) + .5j*np.sin(4*np.pi*q/n))
    physical_positions = np.asarray(positions, dtype=float)
    clean = np.asarray([diffraction_intensity(z, probe, r) for r in scans])
    true_blank = np.broadcast_to(np.abs(np.fft.fft(probe))**2, clean.shape)
    defect_intensity = clean-true_blank
    noise_rng = np.random.default_rng(stable_seed(job["namespace"], "noise", sparsity, job["trial"], stress))
    sigma = 0.0
    if job["suite"] in ("poisson", "mixed"):
        total_photons = float(job["photons"])
        frames = []
        for frame in clean:
            photons = total_photons/len(scans)
            scale = frame.sum()/photons
            counts = noise_rng.poisson(frame/scale).astype(float)
            if job["suite"] == "mixed":
                counts += noise_rng.normal(0, 2.0, n)  # Two-electron rms read noise per pixel.
            frames.append(counts*scale)
        measured = np.asarray(frames)
    else:
        snr = job.get("snr", 60)
        if snr != "inf":
            sigma = np.linalg.norm(defect_intensity)/np.sqrt(clean.size)*10**(-float(snr)/20)
        measured = clean + sigma*noise_rng.standard_normal(clean.shape)
    truth = sparse_perturbation(n, positions, amplitudes)
    error_norm = float(np.linalg.norm(measured-clean))
    effective_snr = 20*np.log10(np.linalg.norm(defect_intensity)/error_norm) if error_norm else None
    total_snr = 20*np.log10(np.linalg.norm(clean)/error_norm) if error_norm else None
    return dict(n=n, scans=scans, candidates=candidates, upper=upper, spacing=spacing,
                gamma=gamma, probe=probe, positions=positions,
                physical_positions=physical_positions, amplitudes=amplitudes, truth=truth,
                clean=clean, measured=measured, sigma=sigma, effective_snr=effective_snr,
                total_snr=total_snr, margin=True)


def calculate_metrics(instance, estimate, support):
    truth = instance["truth"]
    support = tuple(int(p) for p in support)
    exact = set(support)==set(instance["positions"])
    nrmse = float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))
    loc, amp, phase = None, None, None
    if len(support)==len(instance["positions"]):
        physical = instance["physical_positions"]
        ordering = min(itertools.permutations(support), key=lambda s: np.sum((np.asarray(s)-physical)**2))
        loc = float(np.sqrt(np.mean((np.asarray(ordering)-physical)**2)))
        recovered = estimate[position_to_index(np.asarray(ordering), instance["n"])]
        amp = float(np.linalg.norm(np.abs(recovered)-np.abs(instance["amplitudes"]))/np.linalg.norm(instance["amplitudes"]))
        phase = float(np.sqrt(np.mean(np.angle(recovered*np.conj(instance["amplitudes"]))**2)))
    predicted = np.asarray([diffraction_intensity(1+estimate, instance["probe"], r) for r in instance["scans"]])
    data_residual = float(np.linalg.norm(predicted-instance["measured"])/max(np.linalg.norm(instance["measured"]), 1e-300))
    return {"support_exact": int(exact),
            "grid_support_exact":int(exact),
            "support_metric":"exact_integer_support",
            "defect_nrmse": nrmse,
            "grid_proxy_nrmse": nrmse, "localization_rmse": loc, "amplitude_relative_error": amp,
            "phase_rmse_rad": phase, "relative_intensity_residual": data_residual,
            "estimated_support": ";".join(map(str,support)), "estimated_sparsity": len(support)}


def recover(method, instance, seed, config):
    if method not in METHODS:
        raise ValueError(f"Unknown reconstruction method: {method}")
    if method == "joint":
        solver = joint_solver(instance["n"], instance["gamma"], instance["scans"], instance["candidates"],
                              instance["upper"], instance["spacing"], instance["margin"])
        extraction_start=time.perf_counter()
        b = extract_raw_data(instance["measured"], solver.dictionary)
        extraction_seconds=time.perf_counter()-extraction_start
        # A 99% Laurent-Massart Gaussian norm bound in the raw-lag coordinates.
        # Sigma is a specified acquisition parameter, not estimated from truth.
        m = len(b)
        epsilon = instance["sigma"]/np.sqrt(2*instance["n"])*np.sqrt(m+2*np.sqrt(m*np.log(100))+2*np.log(100))
        result = solver.solve(b)
        return result.u, result.support, result.status, {"iterations": 1, "restarts": 1,
            "legal_support_count": result.candidate_count, "raw_lag_residual": result.residual,
            "setup_seconds": solver.precompute_seconds+solver.dictionary.build_seconds,
            "online_seconds": result.timings["total_online"]+extraction_seconds,
            "noise_bound": epsilon, "selection": "residual"}
    from twoscan_ptycho.baselines import BaselineConfig, recover_baseline
    answer = recover_baseline(method, instance["measured"], instance["probe"], instance["scans"],
                              instance["candidates"], instance["upper"], instance["spacing"],
                              seed=seed, config=BaselineConfig(**config))
    return answer.u, answer.support, answer.status, {"iterations":answer.iterations,
        "restarts":config.get("restarts",1), **answer.diagnostics,
        "online_seconds":answer.diagnostics["optimization_seconds"]}


def perform_job(job):
    instance = make_instance(job)
    records = []
    for method in job.get("methods", METHODS):
        config = job.get("baseline_configs",{}).get(method,{})
        start = time.perf_counter()
        row = {key:job[key] for key in ("id","suite","trial","sparsity","snr","photons","stress","value","config_id") if key in job}
        if job["suite"] == "mismatch":
            row["suite"] = "separation" if job["stress"] == "cluster" else "background"
        row.update(method=method, true_support=";".join(map(str,instance["positions"])),
                   effective_snr_db=instance["effective_snr"], total_snr_db=instance["total_snr"],
                   coefficient_min=float(np.min(np.abs(instance["amplitudes"]))),
                   n=instance["n"],gamma=instance["gamma"],candidate_sites=len(instance["candidates"]))
        try:
            estimate, support, status, diagnostics = recover(method,instance,stable_seed(job["namespace"],job["trial"],method),
                    config)
            if not np.all(np.isfinite(estimate)):
                raise FloatingPointError("Nonfinite estimate")
            row.update(calculate_metrics(instance,estimate,support))
            row.update(status=status, failed=0, iterations=int(diagnostics.get("iterations",0)),
                       restarts=int(diagnostics.get("restarts",1)), diagnostics=diagnostics,
                       setup_seconds=float(diagnostics.get("setup_seconds",0)),
                       online_seconds=float(diagnostics.get("online_seconds",0)))
            row["algorithm_total_seconds"]=row["setup_seconds"]+row["online_seconds"]
        except Exception as error:
            row.update(status=f"{type(error).__name__}: {error}",failed=1,support_exact=0,
                       defect_nrmse=None,grid_proxy_nrmse=None,localization_rmse=None,
                       amplitude_relative_error=None,phase_rmse_rad=None,relative_intensity_residual=None,
                       estimated_support="",estimated_sparsity=0,iterations=0,restarts=config.get("restarts",1))
        row["wall_seconds"] = time.perf_counter()-start
        records.append(row)
    return {"job":job,"rows":records}


def wilson(successes,total):
    z=1.959963984540054
    p=successes/total; denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0.,center-half),min(1.,center+half)]


def summarize(rows):
    keys=("suite","sparsity","snr","photons","stress","value","method","config_id")
    groups={}
    for row in rows:
        key=tuple(row.get(k) for k in keys)
        groups.setdefault(key,[]).append(row)
    result=[]
    for key,values in groups.items():
        total=len(values)
        support_values=[v["support_exact"] for v in values if v["support_exact"] is not None]
        successes=sum(support_values)
        item={k:v for k,v in zip(keys,key) if v is not None}
        item.update(trials=total,failures=sum(v["failed"] for v in values),
                    support_success_rate=successes/len(support_values) if support_values else None,
                    support_wilson95=wilson(successes,len(support_values)) if support_values else None,
                    support_applicable_trials=len(support_values))
        for field in ("defect_nrmse","localization_rmse","amplitude_relative_error","phase_rmse_rad",
                      "relative_intensity_residual","wall_seconds","setup_seconds","online_seconds",
                      "algorithm_total_seconds","iterations","effective_snr_db","total_snr_db"):
            array=np.array([v[field] for v in values if v.get(field) is not None],float)
            if len(array):
                item[field+"_mean"]=float(np.mean(array))
                item[field+"_median"]=float(np.median(array))
                item[field+"_q25_q75"]=[float(x) for x in np.quantile(array,[.25,.75])]
                item[field+"_q10_q90"]=[float(x) for x in np.quantile(array,[.1,.9])]
                item[field+"_available"]=len(array)
        result.append(item)
    return result


def execute(jobs,output,workers,resume=True,metadata=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for i,job in enumerate(jobs):
        job["id"]=f"{i:06d}"
    identity={"jobs":jobs,"source_hashes":source_hashes()}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous["run_hash"]!=digest:
            raise RuntimeError(f"Refusing to mix different jobs/code in {output}; choose a fresh output directory.")
        if not resume:
            raise RuntimeError("Existing run: use resume or a fresh output directory")
    else:
        dump_json(manifest_path,{"run_hash":digest,"created_utc":datetime.now(timezone.utc).isoformat(),
                    "hardware":hardware(),"workers":workers,"expected_jobs":len(jobs),
                    "expected_method_trials":sum(len(j.get("methods",METHODS)) for j in jobs),
                    "timing_is_isolated":workers==1 and all(j["suite"]=="timing" for j in jobs),
                    "source_hashes":identity["source_hashes"],"metadata":metadata or {},"jobs":jobs})
    checkpoints=output/"checkpoints"; checkpoints.mkdir(exist_ok=True)
    todo=[j for j in jobs if not (checkpoints/(j["id"]+".json")).exists()]
    print(f"{output.name}: {len(todo)}/{len(jobs)} jobs remaining; workers={workers}",flush=True)
    start=time.perf_counter()
    if workers==1:
        for i,job in enumerate(todo,1):
            dump_json(checkpoints/(job["id"]+".json"),perform_job(job))
            if i%20==0 or i==len(todo): print(f"{output.name}: {i}/{len(todo)}, {time.perf_counter()-start:.1f}s",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(perform_job,j):j for j in todo}
            for i,future in enumerate(as_completed(pending),1):
                job=pending[future]
                dump_json(checkpoints/(job["id"]+".json"),future.result())
                if i%100==0 or i==len(todo): print(f"{output.name}: {i}/{len(todo)}, {time.perf_counter()-start:.1f}s",flush=True)
    rows=[]
    for job in jobs:
        rows.extend(json.loads((checkpoints/(job["id"]+".json")).read_text())["rows"])
    fields=sorted(set().union(*(r.keys() for r in rows)))
    fields.remove("diagnostics") if "diagnostics" in fields else None
    with (output/"trials.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    summary={"completed_utc":datetime.now(timezone.utc).isoformat(),"run_hash":digest,
             "jobs_completed":len(jobs),"method_trials":len(rows),"failures":sum(r["failed"] for r in rows),
             "aggregate":summarize(rows)}
    dump_json(output/"summary.json",summary)
    return rows


def formal_jobs(suite,trials,configs):
    conditions=[]
    if suite=="gaussian":
        conditions=[dict(sparsity=n,snr=snr) for n in (1,2,3) for snr in (0,10,20,30,40,60,80,"inf")]
    elif suite in ("poisson","mixed"):
        conditions=[dict(sparsity=2,photons=p) for p in (1e5,1e6,1e7,1e8,1e9,1e10,1e11)]
    elif suite=="separation":
        conditions=[dict(sparsity=2,snr=60,stress="cluster",value=v) for v in (2,4,5,6,8)]
    elif suite=="background":
        conditions=[dict(sparsity=2,snr=60,stress="background",value=v) for v in (0,.001)]
    elif suite=="timing":
        conditions=[dict(sparsity=n,snr=snr) for n in (1,2,3) for snr in (20,60,"inf")]
    else:
        raise ValueError(suite)
    # Keep the acquisition namespace and internal suite identity for exact pairing.
    acquisition_suite = "mismatch" if suite in ("separation", "background") else suite
    return [dict(suite=acquisition_suite,namespace=SEEDS["timing" if suite=="timing" else "formal"],trial=t,
                 baseline_configs=configs,**c) for c in conditions for t in range(trials)]


def tune(output,workers,trials):
    grid={
       "pie":[dict(iterations=i,restarts=r,pie_relaxation=a,threshold_after=t) for i,r,a,t in
              ((200,4,.5,0),(600,8,.5,0),(1200,8,.5,0),(600,8,1.0,0),(600,8,.5,100),(1200,32,.5,200))],
       "htwf":[dict(iterations=i,restarts=r,learning_rate=a,threshold_after=t) for i,r,a,t in
               ((200,4,.5,0),(600,8,.5,0),(1200,8,.5,0),(600,8,1.0,0),(600,8,.5,100),(1200,32,.5,200))],
       "htaf":[dict(iterations=i,restarts=r,learning_rate=a,threshold_after=t) for i,r,a,t in
               ((200,4,.5,0),(600,8,.5,0),(1200,8,.5,0),(600,8,1.0,0),(600,8,.5,100),(1200,32,.5,200))],
       "full_linear":[dict(iterations=i,restarts=r) for i,r in ((4,1),(12,1),(12,4),(24,8))],
    }
    jobs=[dict(suite="baseline_tuning",namespace=SEEDS["tuning"],trial=t,sparsity=n,snr=snr,
               methods=(method,),baseline_configs={method:config},config_id=f"{method}_{index}")
          for method,choices in grid.items() for index,config in enumerate(choices)
          for n in (2,3) for snr in (20,60,"inf") for t in range(trials)]
    rows=execute(jobs,output,workers)
    candidates={}
    for method,choices in grid.items():
        scores=[]
        for index,config in enumerate(choices):
            group=[r for r in rows if r["config_id"]==f"{method}_{index}"]
            # Failures receive a fixed penalty, not deletion. Truth is used only
            # in disjoint validation, never inside a recovery method.
            score=float(np.mean([math.log1p(min(r["defect_nrmse"],100)) if not r["failed"] else math.log1p(100) for r in group]))
            scores.append({"config":config,"score_mean_log1p_nrmse":score,"runtime_median":float(np.median([r["wall_seconds"] for r in group]))})
        best=min(range(len(scores)),key=lambda i:(scores[i]["score_mean_log1p_nrmse"],scores[i]["runtime_median"]))
        candidates[method]={"selected":scores[best]["config"],"validation":scores}
    frozen={"created_utc":datetime.now(timezone.utc).isoformat(),"seed":SEEDS["tuning"],
            "selection_criterion":"Minimum mean log(1+min(defect NRMSE,100)); numerical failures assigned log(101).",
            "baseline_configs":{m:c["selected"] for m,c in candidates.items()},"validation":candidates,
            "distribution_source_hashes":json.loads((Path(output)/"manifest.json").read_text())["source_hashes"]}
    dump_json(Path(output)/"frozen.json",frozen)
    print(json.dumps(frozen["baseline_configs"],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite",choices=("tune","gaussian","poisson","mixed","separation","background","timing","smoke"))
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--trials",type=int)
    parser.add_argument("--frozen",type=Path,default=ROOT/"results/baseline_tuning/frozen.json")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.workers < 1 or (args.trials is not None and args.trials < 1):
        parser.error("workers and trials must be positive")
    output=args.output or ROOT/"results/reproduced"/("baseline_tuning" if args.suite=="tune" else args.suite)
    if args.suite=="tune":
        tune(output,args.workers,args.trials or 20); return
    if args.suite=="smoke":
        configs={m:{"iterations":40,"restarts":2} for m in ("pie","htwf","htaf")}
        configs["full_linear"]={"iterations":4,"restarts":1}
        jobs=[dict(suite="smoke",namespace=SEEDS["tuning"],trial=t,sparsity=n,snr="inf",baseline_configs=configs)
              for n in (1,2,3) for t in range(args.trials or 2)]
        execute(jobs,output,args.workers); return
    if not args.frozen.is_file():
        parser.error("Frozen settings not found; supply --frozen from independent baseline tuning")
    frozen=json.loads(args.frozen.read_text())
    validate_frozen_sources(frozen)
    defaults={"gaussian":200,"poisson":200,"mixed":200,"separation":100,"background":100,"timing":30}
    if args.suite=="timing" and args.workers!=1:
        parser.error("Timing must run alone with --workers 1")
    jobs=formal_jobs(args.suite,args.trials or defaults[args.suite],frozen["baseline_configs"])
    if args.suite=="timing":
        # Explicit excluded warm-up; the first retained trial still records
        # dictionary setup separately through diagnostics.
        warm=dict(jobs[0],trial=1000000,id="warmup")
        perform_job(warm)
    execute(jobs,output,args.workers,metadata={"frozen":frozen,"warmups":1 if args.suite=="timing" else 0})


if __name__=="__main__":
    main()
