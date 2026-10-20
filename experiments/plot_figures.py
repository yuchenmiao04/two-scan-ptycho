"""Recreate the manuscript figures from authenticated per-trial measurements.

No solver is run and no recorded result is overwritten. Wrong-support outputs
remain in the error distributions; noiseless controls are displayed separately.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.verify_results import verify_files, verify_paper

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("joint", "full_linear", "pie", "htwf", "htaf")
SNRS = (0, 10, 20, 30, 40, 60, 80, "inf")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        values = tuple(row[key] for key in keys)
        groups[values].append(row)
    output = []
    for values, part in groups.items():
        condition = {}
        for key, value in zip(keys, values):
            try:
                numeric = float(value)
                condition[key] = int(numeric) if math.isfinite(numeric) and numeric.is_integer() else numeric
            except (ValueError, TypeError):
                condition[key] = value
            if key == "snr" and value == "inf":
                condition[key] = "inf"
        count = len(part)
        successes = sum(int(row["support_exact"]) for row in part)
        rate = successes / count
        z = 1.959963984540054
        denominator = 1 + z*z/count
        center = (rate+z*z/(2*count))/denominator
        half = z*math.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/denominator
        error_field = "defect_nrmse" if "defect_nrmse" in part[0] else "perturbation_nrmse"
        errors = np.asarray([float(row[error_field]) for row in part])
        require(np.all(np.isfinite(errors)), "A plotted error distribution contains unavailable observations")
        output.append({
            "condition": condition, "trials": count,
            "physical_exact_support": {"rate": rate, "successes": successes,
                "wilson95": [max(0.,center-half),min(1.,center+half)]},
            "metrics": {"defect_nrmse": {"median": float(np.median(errors)),
                "q25": float(np.quantile(errors,.25)), "q75": float(np.quantile(errors,.75))}},
        })
    return output


def load_evidence(root):
    data = verify_files(root)
    verify_paper(data)
    return {
        "gaussian": aggregate(data["gaussian"], ("sparsity","snr","method")),
        "poisson": aggregate(data["poisson"], ("sparsity","photons","method")),
        "mixed": aggregate(data["mixed"], ("sparsity","photons","method")),
        "cluster": aggregate(data["separation"], ("sparsity","snr","value","method")),
        "case2d": aggregate(data["case2d"], ("total_photons_two_frames","method")),
    }


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
        "axes.labelpad": 1.8, "axes.linewidth": .5, "grid.linewidth": .35,
        "grid.color": "#DFDFDF", "lines.linewidth": .85, "legend.handlelength": 1.9,
        "legend.columnspacing": .9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": .5, "ytick.major.width": .5, "xtick.major.size": 2.4,
        "ytick.major.size": 2.4, "xtick.minor.width": .4, "ytick.minor.width": .4,
        "xtick.minor.size": 1.4, "ytick.minor.size": 1.4, "xtick.major.pad": 2.1,
        "ytick.major.pad": 2.1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "two-scan-ptycho",
    })

STYLES = {
    "joint": ("#171717", "o", "-", "Joint LS"),
    "full_linear": ("#006EB8", "s", "--", "Full-linear"),
    "pie": ("#CB2929", "^", "-.", "Sparse PIE"),
    "htwf": ("#267446", "v", ":", "HT-WF"),
    "htaf": ("#8056A3", "D", (0, (4, 1, 1, 1)), "HT-AF"),
}
DRAW_ORDER = ("pie", "htwf", "htaf", "joint", "full_linear")
FIGSIZE = (3.5, 2.63)
PANEL_INCHES = (1.2425, .803)



def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DFDFDF", linewidth=.35)
    ax.tick_params(top=False, right=False)


def grid_figure():
    fig = plt.figure(figsize=FIGSIZE)
    # Keep the original panel dimensions; save height in the legend strip.
    # The upper row moves up .04 inch to keep photon labels clear of row two.
    positions = [(left, bottom / FIGSIZE[1], .355, PANEL_INCHES[1] / FIGSIZE[1])
                 for bottom in (1.51125, .28875) for left in (.115, .615)]
    axes = [fig.add_axes(p) for p in positions]
    for ax in axes:
        axes_style(ax)
    handles = [Line2D([], [], color=STYLES[m][0], marker=STYLES[m][1],
                      linestyle=STYLES[m][2], markerfacecolor="white", markersize=3.2,
                      markeredgewidth=.7, linewidth=.85, label=STYLES[m][3]) for m in METHODS]
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(.5, .995), ncol=5, frameon=False, borderaxespad=0,
               fontsize=7, handlelength=1.6, handletextpad=.35, columnspacing=.65)
    return fig, axes


def heading(ax, text):
    ax.set_title(text, loc="left", fontsize=8, pad=1.75)


def support(ax, x, groups, method, *, isolated_last=False, custom_style=None):
    color, marker, line, _ = custom_style or STYLES[method]
    y = np.asarray([g["physical_exact_support"]["rate"] * 100 for g in groups])
    ci = np.asarray([g["physical_exact_support"]["wilson95"] for g in groups]) * 100
    ax.vlines(x, ci[:, 0], ci[:, 1], colors=color, alpha=.32, linewidth=.45, zorder=2)
    split = len(x) - 1 if isolated_last else len(x)
    options = dict(color=color, marker=marker, linestyle=line, markerfacecolor="white",
                   markeredgewidth=.7, markersize=3.7 if method == "joint" else 2.9,
                   linewidth=.85, zorder=4 if method in ("joint", "full_linear") else 3)
    ax.plot(x[:split], y[:split], **options)
    if isolated_last:
        ax.plot(x[-1:], y[-1:], **dict(options, linestyle="none"))
    ax.set(ylim=(-4, 105), yticks=(0, 50, 100))


def snr_axis(ax):
    ax.axvspan(90, 106, color="#F5F5F5", zorder=0)
    ax.axvline(90, color="#B1B1B1", linestyle=(0, (2, 2)), linewidth=.5, zorder=1)
    ax.set(xlim=(-4, 106), xticks=(0, 20, 40, 60, 80, 100),
           xticklabels=("0", "20", "40", "60", "80", r"$\infty$"))


def benchmarks(evidence):
    groups = evidence["gaussian"]
    lookup = {(g["condition"]["sparsity"], g["condition"]["snr"], g["condition"]["method"]): g
              for g in groups}
    fig, axes = grid_figure()
    x = np.asarray((*SNRS[:-1], 100), float)
    for ax, n, tag in zip((axes[1], axes[2], axes[3]), (2, 1, 3), ("b", "c", "d")):
        for method in DRAW_ORDER:
            support(ax, x, [lookup[n, s, method] for s in SNRS], method, isolated_last=True)
        snr_axis(ax)
        heading(ax, fr"({tag}) $N={n}$")
    axes[2].set_ylabel("Exact support (%)")
    fig.text(.55, .055 / FIGSIZE[1], "Target defect-intensity SNR (dB)", ha="center", fontsize=7.5)

    # The error panel intentionally shows finite SNR only. No broken scale,
    # clipped quartiles, or artificial floor hides its noise-regime comparison.
    error_ax = axes[0]
    error_ax.set(yscale="log", ylim=(2e-5, 60), yticks=(1e-4, 1e-2, 1),
                 xlim=(-3, 83), xticks=(0, 20, 40, 60, 80))
    error_ax.yaxis.set_minor_locator(NullLocator())
    for method in DRAW_ORDER:
        ds = [lookup[2, s, method]["metrics"]["defect_nrmse"] for s in SNRS[:-1]]
        require(all(2e-5 < v < 60 for d in ds for v in (d["q25"], d["median"], d["q75"])),
                "NRMSE display limits would hide a statistic")
        color, marker, line, _ = STYLES[method]
        y, lo, hi = ([d[k] for d in ds] for k in ("median", "q25", "q75"))
        error_ax.fill_between(x[:-1], lo, hi, color=color, alpha=.075, linewidth=0)
        error_ax.plot(x[:-1], y, color=color, marker=marker, linestyle=line,
                      markerfacecolor="white", markersize=3.7 if method == "joint" else 2.9,
                      markeredgewidth=.7, linewidth=.85, zorder=4)
    error_ax.set_ylabel("Defect NRMSE")
    heading(error_ax, r"(a) $N=2$: median NRMSE")
    caption = ("Gaussian benchmark: N=1,2,3, seven finite target defect-intensity SNRs "
               "(0,10,20,30,40,60,80 dB) and one noiseless control, five methods, 200 paired "
               "trials per condition (24000 method-level records). (b-d) Exact-support rates "
               "and pointwise 95% Wilson intervals. (a) N=2, seven finite SNRs only: defect "
               "NRMSE medians and IQR, all 200 observations including incorrect-support "
               "outputs; 7000 method-level "
               "records. The noiseless control occupies a separated final column and is not "
               "connected to finite SNR in (b-d); (a) has an unbroken logarithmic error "
               "axis and no noiseless point. All displayed quartiles are within the axis "
               "limits. All NRMSE observations are available; no "
               "exceptions or zero imputation. Methods use the same frames within each trial.")
    return fig, caption, groups


def design_figure(evidence):
    fig, axes = grid_figure()
    displayed = []
    for ax, suite, tag, title in zip(axes[:2], ("poisson", "mixed"), ("a", "b"),
                                     ("Poisson", "Poisson + read noise")):
        groups = evidence[suite]
        displayed.extend(groups)
        for method in DRAW_ORDER:
            part = sorted((g for g in groups if g["condition"]["method"] == method),
                          key=lambda g: g["condition"]["photons"])
            support(ax, np.log10([g["condition"]["photons"] for g in part]), part, method)
        ax.set(xlim=(4.8, 11.2), xticks=(5, 7, 9, 11))
        ax.set_xlabel(r"$\log_{10}$ total photons")
        heading(ax, f"({tag}) {title}")
    cluster = evidence["cluster"]
    displayed.extend(cluster)
    ax = axes[3]
    for method in DRAW_ORDER:
        part = sorted((g for g in cluster if g["condition"]["method"] == method),
                      key=lambda g: g["condition"]["value"])
        support(ax, [g["condition"]["value"] for g in part], part, method)
    ax.axvspan(1.7, 4, color="#F0F0F0", zorder=0)
    ax.axvline(4, color="#A0A0A0", linewidth=.5, linestyle=(0, (2, 2)), zorder=1)
    ax.text(2.9, 7, r"$d\leq 2N$", ha="center", va="bottom", fontsize=6.2, color="#555555")
    ax.set(xlim=(1.7, 8.3), xticks=(2, 4, 5, 6, 8), xlabel="Separation (pixels)")
    heading(ax, r"(d) Separation, $N=2$")
    ax = axes[2]
    groups = evidence["case2d"]
    displayed.extend(groups)
    handles = []
    for method, item in (("calibrated", ("#006EB8", "o", "-", "Calibrated")),
                         ("gamma_plus_2pct", ("#CB2929", "s", "--", r"$+2\%\;\gamma$"))):
        part = sorted((g for g in groups if g["condition"]["method"] == method),
                      key=lambda g: g["condition"]["total_photons_two_frames"])
        support(ax, np.log10([g["condition"]["total_photons_two_frames"] for g in part]),
                part, method, custom_style=item)
        handles.append(Line2D([], [], color=item[0], marker=item[1], linestyle=item[2],
                              markerfacecolor="white", markersize=3, linewidth=.85, label=item[3]))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7,
              borderaxespad=.15, handlelength=1.4, labelspacing=.15)
    ax.set(xlim=(7.65, 10.45), xticks=(8, 9, 10), xlabel=r"$\log_{10}$ total photons")
    heading(ax, r"(c) 2D case, $N=3$")
    fig.text(.015, 1.38 / FIGSIZE[1], "Exact support (%)", rotation=90, va="center", ha="left", fontsize=7.5)
    caption = ("All support panels show pointwise 95% Wilson intervals. (a,b) N=2 Poisson "
               "and Poisson plus two-electron rms read noise per pixel: all seven total "
               "two-frame photon budgets 1e5..1e11, five methods, 200 paired trials per "
               "budget within each noise family (7000 records each). Poisson and mixed "
               "do not reuse identical shot-noise draws; their difference is not a paired "
               "read-noise-only effect. (d) Full fixed-separation sweep at 60 dB, N=2: "
               "separations 2,4,5,6,8, five methods, 100 trials each (2500 displayed records "
               "from the complete 26500-row mismatch suite). Shading marks separation "
               "at most 2N=4, outside the sufficient strict-separation condition. "
               "(c) Fixed 2D synthetic common-signature, nonnegative-strength case with "
               "known phase and exact blank calibration: six budgets 6e7..2e10, 200 noise "
               "realizations each, calibrated and +2% assumed-gamma reconstructions "
               "(2400 records), not general complex 2D recovery. Photon axes display "
               "log10 of total expected photons over the same two frames; the small "
               "legend in (c) is separate from the five-method benchmark legend.")
    return fig, caption, displayed



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT/"results/reproduced/figures")
    parser.add_argument("--formats", nargs="+", choices=("png","svg","pdf"), default=("png","svg"))
    parser.add_argument("--replace", action="store_true", help="Replace this command's named figure outputs")
    args = parser.parse_args()
    evidence = load_evidence(args.root.resolve())
    output = args.output.resolve()
    records = (args.root.resolve()/"results").resolve()
    require(output != records and (records not in output.parents or records/"reproduced" in output.parents),
            "Figure output must not overwrite recorded results")
    names = ("gaussian_comparison", "acquisition_tests")
    for name in names:
        for suffix in (*args.formats, "json"):
            target = output/f"{name}.{suffix}"
            require(args.replace or not target.exists(), f"Output exists: {target.name}; use --replace")
    global plt, Line2D, NullLocator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    style()
    output.mkdir(parents=True,exist_ok=True)
    for name, make in zip(names,(benchmarks,design_figure)):
        fig,caption,groups = make(evidence)
        for fmt in args.formats:
            metadata = {"Date": None, "Creator": "two-scan-ptycho"} if fmt=="svg" else (
                {"CreationDate": None, "ModDate": None, "Creator": "two-scan-ptycho"} if fmt=="pdf" else
                {"Software": "two-scan-ptycho"})
            fig.savefig(output/f"{name}.{fmt}",dpi=400,bbox_inches=None,metadata=metadata)
            if fmt == "svg":
                svg = output/f"{name}.{fmt}"
                svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
        plt.close(fig)
        (output/f"{name}.json").write_text(json.dumps({"caption":caption,
            "figure_inches":list(FIGSIZE),"conditions":groups},indent=2,allow_nan=False)+"\n")
        print(f"Wrote {name}")


if __name__ == "__main__":
    main()
