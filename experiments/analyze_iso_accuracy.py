"""Combine raw iso-accuracy sweeps and compute tolerance frontiers."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
JULIA_INPUT = Path(
    os.environ.get(
        "ISO_JULIA_OUTPUT",
        FIGURES / "iso_accuracy_shared_krylov_raw.csv",
    )
)
MATFREE_INPUT = Path(
    os.environ.get(
        "ISO_MATFREE_OUTPUT",
        FIGURES / "iso_accuracy_matfree_raw.csv",
    )
)
OPTIMIZED_INPUT = Path(
    os.environ.get(
        "OPT_ISO_OUTPUT",
        FIGURES / "iso_accuracy_shared_krylov_optimized_raw.csv",
    )
)
COMBINED_OUTPUT = FIGURES / "iso_accuracy_combined_raw.csv"
FRONTIER_OUTPUT = FIGURES / "iso_accuracy_frontier.csv"
PLOT_OUTPUT = FIGURES / "iso_accuracy_parameter_frontier.png"
MAX_VJP_PLOT_OUTPUT = FIGURES / "iso_accuracy_max_vjp_frontier.png"

TOLERANCES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10)
METRICS = {
    "parameter": "parameter_vjp_relerr",
    "max_vjp": "max_vjp_relerr",
    "overall": "overall_relerr",
}


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def family(row):
    method = row["method"]
    if method == "shared_krylov_quadrature":
        return "Ours GL"
    if method == "projected_core":
        return "Ours projected core"
    if method == "shared_krylov_quadrature_optimized":
        return "Ours optimized GL"
    return "Matfree"


def configuration(row):
    pieces = [f"m={row['requested_m']}"]
    if row.get("n_quad") and math.isfinite(as_float(row, "n_quad")):
        pieces.append(f"nq={row['n_quad']}")
    if row.get("core_rtol") and math.isfinite(as_float(row, "core_rtol")):
        pieces.append(f"core_rtol={row['core_rtol']}")
    if row.get("core_rank") and math.isfinite(as_float(row, "core_rank")):
        pieces.append(f"rank={row['core_rank']}")
    return ",".join(pieces)


def pareto_points(rows, metric_key):
    finite = [
        row
        for row in rows
        if math.isfinite(as_float(row, metric_key))
        and math.isfinite(as_float(row, "loss_and_full_vjp_ms"))
        and as_float(row, metric_key) > 0
        and as_float(row, "loss_and_full_vjp_ms") > 0
    ]
    finite.sort(key=lambda row: as_float(row, metric_key))
    frontier = []
    best_runtime = math.inf
    for row in finite:
        runtime = as_float(row, "loss_and_full_vjp_ms")
        if runtime < best_runtime:
            frontier.append(row)
            best_runtime = runtime
    return frontier


def write_rows(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(rows, metric_key, x_label, output_path):
    colors = {
        "Ours GL": "#0072B2",
        "Ours optimized GL": "#56B4E9",
        "Ours projected core": "#009E73",
        "Matfree": "#D55E00",
    }
    markers = {
        "Ours GL": "o",
        "Ours optimized GL": "P",
        "Ours projected core": "s",
        "Matfree": "^",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, dt in zip(axes, (0.3, 1.0)):
        dt_rows = [row for row in rows if as_float(row, "dt") == dt]
        for method_family in colors:
            method_rows = [
                row for row in dt_rows if family(row) == method_family
            ]
            frontier = pareto_points(method_rows, metric_key)
            axis.plot(
                [as_float(row, metric_key) for row in frontier],
                [as_float(row, "loss_and_full_vjp_ms") for row in frontier],
                marker=markers[method_family],
                color=colors[method_family],
                linewidth=1.8,
                markersize=4.5,
                label=method_family,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_title(f"dt = {dt}")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("full loss + parameter/state VJP (ms)")
    axes[1].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    rows = read_rows(JULIA_INPUT) + read_rows(MATFREE_INPUT)
    if OPTIMIZED_INPUT.exists():
        rows += read_rows(OPTIMIZED_INPUT)
    write_rows(COMBINED_OUTPUT, rows)

    frontier_rows = []
    for dt in sorted({as_float(row, "dt") for row in rows}):
        dt_rows = [row for row in rows if as_float(row, "dt") == dt]
        for metric_name, metric_key in METRICS.items():
            for tolerance in TOLERANCES:
                for method_family in (
                    "Ours GL",
                    "Ours optimized GL",
                    "Ours projected core",
                    "Matfree",
                ):
                    candidates = [
                        row
                        for row in dt_rows
                        if family(row) == method_family
                        and as_float(row, metric_key) <= tolerance
                    ]
                    if not candidates:
                        continue
                    best = min(
                        candidates,
                        key=lambda row: as_float(
                            row, "loss_and_full_vjp_ms"
                        ),
                    )
                    frontier_rows.append(
                        {
                            "dt": dt,
                            "metric": metric_name,
                            "tolerance": tolerance,
                            "method": method_family,
                            "achieved_error": as_float(best, metric_key),
                            "loss_and_full_vjp_ms": as_float(
                                best, "loss_and_full_vjp_ms"
                            ),
                            "requested_m": best["requested_m"],
                            "n_quad": best.get("n_quad", ""),
                            "core_rtol": best.get("core_rtol", ""),
                            "core_rank": best.get("core_rank", ""),
                            "configuration": configuration(best),
                        }
                    )
    write_rows(FRONTIER_OUTPUT, frontier_rows)

    plot_frontier(
        rows,
        "parameter_vjp_relerr",
        "parameter-VJP relative error",
        PLOT_OUTPUT,
    )
    plot_frontier(
        rows,
        "max_vjp_relerr",
        "max(parameter-VJP, state-VJP) relative error",
        MAX_VJP_PLOT_OUTPUT,
    )

    print(f"Saved {COMBINED_OUTPUT}")
    print(f"Saved {FRONTIER_OUTPUT}")
    print(f"Saved {PLOT_OUTPUT}")
    print(f"Saved {MAX_VJP_PLOT_OUTPUT}")
    for row in frontier_rows:
        if (
            row["metric"] in ("parameter", "max_vjp")
            and row["tolerance"] in (1e-4, 1e-6, 1e-8)
        ):
            print(
                f"{row['metric']:<9s} dt={row['dt']:.1f} "
                f"tol={row['tolerance']:.0e} "
                f"{row['method']:<20s} "
                f"error={row['achieved_error']:.2e} "
                f"time={row['loss_and_full_vjp_ms']:.3f} ms "
                f"{row['configuration']}"
            )


if __name__ == "__main__":
    main()
