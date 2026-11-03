"""Combine portable benchmark runs and extract matched-accuracy frontiers."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "benchmark_results" / "portable_iso"
TOLERANCES = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-8)


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def method_label(row):
    method = row["method"]
    if method == "Matfree_finite_Arnoldi_VJP":
        return "Matfree"
    if method == "Structured_semigroup_core":
        return "Ours core"
    if method == "Structured_semigroup_quadrature":
        return "Ours quadrature"
    if method == "Julia_optimized_semigroup_quadrature":
        return "Ours Julia"
    return "Dense AD"


def configuration(row):
    pieces = []
    if row.get("m"):
        pieces.append(f"m={row['m']}")
    if row.get("core_rank"):
        pieces.append(f"rank={row['core_rank']}")
    if row.get("n_quad"):
        pieces.append(f"nq={row['n_quad']}")
    return ",".join(pieces)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--directions", type=int)
    arguments = parser.parse_args()

    inputs = arguments.inputs or sorted(
        arguments.output_dir.glob("portable_iso_*.csv")
    )
    inputs = [
        path
        for path in inputs
        if path.name not in ("portable_iso_combined.csv", "portable_iso_frontier.csv")
    ]
    if not inputs:
        raise SystemExit("no portable benchmark CSVs found")
    rows = read_rows(inputs)
    combined_path = arguments.output_dir / "portable_iso_combined.csv"
    write_rows(combined_path, rows)

    frontier_rows = []
    group_keys = sorted(
        {
            (
                row["run_label"],
                row["backend"],
                row["device_kind"],
                int(float(row["n"])),
                int(float(row["M"])),
                method_label(row),
            )
            for row in rows
        }
    )
    for run_label, backend, device_kind, n, directions, method in group_keys:
        group = [
            row
            for row in rows
            if row["run_label"] == run_label
            and row["backend"] == backend
            and row["device_kind"] == device_kind
            and int(float(row["n"])) == n
            and int(float(row["M"])) == directions
            and method_label(row) == method
        ]
        for tolerance in TOLERANCES:
            candidates = [
                row
                for row in group
                if as_float(row, "max_vjp_relerr") <= tolerance
                and math.isfinite(as_float(row, "runtime_median_ms"))
            ]
            if not candidates:
                continue
            best = min(
                candidates, key=lambda row: as_float(row, "runtime_median_ms")
            )
            frontier_rows.append(
                {
                    "run_label": run_label,
                    "backend": backend,
                    "device_kind": device_kind,
                    "n": n,
                    "M": directions,
                    "method": method,
                    "tolerance": tolerance,
                    "achieved_error": as_float(best, "max_vjp_relerr"),
                    "runtime_median_ms": as_float(
                        best, "runtime_median_ms"
                    ),
                    "runtime_q25_ms": as_float(best, "runtime_q25_ms"),
                    "runtime_q75_ms": as_float(best, "runtime_q75_ms"),
                    "configuration": configuration(best),
                }
            )
    frontier_path = arguments.output_dir / "portable_iso_frontier.csv"
    write_rows(frontier_path, frontier_rows)

    available_directions = sorted(
        {int(float(row["M"])) for row in rows}
    )
    selected_directions = (
        arguments.directions
        if arguments.directions is not None
        else available_directions[-1]
    )
    colors = {
        "Matfree": "#D55E00",
        "Ours core": "#009E73",
        "Ours quadrature": "#0072B2",
        "Ours Julia": "#56B4E9",
        "Dense AD": "#6A3D9A",
    }
    markers = {
        "Matfree": "^",
        "Ours core": "s",
        "Ours quadrature": "o",
        "Ours Julia": "P",
        "Dense AD": "D",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    for axis, tolerance in zip(axes, (1.0e-4, 1.0e-6)):
        selected = [
            row
            for row in frontier_rows
            if row["M"] == selected_directions
            and row["tolerance"] == tolerance
        ]
        series_keys = sorted(
            {
                (
                    row["run_label"],
                    row["backend"],
                    row["device_kind"],
                    row["method"],
                )
                for row in selected
            }
        )
        for run_label, backend, device_kind, method in series_keys:
            series = sorted(
                (
                    row
                    for row in selected
                    if row["run_label"] == run_label
                    and row["backend"] == backend
                    and row["device_kind"] == device_kind
                    and row["method"] == method
                ),
                key=lambda row: row["n"],
            )
            axis.plot(
                [row["n"] for row in series],
                [row["runtime_median_ms"] for row in series],
                marker=markers[method],
                color=colors[method],
                linewidth=1.7,
                markersize=5,
                label=f"{method}, {run_label}",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("state dimension n")
        axis.set_title(f"max VJP error <= {tolerance:.0e}")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("loss + parameter/state VJP (ms)")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, frameon=False, fontsize=8)
    figure.suptitle(f"Iso-accuracy scaling, M = {selected_directions}")
    figure.tight_layout()
    plot_path = arguments.output_dir / "portable_iso_scaling.png"
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"Read {len(inputs)} run files and {len(rows)} rows")
    print(f"Saved {combined_path}")
    print(f"Saved {frontier_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
