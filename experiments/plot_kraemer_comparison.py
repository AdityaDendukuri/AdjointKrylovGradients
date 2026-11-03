"""Plot matched Matfree versus shared-Krylov accuracy/runtime results."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ferechet-matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "paper" / "figures" / "kraemer_shared_krylov.csv",
    ROOT / "paper" / "figures" / "kraemer_matfree.csv",
)
OUTPUT = ROOT / "paper" / "figures" / "fig_kraemer_accuracy_runtime.png"

BACKENDS = {
    "Shared_Krylov_Julia": ("Shared Krylov", "#2474B5"),
    "Compressed_Core_Julia": ("Compressed core", "#009E73"),
    "Matfree_JAX": ("Matfree", "#D55E00"),
}
PRECISIONS = {
    "float64": ("Float64", "-", "o"),
    "float32": ("Float32", "--", "s"),
}


def read_rows():
    rows = []
    for input_path in INPUTS:
        with input_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    for row in rows:
        for key in (
            "dt",
            "m",
            "parameter_vjp_relerr",
            "state_vjp_relerr",
            "loss_and_full_vjp_ms",
        ):
            row[key] = float(row[key])
    return rows


def main():
    rows = read_rows()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(10.5, 7.8), sharex=True, constrained_layout=True
    )

    metrics = (
        ("parameter_vjp_relerr", "Parameter-VJP relative error"),
        ("state_vjp_relerr", "State-VJP relative error"),
    )
    for row_index, dt in enumerate((0.3, 1.0)):
        for column_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for backend, (backend_label, color) in BACKENDS.items():
                for precision, (precision_label, linestyle, marker) in PRECISIONS.items():
                    selected = sorted(
                        (
                            row
                            for row in rows
                            if row["backend"] == backend
                            and row["precision"] == precision
                            and row["dt"] == dt
                        ),
                        key=lambda row: row["m"],
                    )
                    axis.plot(
                        [row["loss_and_full_vjp_ms"] for row in selected],
                        [row[metric] for row in selected],
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=5.5,
                        linewidth=1.8,
                        label=f"{backend_label}, {precision_label}",
                    )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25, linewidth=0.7)
            if row_index == 0:
                axis.set_title(ylabel.replace(" relative error", ""))
            axis.text(
                0.97,
                0.95,
                rf"$\Delta t={dt:g}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                    "pad": 2,
                },
            )
            axis.set_ylabel("Relative error")
            if row_index == 1:
                axis.set_xlabel("Complete loss + parameter/state VJP (ms)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
