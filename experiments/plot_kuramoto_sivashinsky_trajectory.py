"""Plot the saved Kuramoto--Sivashinsky held-out trajectory."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = (
    ROOT / "paper" / "figures" / "kuramoto_sivashinsky_composable_trajectory.csv"
)
RAW_PATH = (
    ROOT / "paper" / "figures" / "kuramoto_sivashinsky_composable_raw.csv"
)
OUTPUT_PATH = (
    ROOT / "paper" / "figures" / "kuramoto_sivashinsky_trajectory_plot.png"
)


def load_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with TRAJECTORY_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    times = np.array(sorted({float(row["time"]) for row in rows}))
    xs = np.array(sorted({float(row["x"]) for row in rows}))
    time_index = {value: index for index, value in enumerate(times)}
    x_index = {value: index for index, value in enumerate(xs)}
    reference = np.empty((len(times), len(xs)))
    prediction = np.empty_like(reference)

    for row in rows:
        i = time_index[float(row["time"])]
        j = x_index[float(row["x"])]
        reference[i, j] = float(row["reference"])
        prediction[i, j] = float(row["prediction"])

    return times, xs, reference, prediction


def load_best_seed() -> dict[str, str]:
    with RAW_PATH.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == "full_state_pullback"
        ]
    return min(rows, key=lambda row: float(row["parameter_relerr"]))


def main() -> None:
    times, xs, reference, prediction = load_trajectory()
    best = load_best_seed()
    difference = reference - reference[0]
    absolute_error = np.abs(prediction - reference)
    domain_length = 22.0
    training_horizon = 0.125

    field_limit = np.max(np.abs(reference))
    change_limit = np.max(np.abs(difference))
    error_limit = np.max(absolute_error)
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
    change_norm = TwoSlopeNorm(
        vmin=-change_limit,
        vcenter=0.0,
        vmax=change_limit,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure = plt.figure(figsize=(13.2, 8.4))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.9],
        left=0.065,
        right=0.965,
        bottom=0.14,
        top=0.80,
        hspace=0.38,
        wspace=0.42,
    )

    extent = (0.0, domain_length, times[0], times[-1])
    panels = [
        (
            reference,
            "Reference field",
            "RdBu_r",
            field_norm,
            r"$u(x,t)$",
        ),
        (
            difference,
            "Change from the initial field",
            "RdBu_r",
            change_norm,
            r"$u(x,t)-u(x,0)$",
        ),
        (
            absolute_error,
            "Learned rollout error",
            "magma",
            None,
            r"$|\hat{u}(x,t)-u(x,t)|$",
        ),
    ]

    for column, (values, title, cmap, norm, colorbar_label) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        if norm is None:
            image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=0.0,
                vmax=error_limit,
                interpolation="nearest",
            )
        else:
            image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
        axis.axhline(
            training_horizon,
            color="white" if column != 1 else "#27272a",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            alpha=0.95,
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(r"Position $x$")
        if column == 0:
            axis.set_ylabel(r"Time $t$")
            axis.text(
                0.45,
                training_horizon + 0.006,
                "end of training horizon",
                color="white",
                fontsize=8.5,
                va="bottom",
            )
        else:
            axis.set_yticklabels([])
        colorbar = figure.colorbar(image, ax=axis, shrink=0.88, pad=0.025)
        colorbar.set_label(colorbar_label)

    snapshot_axis = figure.add_subplot(grid[1, :])
    snapshot_times = [0.0, 0.125, 0.25]
    colors = ["#2563eb", "#d97706", "#15803d"]
    for target_time, color in zip(snapshot_times, colors):
        index = int(np.argmin(np.abs(times - target_time)))
        snapshot_axis.plot(
            xs,
            reference[index],
            color=color,
            linewidth=2.0,
            label=rf"reference, $t={times[index]:.3f}$",
        )
        if target_time > 0:
            snapshot_axis.plot(
                xs,
                prediction[index],
                color=color,
                linewidth=1.4,
                linestyle=(0, (4, 3)),
                label=rf"learned, $t={times[index]:.3f}$",
            )

    snapshot_axis.axhline(0.0, color="#a1a1aa", linewidth=0.8)
    snapshot_axis.set_xlim(0.0, domain_length)
    snapshot_axis.set_xlabel(r"Position $x$")
    snapshot_axis.set_ylabel(r"Field value $u$")
    snapshot_axis.set_title(
        "Representative spatial snapshots",
        loc="left",
        fontweight="bold",
    )
    snapshot_axis.grid(axis="y", color="#e4e4e7", linewidth=0.8)
    snapshot_axis.spines[["top", "right"]].set_visible(False)
    snapshot_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    relative_change = np.linalg.norm(reference[-1] - reference[0]) / np.linalg.norm(
        reference[0]
    )
    figure.suptitle(
        "Kuramoto–Sivashinsky held-out rollout",
        fontsize=16,
        fontweight="bold",
        x=0.055,
        y=0.955,
        ha="left",
    )
    figure.text(
        0.055,
        0.905,
        (
            "Velocity form on a periodic domain of length 22. "
            f"Best full-pullback seed {best['seed']}, 64 grid points, "
            r"$\Delta t=0.0125$."
        ),
        fontsize=10.5,
        color="#52525b",
        ha="left",
    )
    figure.text(
        0.055,
        0.872,
        (
            f"Final-state change {100 * relative_change:.1f}%   |   "
            f"rollout error {float(best['heldout_rollout_relerr']):.2e}"
        ),
        fontsize=10.5,
        color="#52525b",
        ha="left",
    )

    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
