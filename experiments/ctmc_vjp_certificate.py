"""
Prototype residual-based physical-VJP bounds for a CTMC generator library.

This experiment implements the path-residual component of the certificate
derived in ``invariant_vjp_audit.md``.  It uses the contractivity of a Markov
semigroup in the 1-norm and of its adjoint in the infinity norm.

For approximate forward and adjoint paths,

    r_f = phi_hat' - dt * A phi_hat
    r_a = lambda_hat' + dt * A.T lambda_hat,

Duhamel's formula gives computable path-error bounds.  These are propagated
to one bound per generator direction.  Dense paths and ``expm_frechet`` are
used only to validate coverage; they are not inputs to the bound.

The numerical integration of residual norms and the nested Gauss estimate
are not yet interval-rigorous.  Accordingly, this driver calls the result a
``certificate prototype`` rather than a completed mathematical certificate.

Run
---

    .venv-matfree/bin/python experiments/ctmc_vjp_certificate.py

Environment variables
---------------------

    CERTIFICATE_M_VALUES=3,4,5,6,8,10,12
    CERTIFICATE_EPS_VALUES=1e-10,0.2
    CERTIFICATE_DT=1.0
    CERTIFICATE_N_QUAD=12
    CERTIFICATE_GRID_SIZE=2001
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from numpy.polynomial.legendre import leggauss

from invariant_vjp_audit import arnoldi_full
from stationary_start_ctmc import (
    assemble,
    exact_predictions,
    initial_state,
    make_problem,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT = (
    ROOT / "paper" / "figures" / "ctmc_vjp_certificate_raw.csv"
)
SUMMARY_OUTPUT = (
    ROOT / "paper" / "figures" / "ctmc_vjp_certificate_summary.csv"
)


def parse_values(name: str, default: str, cast):
    return tuple(cast(value) for value in os.getenv(name, default).split(","))


def relative_error(value, reference):
    value = np.asarray(value)
    reference = np.asarray(reference)
    return float(
        np.linalg.norm(value - reference)
        / max(np.linalg.norm(reference), 1.0e-14)
    )


def trapezoid_cumulative_forward(values, grid):
    increments = (
        0.5
        * (values[:-1] + values[1:])
        * np.diff(grid)
    )
    return np.concatenate((np.zeros(1), np.cumsum(increments)))


def trapezoid_cumulative_backward(values, grid):
    increments = (
        0.5
        * (values[:-1] + values[1:])
        * np.diff(grid)
    )
    return np.concatenate(
        (np.cumsum(increments[::-1])[::-1], np.zeros(1))
    )


def reduced_path(small_matrix, beta, times):
    first = np.zeros(len(small_matrix), dtype=small_matrix.dtype)
    first[0] = beta
    times = np.asarray(times)
    return np.column_stack(
        [
            scipy.linalg.expm(time_value * small_matrix) @ first
            for time_value in times
        ]
    )


def krylov_path(matrix, vector, dimension, scaled_times):
    basis, small_matrix, beta = arnoldi_full(
        matrix, vector, dimension
    )
    reduced = reduced_path(small_matrix, beta, scaled_times)
    return basis, small_matrix, reduced, basis @ reduced


def exact_parameter_vjp(matrix, directions, state, upstream, dt):
    gradient = np.empty(len(directions))
    for index, direction in enumerate(directions):
        frechet = scipy.linalg.expm_frechet(
            dt * matrix, dt * direction, compute_expm=False
        )
        gradient[index] = np.vdot(
            upstream, frechet @ state
        ).real
    return gradient


def gauss_gradient(
    matrix,
    directions,
    state,
    upstream,
    dt,
    dimension,
    quadrature_order,
):
    nodes, weights = leggauss(quadrature_order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    _, _, _, forward = krylov_path(
        matrix, state, dimension, dt * nodes
    )
    _, _, _, adjoint_reversed = krylov_path(
        matrix.T, upstream, dimension, dt * (1.0 - nodes)
    )
    direction_paths = np.einsum(
        "mij,jq->miq", directions, forward
    )
    integrands = np.einsum(
        "iq,miq->mq", adjoint_reversed, direction_paths
    )
    return dt * (integrands @ weights)


def certificate_rows(
    problem,
    epsilon,
    dt,
    dimension,
    quadrature_order,
    grid_size,
):
    matrix = problem["base"]
    directions = problem["directions"]
    state = initial_state(problem, epsilon)
    target = exact_predictions(
        problem, problem["theta_true"], state, (dt,)
    )[0]

    grid = np.linspace(0.0, 1.0, grid_size)
    (
        forward_basis,
        forward_small,
        forward_reduced,
        forward_path,
    ) = krylov_path(matrix, state, dimension, dt * grid)
    prediction = forward_path[:, -1]
    upstream = prediction - target
    (
        adjoint_basis,
        adjoint_small,
        adjoint_reduced_forward,
        adjoint_path_forward,
    ) = krylov_path(matrix.T, upstream, dimension, dt * grid)
    adjoint_path = adjoint_path_forward[:, ::-1]
    adjoint_reduced = adjoint_reduced_forward[:, ::-1]

    # The residual expressions use only sparse-operator actions, bases, and
    # reduced paths.  Dense matrices are used here solely because n=24 makes
    # the validation experiment transparent.
    forward_residual_operator = (
        forward_basis @ forward_small
        - matrix @ forward_basis
    )
    forward_residual = (
        dt * forward_residual_operator @ forward_reduced
    )
    adjoint_residual_operator = (
        matrix.T @ adjoint_basis
        - adjoint_basis @ adjoint_small
    )
    adjoint_residual = (
        dt * adjoint_residual_operator @ adjoint_reduced
    )

    forward_residual_norm = np.linalg.norm(
        forward_residual, ord=1, axis=0
    )
    adjoint_residual_norm = np.linalg.norm(
        adjoint_residual, ord=np.inf, axis=0
    )
    forward_error_bound = trapezoid_cumulative_forward(
        forward_residual_norm, grid
    )
    adjoint_error_bound = trapezoid_cumulative_backward(
        adjoint_residual_norm, grid
    )

    exact_forward_path = scipy.sparse.linalg.expm_multiply(
        dt * matrix,
        state,
        start=0.0,
        stop=1.0,
        num=grid_size,
        endpoint=True,
    ).T
    exact_adjoint_forward = scipy.sparse.linalg.expm_multiply(
        dt * matrix.T,
        upstream,
        start=0.0,
        stop=1.0,
        num=grid_size,
        endpoint=True,
    ).T
    exact_adjoint_path = exact_adjoint_forward[:, ::-1]
    forward_error_actual = np.linalg.norm(
        exact_forward_path - forward_path, ord=1, axis=0
    )
    adjoint_error_actual = np.linalg.norm(
        exact_adjoint_path - adjoint_path, ord=np.inf, axis=0
    )

    direction_paths = np.einsum(
        "mij,jq->miq", directions, forward_path
    )
    approximate_integrands = np.einsum(
        "iq,miq->mq", adjoint_path, direction_paths
    )
    high_resolution_gradient = dt * np.trapezoid(
        approximate_integrands, grid, axis=1
    )
    exact_gradient = exact_parameter_vjp(
        matrix, directions, state, upstream, dt
    )
    gauss_gradient_primary = gauss_gradient(
        matrix,
        directions,
        state,
        upstream,
        dt,
        dimension,
        quadrature_order,
    )
    gauss_gradient_nested = gauss_gradient(
        matrix,
        directions,
        state,
        upstream,
        dt,
        dimension,
        2 * quadrature_order,
    )

    direction_transpose_adjoint = np.einsum(
        "mji,jq->miq", directions, adjoint_path
    )
    direction_norms = np.linalg.norm(directions, ord=1, axis=(1, 2))
    term_forward = (
        np.max(np.abs(direction_transpose_adjoint), axis=1)
        * forward_error_bound[None, :]
    )
    term_adjoint = (
        np.sum(np.abs(direction_paths), axis=1)
        * adjoint_error_bound[None, :]
    )
    term_product = (
        direction_norms[:, None]
        * forward_error_bound[None, :]
        * adjoint_error_bound[None, :]
    )
    path_bound = dt * np.trapezoid(
        term_forward + term_adjoint + term_product,
        grid,
        axis=1,
    )

    path_error_actual = np.abs(
        exact_gradient - high_resolution_gradient
    )
    quadrature_error_actual = np.abs(
        gauss_gradient_primary - high_resolution_gradient
    )
    quadrature_error_estimate = np.abs(
        gauss_gradient_primary - gauss_gradient_nested
    )
    total_error_actual = np.abs(
        exact_gradient - gauss_gradient_primary
    )
    total_bound_with_actual_quadrature = (
        path_bound + quadrature_error_actual
    )
    total_bound_with_nested_estimate = (
        path_bound + quadrature_error_estimate
    )

    true_support = set(problem["true_indices"].tolist())
    support_size = len(true_support)
    approximate_topk = set(
        np.argsort(np.abs(gauss_gradient_primary))[
            -support_size:
        ].tolist()
    )
    exact_topk = set(
        np.argsort(np.abs(exact_gradient))[-support_size:].tolist()
    )
    interval_lower = np.maximum(
        np.abs(gauss_gradient_primary)
        - total_bound_with_nested_estimate,
        0.0,
    )
    interval_upper = (
        np.abs(gauss_gradient_primary)
        + total_bound_with_nested_estimate
    )
    selected_indices = np.asarray(sorted(approximate_topk), dtype=int)
    unselected_indices = np.asarray(
        sorted(set(range(len(directions))) - approximate_topk),
        dtype=int,
    )
    support_ranking_margin = float(
        np.min(interval_lower[selected_indices])
        - np.max(interval_upper[unselected_indices])
    )
    support_ranking_certified = int(support_ranking_margin > 0.0)
    topk_matches_exact = int(approximate_topk == exact_topk)
    planted_support_certified = int(
        support_ranking_certified
        and approximate_topk == true_support
    )
    rows = []
    for index in range(len(directions)):
        rows.append(
            {
                "epsilon": epsilon,
                "dt": dt,
                "m": dimension,
                "n_quad": quadrature_order,
                "grid_size": grid_size,
                "direction": index,
                "is_true_direction": int(index in true_support),
                "exact_gradient": exact_gradient[index],
                "gauss_gradient": gauss_gradient_primary[index],
                "high_resolution_projected_gradient": (
                    high_resolution_gradient[index]
                ),
                "path_error_actual": path_error_actual[index],
                "path_error_bound": path_bound[index],
                "path_bound_covers": int(
                    path_error_actual[index]
                    <= path_bound[index] * (1.0 + 2.0e-5) + 1.0e-13
                ),
                "quadrature_error_actual": (
                    quadrature_error_actual[index]
                ),
                "quadrature_error_nested_estimate": (
                    quadrature_error_estimate[index]
                ),
                "total_error_actual": total_error_actual[index],
                "total_bound_with_actual_quadrature": (
                    total_bound_with_actual_quadrature[index]
                ),
                "total_bound_with_nested_estimate": (
                    total_bound_with_nested_estimate[index]
                ),
                "nested_total_bound_covers": int(
                    total_error_actual[index]
                    <= total_bound_with_nested_estimate[index]
                    * (1.0 + 2.0e-5)
                    + 1.0e-13
                ),
                "gradient_abs_interval_lower": interval_lower[index],
                "gradient_abs_interval_upper": interval_upper[index],
                "approximate_topk_support_overlap": len(
                    approximate_topk & true_support
                ),
                "exact_topk_support_overlap": len(
                    exact_topk & true_support
                ),
                "topk_matches_exact": topk_matches_exact,
                "support_ranking_margin": support_ranking_margin,
                "support_ranking_certified": support_ranking_certified,
                "planted_support_certified": (
                    planted_support_certified
                ),
                "forward_final_relerr": relative_error(
                    prediction, exact_forward_path[:, -1]
                ),
                "forward_max_actual_path_error_1": float(
                    np.max(forward_error_actual)
                ),
                "forward_max_residual_bound_1": float(
                    np.max(forward_error_bound)
                ),
                "adjoint_max_actual_path_error_inf": float(
                    np.max(adjoint_error_actual)
                ),
                "adjoint_max_residual_bound_inf": float(
                    np.max(adjoint_error_bound)
                ),
            }
        )
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["epsilon"], row["m"]), []).append(row)

    summaries = []
    for (epsilon, dimension), group in sorted(groups.items()):
        exact = np.asarray([row["exact_gradient"] for row in group])
        approximate = np.asarray(
            [row["gauss_gradient"] for row in group]
        )
        actual_errors = np.asarray(
            [row["total_error_actual"] for row in group]
        )
        estimated_bounds = np.asarray(
            [row["total_bound_with_nested_estimate"] for row in group]
        )
        positive_error = actual_errors > 1.0e-14
        ratios = np.divide(
            estimated_bounds,
            actual_errors,
            out=np.full_like(estimated_bounds, np.nan),
            where=positive_error,
        )
        summaries.append(
            {
                "epsilon": epsilon,
                "m": dimension,
                "n_quad": group[0]["n_quad"],
                "grid_size": group[0]["grid_size"],
                "forward_final_relerr": group[0][
                    "forward_final_relerr"
                ],
                "vjp_relerr": relative_error(approximate, exact),
                "path_bound_coverage_fraction": float(
                    np.mean(
                        [row["path_bound_covers"] for row in group]
                    )
                ),
                "nested_total_coverage_fraction": float(
                    np.mean(
                        [
                            row["nested_total_bound_covers"]
                            for row in group
                        ]
                    )
                ),
                "max_actual_total_error": float(
                    np.max(actual_errors)
                ),
                "max_estimated_total_bound": float(
                    np.max(estimated_bounds)
                ),
                "median_bound_to_error_ratio": float(
                    np.nanmedian(ratios)
                ),
                "max_bound_to_error_ratio": float(
                    np.nanmax(ratios)
                ),
                "approximate_topk_support_overlap": group[0][
                    "approximate_topk_support_overlap"
                ],
                "exact_topk_support_overlap": group[0][
                    "exact_topk_support_overlap"
                ],
                "topk_matches_exact": group[0][
                    "topk_matches_exact"
                ],
                "support_ranking_margin": group[0][
                    "support_ranking_margin"
                ],
                "support_ranking_certified": group[0][
                    "support_ranking_certified"
                ],
                "planted_support_certified": group[0][
                    "planted_support_certified"
                ],
                "forward_max_actual_path_error_1": group[0][
                    "forward_max_actual_path_error_1"
                ],
                "forward_max_residual_bound_1": group[0][
                    "forward_max_residual_bound_1"
                ],
                "adjoint_max_actual_path_error_inf": group[0][
                    "adjoint_max_actual_path_error_inf"
                ],
                "adjoint_max_residual_bound_inf": group[0][
                    "adjoint_max_residual_bound_inf"
                ],
            }
        )
    return summaries


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    dimensions = parse_values(
        "CERTIFICATE_M_VALUES", "3,4,5,6,8,10,12", int
    )
    epsilons = parse_values(
        "CERTIFICATE_EPS_VALUES", "1e-10,0.2", float
    )
    dt = float(os.getenv("CERTIFICATE_DT", "1.0"))
    quadrature_order = int(os.getenv("CERTIFICATE_N_QUAD", "12"))
    grid_size = int(os.getenv("CERTIFICATE_GRID_SIZE", "2001"))
    raw_output = Path(
        os.getenv("CERTIFICATE_RAW_OUTPUT", str(RAW_OUTPUT))
    )
    summary_output = Path(
        os.getenv("CERTIFICATE_SUMMARY_OUTPUT", str(SUMMARY_OUTPUT))
    )
    problem = make_problem(24, 16)

    print(
        "CTMC physical-VJP certificate prototype: "
        f"m={dimensions}, eps={epsilons}, dt={dt:g}, "
        f"nq={quadrature_order}, grid={grid_size}"
    )
    rows = []
    for epsilon in epsilons:
        for dimension in dimensions:
            current = certificate_rows(
                problem,
                epsilon,
                dt,
                dimension,
                quadrature_order,
                grid_size,
            )
            rows.extend(current)
            summary = summarize(current)[0]
            print(
                f"  eps={epsilon:8.1e} m={dimension:2d} "
                f"forward={summary['forward_final_relerr']:.2e} "
                f"vjp={summary['vjp_relerr']:.2e} "
                f"path-cover="
                f"{summary['path_bound_coverage_fraction']:.2f} "
                f"total-cover="
                f"{summary['nested_total_coverage_fraction']:.2f} "
                f"median-slack="
                f"{summary['median_bound_to_error_ratio']:.1f}x "
                f"top-k="
                f"{summary['approximate_topk_support_overlap']}/4 "
                f"exact-match={summary['topk_matches_exact']} "
                f"ranking-cert="
                f"{summary['support_ranking_certified']} "
                f"support-cert="
                f"{summary['planted_support_certified']}"
            )

    summaries = summarize(rows)
    if not all(row["path_bound_covers"] for row in rows):
        raise AssertionError(
            "At least one residual path bound missed its dense reference."
        )
    if not all(row["nested_total_bound_covers"] for row in rows):
        raise AssertionError(
            "At least one estimated total bound missed its dense reference."
        )
    write_csv(raw_output, rows)
    write_csv(summary_output, summaries)
    print(f"saved {raw_output}")
    print(f"saved {summary_output}")


if __name__ == "__main__":
    main()
