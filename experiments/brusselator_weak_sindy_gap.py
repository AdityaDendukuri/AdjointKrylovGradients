"""Temporal-gap stress test: weak-form SINDy versus the CME semigroup.

This is deliberately stronger than applying ordinary Weak SINDy to the two
path coordinates X(t), Y(t). The weak fits receive the same full-distribution
snapshots, 13-by-13 state grid, contexts, initial states, and 40 sparse CME
generator directions as the finite-time experiment in the SISC manuscript.

The weak baseline uses

    -∫ phi'(t) p(t) dt = sum_j theta_j ∫ phi(t) D_j p(t) dt,

with the compactly supported polynomial test functions and piecewise-linear
integration weights constructed by PySINDy's WeakPDELibrary.  A deterministic
interval/box-test variant is also evaluated:

    p(t_{i+1}) - p(t_i)
        ≈ sum_j theta_j Δt_i D_j (p(t_i) + p(t_{i+1})) / 2.

Thus the comparison does not rely on a trajectory-level finite-difference
strawman.  It asks whether weak temporal quadrature remains reliable when only
widely separated distribution snapshots are available.  At the true
parameters the semigroup relation is exact at every observation spacing,
whereas the weak regressions inherit temporal interpolation/quadrature error.

Run:

    .venv-matfree/bin/python experiments/brusselator_weak_sindy_gap.py

Useful environment variables:

    WEAK_SINDY_SMOKE=1
    WEAK_SINDY_XMAX=12
    WEAK_SINDY_YMAX=12
    WEAK_SINDY_K=20
    WEAK_SINDY_OUTPUT_DIR=paper/figures

Outputs:

    brusselator_weak_sindy_gap.csv
    brusselator_weak_sindy_configs.csv
"""

from __future__ import annotations

import csv
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysindy as ps
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import _cd_fast


CHANNELS = ((1, 0), (-1, 0), (-1, 1), (1, -1))
MONOMIALS = tuple((a, degree - a) for degree in range(4) for a in range(degree + 1))
M = len(CHANNELS) * len(MONOMIALS)
TRUE_SUPPORT = (0, 12, 22, 38)  # zero-based: Julia directions 1, 13, 23, 39

TRAIN_CONTEXTS = (
    (0.80, 2.40),
    (0.80, 3.60),
    (1.40, 2.40),
    (1.40, 3.60),
)
TEST_CONTEXTS = (
    (0.90, 2.50),
    (1.10, 2.90),
    (1.00, 3.40),
    (1.30, 3.10),
)
INITIAL_STATES = tuple(
    (x_initial, y_initial)
    for x_initial in (1, 3, 5)
    for y_initial in (0, 2)
)

L1_RATIOS = (0.5, 0.9, 1.0)
ALPHA_MULTIPLIERS = 10.0 ** np.linspace(-5.0, -0.2, 12)
CUTOFFS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.75, 1.00)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def comb(value: int, order: int) -> float:
    if value < order:
        return 0.0
    return float(math.comb(value, order))


def build_cme_library(x_max: int, y_max: int) -> list[sparse.csc_matrix]:
    n = (x_max + 1) * (y_max + 1)

    def linear_index(x: int, y: int) -> int:
        return x * (y_max + 1) + y

    directions: list[sparse.csc_matrix] = []
    for dx, dy in CHANNELS:
        for a, b in MONOMIALS:
            rows: list[int] = []
            cols: list[int] = []
            values: list[float] = []
            for x in range(x_max + 1):
                for y in range(y_max + 1):
                    propensity = comb(x, a) * comb(y, b)
                    if propensity == 0.0:
                        continue
                    x_next, y_next = x + dx, y + dy
                    if not (0 <= x_next <= x_max and 0 <= y_next <= y_max):
                        continue
                    source = linear_index(x, y)
                    target = linear_index(x_next, y_next)
                    rows.extend((target, source))
                    cols.extend((source, source))
                    values.extend((propensity, -propensity))
            directions.append(
                sparse.csc_matrix((values, (rows, cols)), shape=(n, n))
            )
    return directions


def true_theta(context: tuple[float, float]) -> np.ndarray:
    a_context, b_context = context
    theta = np.zeros(M)
    theta[0] = a_context
    theta[12] = 1.0
    theta[22] = b_context
    theta[38] = 1.0
    return theta


def true_beta() -> np.ndarray:
    """Return the M x 3 coefficient matrix for theta(A,B)=beta@[1,A,B]."""
    beta = np.zeros((M, 3))
    beta[0, 1] = 1.0
    beta[12, 0] = 1.0
    beta[22, 2] = 1.0
    beta[38, 0] = 1.0
    return beta


def assemble_generator(
    directions: list[sparse.csc_matrix], theta: np.ndarray
) -> sparse.csc_matrix:
    generator = sparse.csc_matrix(directions[0].shape)
    for coefficient, direction in zip(theta, directions):
        if coefficient != 0.0:
            generator = generator + coefficient * direction
    return generator.tocsc()


def initial_distribution(
    initial_state: tuple[int, int], x_max: int, y_max: int
) -> np.ndarray:
    x_initial, y_initial = initial_state
    x_initial = min(max(x_initial, 0), x_max)
    state = np.zeros((x_max + 1) * (y_max + 1))
    state[x_initial * (y_max + 1) + min(y_initial, y_max)] = 1.0
    return state


def exact_trajectory(
    generator: sparse.csc_matrix, initial: np.ndarray, dense_times: np.ndarray
) -> np.ndarray:
    trace = float(generator.diagonal().sum())
    return np.asarray(
        expm_multiply(
            generator,
            initial,
            start=float(dense_times[0]),
            stop=float(dense_times[-1]),
            num=len(dense_times),
            endpoint=True,
            traceA=trace,
        )
    )


@dataclass
class SufficientStatistics:
    gram_sum: np.ndarray
    cross_sum: np.ndarray
    response_norm_sum: float
    true_residual_sum: float
    n_rows: int

    @classmethod
    def empty(cls) -> "SufficientStatistics":
        width = 3 * M
        return cls(
            gram_sum=np.zeros((width, width)),
            cross_sum=np.zeros(width),
            response_norm_sum=0.0,
            true_residual_sum=0.0,
            n_rows=0,
        )

    def add_window(
        self,
        response: np.ndarray,
        integrated_state: np.ndarray,
        context: tuple[float, float],
        direction_stack: sparse.csr_matrix,
    ) -> None:
        state_dimension = len(integrated_state)
        base = np.asarray(direction_stack @ integrated_state).reshape(
            M, state_dimension
        ).T
        z = np.array((1.0, context[0], context[1]))
        base_gram = base.T @ base
        base_cross = base.T @ response
        self.gram_sum += np.kron(base_gram, np.outer(z, z))
        self.cross_sum += (base_cross[:, None] * z[None, :]).reshape(-1)
        self.response_norm_sum += float(response @ response)

        true_prediction = base @ true_theta(context)
        residual = response - true_prediction
        self.true_residual_sum += float(residual @ residual)
        self.n_rows += len(response)


@dataclass
class RegressionStats:
    gram: np.ndarray
    cross: np.ndarray
    scales: np.ndarray
    response_norm: float


def standardize_statistics(stats: SufficientStatistics) -> RegressionStats:
    gram_raw = stats.gram_sum / stats.n_rows
    cross_raw = stats.cross_sum / stats.n_rows
    scales = np.maximum(np.sqrt(np.maximum(np.diag(gram_raw), 0.0)), 1e-12)
    return RegressionStats(
        gram=np.ascontiguousarray(gram_raw / np.outer(scales, scales)),
        cross=np.ascontiguousarray(cross_raw / scales),
        scales=scales,
        response_norm=stats.response_norm_sum / stats.n_rows,
    )


def nonnegative_elastic_net(
    stats: RegressionStats,
    alpha: float,
    l1_ratio: float,
    initial: np.ndarray | None = None,
    max_sweeps: int = 20_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    gamma = (
        np.zeros_like(stats.cross)
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    ridge = alpha * (1.0 - l1_ratio)
    lasso = alpha * l1_ratio
    # PySINDy depends on scikit-learn; use its compiled Gram-coordinate descent
    # kernel here so an oracle sweep over the same elastic-net grid remains
    # practical without materializing the million-row weak design matrix.
    with warnings.catch_warnings():
        # Highly correlated distractor directions can leave a very small
        # nonzero dual gap even after 20k sweeps.  The largest gaps observed in
        # the oracle path are around 1e-9 in standardized objective units and
        # do not change the selected support under a 1e-8 coefficient check.
        warnings.simplefilter("ignore", ConvergenceWarning)
        gamma, _, _, _ = _cd_fast.enet_coordinate_descent_gram(
            np.ascontiguousarray(gamma),
            lasso,
            ridge,
            stats.gram,
            stats.cross,
            np.array([math.sqrt(max(stats.response_norm, 0.0))]),
            max_sweeps,
            tolerance,
            np.random.RandomState(0),
            random=0,
            positive=1,
        )
    return np.asarray(gamma) / stats.scales


def support_metrics(beta: np.ndarray, contexts: tuple[tuple[float, float], ...]):
    z = np.array([(1.0, context[0], context[1]) for context in contexts])
    rates = z @ beta.T
    scores = np.max(np.abs(rates), axis=0)
    return rates, scores


def better_result(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    return (
        candidate["true_hits"] > incumbent["true_hits"]
        or (
            candidate["true_hits"] == incumbent["true_hits"]
            and candidate["spurious"] < incumbent["spurious"]
        )
        or (
            candidate["true_hits"] == incumbent["true_hits"]
            and candidate["spurious"] == incumbent["spurious"]
            and candidate["train_coefficient_error"]
            < incumbent["train_coefficient_error"]
        )
    )


def oracle_fit(
    sufficient: SufficientStatistics,
    contexts: tuple[tuple[float, float], ...],
) -> dict:
    stats = standardize_statistics(sufficient)
    alpha_scale = max(float(np.max(np.abs(stats.cross))), 1e-12)
    true_rates = np.array([true_theta(context) for context in contexts])
    true_support = set(TRUE_SUPPORT)
    best: dict | None = None

    for l1_ratio in L1_RATIOS:
        gamma_initial = None
        for multiplier in ALPHA_MULTIPLIERS[::-1]:
            alpha = alpha_scale * multiplier
            physical = nonnegative_elastic_net(
                stats,
                alpha=alpha,
                l1_ratio=l1_ratio,
                initial=gamma_initial,
            )
            gamma_initial = physical * stats.scales
            beta_unthresholded = physical.reshape(M, 3)
            _, scores = support_metrics(beta_unthresholded, contexts)

            for cutoff in CUTOFFS:
                beta = beta_unthresholded.copy()
                active = scores >= cutoff
                beta[~active] = 0.0
                support = set(np.flatnonzero(active))
                predicted_rates, _ = support_metrics(beta, contexts)
                candidate = {
                    "true_hits": len(support & true_support),
                    "spurious": len(support - true_support),
                    "support": sorted(int(index) + 1 for index in support),
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "cutoff": cutoff,
                    "beta": beta,
                    "train_coefficient_error": float(
                        np.linalg.norm(predicted_rates - true_rates)
                        / np.linalg.norm(true_rates)
                    ),
                }
                if better_result(candidate, best):
                    best = candidate
    assert best is not None
    return best


def interval_statistics(
    trajectories: dict[tuple[tuple[float, float], int], np.ndarray],
    selected_indices: np.ndarray,
    dense_times: np.ndarray,
    contexts: tuple[tuple[float, float], ...],
    direction_stack: sparse.csr_matrix,
) -> SufficientStatistics:
    stats = SufficientStatistics.empty()
    times = dense_times[selected_indices]
    for context in contexts:
        for initial_state in INITIAL_STATES:
            trajectory = trajectories[(context, initial_state)][selected_indices]
            for index, delta_t in enumerate(np.diff(times)):
                response = trajectory[index + 1] - trajectory[index]
                integrated_state = (
                    0.5 * delta_t * (trajectory[index] + trajectory[index + 1])
                )
                stats.add_window(
                    response, integrated_state, context, direction_stack
                )
    return stats


def pysindy_statistics(
    trajectories: dict[tuple[tuple[float, float], int], np.ndarray],
    selected_indices: np.ndarray,
    dense_times: np.ndarray,
    contexts: tuple[tuple[float, float], ...],
    direction_stack: sparse.csr_matrix,
    half_width: float,
    polynomial_degree: int,
    domain_seed: int,
    domain_count: int,
) -> tuple[SufficientStatistics, int]:
    times = dense_times[selected_indices]
    np.random.seed(domain_seed)
    library = ps.WeakPDELibrary(
        spatiotemporal_grid=times,
        K=domain_count,
        H_xt=half_width,
        p=polynomial_degree,
    )
    unique_domains = len({tuple(indices[0]) for indices in library.inds_k})

    stats = SufficientStatistics.empty()
    for context in contexts:
        for initial_state in INITIAL_STATES:
            trajectory = trajectories[(context, initial_state)][selected_indices]
            for domain in range(library.K):
                indices = np.asarray(library.inds_k[domain][0], dtype=int)
                response = np.tensordot(
                    library.fulltweights[domain],
                    -trajectory[indices],
                    axes=(0, 0),
                )
                integrated_state = np.tensordot(
                    library.fullweights0[domain],
                    trajectory[indices],
                    axes=(0, 0),
                )
                stats.add_window(
                    response, integrated_state, context, direction_stack
                )
    return stats, unique_domains


def fitted_rate_error(beta: np.ndarray, contexts) -> float:
    z = np.array([(1.0, context[0], context[1]) for context in contexts])
    predicted = z @ beta.T
    reference = np.array([true_theta(context) for context in contexts])
    return float(np.linalg.norm(predicted - reference) / np.linalg.norm(reference))


def fitted_semigroup_error(
    beta: np.ndarray,
    trajectories: dict[tuple[tuple[float, float], int], np.ndarray],
    selected_indices: np.ndarray,
    dense_times: np.ndarray,
    contexts: tuple[tuple[float, float], ...],
    directions: list[sparse.csc_matrix],
) -> float:
    squared_error = 0.0
    squared_reference = 0.0
    times = dense_times[selected_indices]
    for context in contexts:
        z = np.array((1.0, context[0], context[1]))
        fitted_generator = assemble_generator(directions, beta @ z)
        trace = float(fitted_generator.diagonal().sum())
        for initial_state in INITIAL_STATES:
            reference = trajectories[(context, initial_state)][selected_indices]
            initial = reference[0]
            if np.allclose(np.diff(times), np.diff(times)[0]):
                prediction = np.asarray(
                    expm_multiply(
                        fitted_generator,
                        initial,
                        start=float(times[0]),
                        stop=float(times[-1]),
                        num=len(times),
                        endpoint=True,
                        traceA=trace,
                    )
                )
            else:
                prediction = np.vstack(
                    [
                        expm_multiply(
                            fitted_generator * float(time),
                            initial,
                            traceA=trace * float(time),
                        )
                        for time in times
                    ]
                )
            squared_error += float(np.sum((prediction - reference) ** 2))
            squared_reference += float(np.sum(reference**2))
    return math.sqrt(squared_error / squared_reference)


def semigroup_consistency_error(
    trajectory: np.ndarray,
    selected_indices: np.ndarray,
    dense_times: np.ndarray,
    generator: sparse.csc_matrix,
) -> float:
    times = dense_times[selected_indices]
    squared_error = 0.0
    squared_reference = 0.0
    trace = float(generator.diagonal().sum())
    selected = trajectory[selected_indices]
    for index, delta_t in enumerate(np.diff(times)):
        prediction = expm_multiply(
            generator * float(delta_t),
            selected[index],
            traceA=trace * float(delta_t),
        )
        squared_error += float(np.sum((prediction - selected[index + 1]) ** 2))
        squared_reference += float(np.sum(selected[index + 1] ** 2))
    return math.sqrt(squared_error / squared_reference)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    smoke = os.environ.get("WEAK_SINDY_SMOKE", "0") == "1"
    x_max = env_int("WEAK_SINDY_XMAX", 12)
    y_max = env_int("WEAK_SINDY_YMAX", 12)
    domain_count = env_int("WEAK_SINDY_K", 4 if smoke else 20)
    output_dir = Path(
        os.environ.get("WEAK_SINDY_OUTPUT_DIR", "paper/figures")
    )

    train_contexts = TRAIN_CONTEXTS[:2] if smoke else TRAIN_CONTEXTS
    test_contexts = TEST_CONTEXTS[:1] if smoke else TEST_CONTEXTS
    all_contexts = train_contexts + test_contexts

    dense_times = np.linspace(0.0, 0.75, 76)
    grids = {
        "dense_dt_0.01": np.arange(0, 76, 1, dtype=int),
        "medium_dt_0.05": np.arange(0, 76, 5, dtype=int),
        "coarse_dt_0.15": np.arange(0, 76, 15, dtype=int),
        "submitted_gaps": np.array((0, 5, 25, 75), dtype=int),
    }
    if smoke:
        grids = {
            "medium_dt_0.05": grids["medium_dt_0.05"],
            "submitted_gaps": grids["submitted_gaps"],
        }

    print(
        f"Building 40-direction CME library on "
        f"{x_max + 1}x{y_max + 1} states..."
    )
    directions = build_cme_library(x_max, y_max)
    direction_stack = sparse.vstack(directions, format="csr")
    trajectories: dict[tuple[tuple[float, float], int], np.ndarray] = {}
    generators: dict[tuple[float, float], sparse.csc_matrix] = {}
    for context_index, context in enumerate(all_contexts, start=1):
        generator = assemble_generator(directions, true_theta(context))
        generators[context] = generator
        for initial_state in INITIAL_STATES:
            initial = initial_distribution(initial_state, x_max, y_max)
            trajectories[(context, initial_state)] = exact_trajectory(
                generator, initial, dense_times
            )
        print(f"  exact context {context_index}/{len(all_contexts)}: {context}")

    summary_rows: list[dict] = []
    configuration_rows: list[dict] = []
    beta_reference = true_beta()

    for grid_name, selected_indices in grids.items():
        times = dense_times[selected_indices]
        max_gap = float(np.max(np.diff(times)))
        print(
            f"\n{grid_name}: {len(times)} snapshots, max gap={max_gap:.3f}"
        )

        interval_stats = interval_statistics(
            trajectories,
            selected_indices,
            dense_times,
            train_contexts,
            direction_stack,
        )
        interval_fit = oracle_fit(interval_stats, train_contexts)
        interval_row = {
            "grid": grid_name,
            "method": "interval_weak_trapezoid",
            "snapshots": len(times),
            "max_gap": max_gap,
            "half_width": "",
            "polynomial_degree": "",
            "domain_seed": "",
            "unique_domains": len(times) - 1,
            "true_hits": interval_fit["true_hits"],
            "spurious": interval_fit["spurious"],
            "support": str(interval_fit["support"]),
            "train_coefficient_error": interval_fit["train_coefficient_error"],
            "test_coefficient_error": fitted_rate_error(
                interval_fit["beta"], test_contexts
            ),
            "test_semigroup_error": fitted_semigroup_error(
                interval_fit["beta"],
                trajectories,
                selected_indices,
                dense_times,
                test_contexts,
                directions,
            ),
            "true_parameter_weak_residual": math.sqrt(
                interval_stats.true_residual_sum
                / interval_stats.response_norm_sum
            ),
            "true_parameter_semigroup_residual": semigroup_consistency_error(
                trajectories[(train_contexts[0], INITIAL_STATES[0])],
                selected_indices,
                dense_times,
                generators[train_contexts[0]],
            ),
            "alpha": interval_fit["alpha"],
            "l1_ratio": interval_fit["l1_ratio"],
            "cutoff": interval_fit["cutoff"],
        }
        summary_rows.append(interval_row)
        print(
            "  interval weak: "
            f"{interval_fit['true_hits']}/4 true, "
            f"{interval_fit['spurious']} spurious, "
            f"true-theta residual="
            f"{interval_row['true_parameter_weak_residual']:.2e}"
        )

        best_pysindy: tuple[dict, dict] | None = None
        half_widths = (0.10, 0.20, 0.30, 0.36)
        polynomial_degrees = (4,)
        domain_seeds = (0,) if smoke else (0, 1, 2)
        for half_width in half_widths:
            if half_width >= 0.5 * (times[-1] - times[0]):
                continue
            for polynomial_degree in polynomial_degrees:
                for domain_seed in domain_seeds:
                    config = {
                        "grid": grid_name,
                        "half_width": half_width,
                        "polynomial_degree": polynomial_degree,
                        "domain_seed": domain_seed,
                        "domain_count": domain_count,
                    }
                    try:
                        weak_stats, unique_domains = pysindy_statistics(
                            trajectories,
                            selected_indices,
                            dense_times,
                            train_contexts,
                            direction_stack,
                            half_width,
                            polynomial_degree,
                            domain_seed,
                            domain_count,
                        )
                        fit = oracle_fit(weak_stats, train_contexts)
                        weak_residual = math.sqrt(
                            weak_stats.true_residual_sum
                            / weak_stats.response_norm_sum
                        )
                        config.update(
                            {
                                "status": "ok",
                                "unique_domains": unique_domains,
                                "true_hits": fit["true_hits"],
                                "spurious": fit["spurious"],
                                "train_coefficient_error": fit[
                                    "train_coefficient_error"
                                ],
                                "true_parameter_weak_residual": weak_residual,
                                "error": "",
                            }
                        )
                        if best_pysindy is None or better_result(
                            fit, best_pysindy[0]
                        ):
                            best_pysindy = (fit, config.copy())
                    except Exception as error:  # record sparse-grid API failures
                        config.update(
                            {
                                "status": "failed",
                                "unique_domains": "",
                                "true_hits": "",
                                "spurious": "",
                                "train_coefficient_error": "",
                                "true_parameter_weak_residual": "",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    configuration_rows.append(config)

        if best_pysindy is None:
            print("  PySINDy weak: no valid subdomain configuration")
        else:
            fit, config = best_pysindy
            pysindy_row = {
                "grid": grid_name,
                "method": "pysindy_weighted_weak_cme",
                "snapshots": len(times),
                "max_gap": max_gap,
                "half_width": config["half_width"],
                "polynomial_degree": config["polynomial_degree"],
                "domain_seed": config["domain_seed"],
                "unique_domains": config["unique_domains"],
                "true_hits": fit["true_hits"],
                "spurious": fit["spurious"],
                "support": str(fit["support"]),
                "train_coefficient_error": fit["train_coefficient_error"],
                "test_coefficient_error": fitted_rate_error(
                    fit["beta"], test_contexts
                ),
                "test_semigroup_error": fitted_semigroup_error(
                    fit["beta"],
                    trajectories,
                    selected_indices,
                    dense_times,
                    test_contexts,
                    directions,
                ),
                "true_parameter_weak_residual": config[
                    "true_parameter_weak_residual"
                ],
                "true_parameter_semigroup_residual": interval_row[
                    "true_parameter_semigroup_residual"
                ],
                "alpha": fit["alpha"],
                "l1_ratio": fit["l1_ratio"],
                "cutoff": fit["cutoff"],
            }
            summary_rows.append(pysindy_row)
            print(
                "  PySINDy weak (oracle H,p,seed): "
                f"{fit['true_hits']}/4 true, {fit['spurious']} spurious, "
                f"true-theta residual="
                f"{config['true_parameter_weak_residual']:.2e}"
            )

        reference_error = fitted_rate_error(beta_reference, test_contexts)
        assert reference_error == 0.0

    write_rows(
        output_dir / "brusselator_weak_sindy_gap.csv",
        summary_rows,
    )
    write_rows(
        output_dir / "brusselator_weak_sindy_configs.csv",
        configuration_rows,
    )
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
