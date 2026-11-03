"""
Matched training comparison for two Arnoldi-gradient semantics.

The forward map is the same finite-m, fully reorthogonalized Arnoldi
approximation in both cases.  Only the parameter pullback changes:

  * ``discrete_arnoldi``: Matfree/Kraemer et al.'s exact VJP of the finite-m
    Arnoldi program.
  * ``continuous_frechet``: the paper's two-basis quadrature approximation to
    the Fréchet derivative of the exact semigroup.

All reported evaluation losses use SciPy's dense matrix exponential, so
neither method is evaluated using its own approximate training objective.

Examples:

    SEMANTICS_SMOKE=1 .venv-matfree/bin/python \
        experiments/gradient_semantics_training.py

    .venv-matfree/bin/python experiments/gradient_semantics_training.py

Environment variables:

    SEMANTICS_DT_VALUES=0.3,1.0
    SEMANTICS_M_VALUES=20,30,40,60
    SEMANTICS_SEEDS=11,22,33,44,55
    SEMANTICS_EPOCHS=250
    SEMANTICS_LR=0.05
    SEMANTICS_N_QUAD=8
    SEMANTICS_EVAL_EVERY=5
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.linalg
from matfree import decomp, funm
from numpy.polynomial.legendre import leggauss

from kraemer_comparison import build_fpe_library, diagonal_representation


ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT = ROOT / "paper" / "figures" / "gradient_semantics_training_raw.csv"
SUMMARY_OUTPUT = (
    ROOT / "paper" / "figures" / "gradient_semantics_training_summary.csv"
)


def parse_values(name: str, default: str, cast):
    return tuple(cast(value) for value in os.getenv(name, default).split(","))


SMOKE = os.getenv("SEMANTICS_SMOKE", "0") == "1"
DT_VALUES = parse_values(
    "SEMANTICS_DT_VALUES", "1.0" if SMOKE else "0.3,1.0", float
)
M_VALUES = parse_values(
    "SEMANTICS_M_VALUES", "20" if SMOKE else "20,30,40,60", int
)
SEEDS = parse_values(
    "SEMANTICS_SEEDS", "11" if SMOKE else "11,22,33,44,55", int
)
EPOCHS = int(os.getenv("SEMANTICS_EPOCHS", "30" if SMOKE else "250"))
LEARNING_RATE = float(os.getenv("SEMANTICS_LR", "0.05"))
N_QUAD = int(os.getenv("SEMANTICS_N_QUAD", "8"))
EVAL_EVERY = int(os.getenv("SEMANTICS_EVAL_EVERY", "1" if SMOKE else "5"))
SUPPORT_THRESHOLD = float(os.getenv("SEMANTICS_SUPPORT_THRESHOLD", "0.5"))


def relative_error(value, reference):
    return float(
        np.linalg.norm(np.asarray(value) - np.asarray(reference))
        / max(np.linalg.norm(reference), 1.0e-14)
    )


def make_initial_states(n: int):
    xs = np.linspace(-2.0, 2.0, n)
    centers = (-1.35, -0.45, 0.45, 1.35)
    states = []
    for center in centers:
        state = np.exp(-0.5 * ((xs - center) / 0.28) ** 2)
        state /= state.sum()
        states.append(state)
    return np.stack(states)


def assemble(base, directions, theta):
    return base + np.einsum("m,mij->ij", theta, directions)


def arnoldi_full(matrix, vector, dimension):
    """Full Arnoldi with two modified-Gram--Schmidt passes."""
    n = len(vector)
    requested = min(dimension, n)
    beta = np.linalg.norm(vector)
    if beta == 0.0:
        raise ValueError("Arnoldi start vector must be nonzero")

    basis = np.zeros((n, requested), dtype=np.result_type(matrix, vector))
    hessenberg = np.zeros(
        (requested, requested), dtype=np.result_type(matrix, vector)
    )
    basis[:, 0] = vector / beta
    actual = requested

    for column in range(requested):
        candidate = matrix @ basis[:, column]
        for _ in range(2):
            coefficients = basis[:, : column + 1].conj().T @ candidate
            hessenberg[: column + 1, column] += coefficients
            candidate -= basis[:, : column + 1] @ coefficients

        if column + 1 < requested:
            next_norm = np.linalg.norm(candidate)
            breakdown_scale = max(
                np.linalg.norm(matrix, ord=1), 1.0
            ) * np.finfo(np.float64).eps
            if next_norm <= breakdown_scale:
                actual = column + 1
                break
            hessenberg[column + 1, column] = next_norm
            basis[:, column + 1] = candidate / next_norm

    return basis[:, :actual], hessenberg[:actual, :actual], beta


def projected_path(basis, hessenberg, beta, times):
    """Evaluate Q exp(tH)e1 beta for several times with a safe fallback."""
    times = np.asarray(times, dtype=np.float64)
    first = np.zeros(len(hessenberg), dtype=hessenberg.dtype)
    first[0] = beta

    try:
        eigenvalues, eigenvectors = scipy.linalg.eig(hessenberg)
        condition = np.linalg.cond(eigenvectors)
        if not np.isfinite(condition) or condition > 1.0e8:
            raise np.linalg.LinAlgError("ill-conditioned projected eigenbasis")
        coefficients = scipy.linalg.solve(eigenvectors, first)
        reduced = eigenvectors @ (
            coefficients[:, None]
            * np.exp(eigenvalues[:, None] * times[None, :])
        )
        values = basis @ reduced
        if np.isrealobj(basis) and np.max(np.abs(values.imag)) < 1.0e-9:
            values = values.real
        if not np.all(np.isfinite(values)):
            raise np.linalg.LinAlgError("nonfinite eigendecomposition path")
        return values
    except (np.linalg.LinAlgError, ValueError):
        return np.column_stack(
            [
                basis @ (scipy.linalg.expm(time_value * hessenberg) @ first)
                for time_value in times
            ]
        )


def make_continuous_frechet_value_and_grad(
    base, directions, initial_states, targets, dt, krylov_dimension, n_quad
):
    nodes, weights = leggauss(n_quad)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    forward_times = np.concatenate((dt * nodes, np.array([dt])))
    adjoint_times = dt * (1.0 - nodes)
    num_contexts = len(initial_states)

    def value_and_grad(theta):
        matrix = assemble(base, directions, theta)
        loss = 0.0
        gradient = np.zeros(len(theta), dtype=np.float64)

        for initial, target in zip(initial_states, targets):
            q_forward, h_forward, beta_forward = arnoldi_full(
                matrix, initial, krylov_dimension
            )
            forward_values = projected_path(
                q_forward, h_forward, beta_forward, forward_times
            )
            prediction = forward_values[:, -1]
            residual = prediction - target
            loss += 0.5 * np.vdot(residual, residual).real

            q_adjoint, h_adjoint, beta_adjoint = arnoldi_full(
                matrix.T, residual, krylov_dimension
            )
            adjoint_values = projected_path(
                q_adjoint, h_adjoint, beta_adjoint, adjoint_times
            )
            path_states = forward_values[:, :-1]

            for index, direction in enumerate(directions):
                direction_path = direction @ path_states
                integrand = np.sum(adjoint_values * direction_path, axis=0)
                gradient[index] += dt * np.dot(weights, integrand).real

        return loss / num_contexts, gradient / num_contexts

    return value_and_grad


def make_matfree_value_and_grad(
    base, directions, initial_states, targets, dt, krylov_dimension
):
    base_main, base_lower, base_upper = diagonal_representation(base)
    direction_diagonals = [diagonal_representation(d) for d in directions]

    main0 = jnp.asarray(base_main)
    lower0 = jnp.asarray(base_lower)
    upper0 = jnp.asarray(base_upper)
    main_dirs = jnp.asarray(
        np.stack([diagonals[0] for diagonals in direction_diagonals])
    )
    lower_dirs = jnp.asarray(
        np.stack([diagonals[1] for diagonals in direction_diagonals])
    )
    upper_dirs = jnp.asarray(
        np.stack([diagonals[2] for diagonals in direction_diagonals])
    )
    states = tuple(jnp.asarray(state) for state in initial_states)
    target_values = tuple(jnp.asarray(target) for target in targets)
    dt_jax = jnp.asarray(dt)

    def matrix_vector_product(vector, main, lower, upper):
        upper_action = jnp.concatenate(
            (upper * vector[1:], jnp.zeros((1,), dtype=vector.dtype))
        )
        lower_action = jnp.concatenate(
            (jnp.zeros((1,), dtype=vector.dtype), lower * vector[:-1])
        )
        return dt_jax * (main * vector + upper_action + lower_action)

    hessenberg = decomp.hessenberg(
        krylov_dimension,
        reortho="full",
        custom_vjp=True,
        reortho_vjp="match",
    )
    exponential_action = funm.funm_arnoldi(
        funm.dense_funm_pade_exp(), hessenberg
    )

    def loss(theta):
        main = main0 + theta @ main_dirs
        lower = lower0 + theta @ lower_dirs
        upper = upper0 + theta @ upper_dirs
        total = jnp.asarray(0.0)
        for initial, target in zip(states, target_values):
            prediction = exponential_action(
                matrix_vector_product, initial, main, lower, upper
            )
            residual = prediction - target
            total = total + 0.5 * jnp.vdot(residual, residual).real
        return total / len(states)

    compiled = jax.jit(jax.value_and_grad(loss))
    # Compile before training; compilation time is excluded from all timings.
    jax.block_until_ready(compiled(jnp.zeros(len(directions))))

    def value_and_grad(theta):
        value, gradient = compiled(jnp.asarray(theta))
        jax.block_until_ready((value, gradient))
        return float(value), np.asarray(gradient)

    return value_and_grad


def exact_metrics(base, directions, theta, theta_true, initial_states, targets, dt):
    transition = scipy.linalg.expm(assemble(base, directions, theta) * dt)
    predictions = (transition @ initial_states.T).T
    residuals = predictions - targets
    exact_loss = 0.5 * np.mean(np.sum(residuals**2, axis=1))
    target_energy = np.mean(np.sum(targets**2, axis=1))
    prediction_relerr = np.sqrt(
        2.0 * exact_loss / max(target_energy, 1.0e-30)
    )
    coefficient_relerr = relative_error(theta, theta_true)

    true_support = set(np.flatnonzero(np.abs(theta_true) > 0.0))
    estimated_support = set(np.flatnonzero(np.abs(theta) >= SUPPORT_THRESHOLD))
    intersection = len(true_support & estimated_support)
    precision = intersection / max(len(estimated_support), 1)
    recall = intersection / len(true_support)
    support_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    top_k = set(np.argsort(np.abs(theta))[-len(true_support) :])

    return {
        "exact_loss": float(exact_loss),
        "exact_prediction_relerr": float(prediction_relerr),
        "coefficient_relerr": coefficient_relerr,
        "support_f1": support_f1,
        "topk_support_exact": int(top_k == true_support),
    }


def exact_parameter_vjp(
    base, directions, theta, initial_states, residuals, dt
):
    matrix = assemble(base, directions, theta)
    gradient = np.empty(len(directions), dtype=np.float64)
    for index, direction in enumerate(directions):
        frechet = scipy.linalg.expm_frechet(
            matrix * dt, direction * dt, compute_expm=False
        )
        actions = (frechet @ initial_states.T).T
        gradient[index] = np.mean(np.sum(residuals * actions, axis=1))
    return gradient


def forward_predictions(base, directions, theta, initial_states, dt, dimension):
    matrix = assemble(base, directions, theta)
    predictions = []
    for initial in initial_states:
        basis, hessenberg, beta = arnoldi_full(matrix, initial, dimension)
        predictions.append(
            projected_path(basis, hessenberg, beta, np.array([dt]))[:, 0]
        )
    return np.stack(predictions)


def validation_errors(
    base,
    directions,
    theta,
    initial_states,
    targets,
    dt,
    dimension,
    continuous_value_and_grad,
    matfree_value_and_grad,
):
    approximate_predictions = forward_predictions(
        base, directions, theta, initial_states, dt, dimension
    )
    exact_predictions = (
        scipy.linalg.expm(assemble(base, directions, theta) * dt)
        @ initial_states.T
    ).T
    continuous_loss, continuous_gradient = continuous_value_and_grad(theta)
    matfree_loss, matfree_gradient = matfree_value_and_grad(theta)

    continuous_residuals = approximate_predictions - targets
    exact_semigroup_gradient = exact_parameter_vjp(
        base,
        directions,
        theta,
        initial_states,
        continuous_residuals,
        dt,
    )

    # Central differences verify Matfree's exact VJP of its approximate loss.
    finite_difference = np.empty_like(theta)
    step = 1.0e-5
    for index in range(len(theta)):
        perturbation = np.zeros_like(theta)
        perturbation[index] = step
        plus, _ = matfree_value_and_grad(theta + perturbation)
        minus, _ = matfree_value_and_grad(theta - perturbation)
        finite_difference[index] = (plus - minus) / (2.0 * step)

    return {
        "forward_relerr": relative_error(
            approximate_predictions, exact_predictions
        ),
        "forward_loss_difference": abs(continuous_loss - matfree_loss),
        "continuous_exact_vjp_relerr": relative_error(
            continuous_gradient, exact_semigroup_gradient
        ),
        "matfree_fd_vjp_relerr": relative_error(
            matfree_gradient, finite_difference
        ),
        "matfree_exact_vjp_relerr": relative_error(
            matfree_gradient, exact_semigroup_gradient
        ),
    }


def train(
    method,
    value_and_grad,
    initial_theta,
    base,
    directions,
    theta_true,
    initial_states,
    targets,
    dt,
    dimension,
    seed,
    validation,
):
    theta = initial_theta.copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    rows = []
    elapsed = 0.0

    for epoch in range(EPOCHS + 1):
        should_evaluate = (
            epoch == 0 or epoch == EPOCHS or epoch % EVAL_EVERY == 0
        )
        if should_evaluate:
            evaluation_loss, evaluation_gradient = value_and_grad(theta)
            metrics = exact_metrics(
                base,
                directions,
                theta,
                theta_true,
                initial_states,
                targets,
                dt,
            )
            rows.append(
                {
                    "method": method,
                    "dt": dt,
                    "m": dimension,
                    "n_quad": N_QUAD if method == "continuous_frechet" else "",
                    "seed": seed,
                    "epoch": epoch,
                    "learning_rate": LEARNING_RATE,
                    "approximate_loss": evaluation_loss,
                    "gradient_norm": float(
                        np.linalg.norm(evaluation_gradient)
                    ),
                    "training_seconds": elapsed,
                    **metrics,
                    **validation,
                }
            )
        if epoch == EPOCHS:
            break

        start = time.perf_counter()
        approximate_loss, gradient = value_and_grad(theta)
        elapsed += time.perf_counter() - start
        gradient_norm = float(np.linalg.norm(gradient))

        step = epoch + 1
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        theta -= LEARNING_RATE * corrected_first / (
            np.sqrt(corrected_second) + 1.0e-8
        )
        theta = np.clip(theta, -10.0, 10.0)

    return rows, theta


def summarize(raw_rows):
    final_rows = [row for row in raw_rows if row["epoch"] == EPOCHS]
    groups = {}
    for row in final_rows:
        key = (row["method"], row["dt"], row["m"])
        groups.setdefault(key, []).append(row)

    summary = []
    metric_names = (
        "exact_loss",
        "exact_prediction_relerr",
        "coefficient_relerr",
        "support_f1",
        "topk_support_exact",
        "training_seconds",
    )
    for (method, dt, dimension), rows in groups.items():
        item = {
            "method": method,
            "dt": dt,
            "m": dimension,
            "n_quad": N_QUAD if method == "continuous_frechet" else "",
            "seeds": len(rows),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        summary.append(item)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    base, directions, theta_true, _, _ = build_fpe_library()
    initial_states = make_initial_states(len(base))
    raw_rows = []

    print(
        f"gradient-semantics training: epochs={EPOCHS}, lr={LEARNING_RATE}, "
        f"nq={N_QUAD}, contexts={len(initial_states)}"
    )
    for dt in DT_VALUES:
        exact_transition = scipy.linalg.expm(
            assemble(base, directions, theta_true) * dt
        )
        targets = (exact_transition @ initial_states.T).T

        for dimension in M_VALUES:
            continuous = make_continuous_frechet_value_and_grad(
                base,
                directions,
                initial_states,
                targets,
                dt,
                dimension,
                N_QUAD,
            )
            matfree = make_matfree_value_and_grad(
                base,
                directions,
                initial_states,
                targets,
                dt,
                dimension,
            )
            theta_validation = theta_true + 0.08 * np.linspace(
                -1.0, 1.0, len(theta_true)
            )
            validation = validation_errors(
                base,
                directions,
                theta_validation,
                initial_states,
                targets,
                dt,
                dimension,
                continuous,
                matfree,
            )
            print(
                f"dt={dt:g} m={dimension:2d} "
                f"forward={validation['forward_relerr']:.2e} "
                f"continuous/exact={validation['continuous_exact_vjp_relerr']:.2e} "
                f"matfree/FD={validation['matfree_fd_vjp_relerr']:.2e} "
                f"matfree/exact={validation['matfree_exact_vjp_relerr']:.2e}"
            )

            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                initial_theta = 0.25 * rng.standard_normal(len(theta_true))
                for method, value_and_grad in (
                    ("continuous_frechet", continuous),
                    ("discrete_arnoldi", matfree),
                ):
                    rows, final_theta = train(
                        method,
                        value_and_grad,
                        initial_theta,
                        base,
                        directions,
                        theta_true,
                        initial_states,
                        targets,
                        dt,
                        dimension,
                        seed,
                        validation,
                    )
                    raw_rows.extend(rows)
                    final = rows[-1]
                    print(
                        f"  {method:20s} seed={seed:3d} "
                        f"loss={final['exact_loss']:.3e} "
                        f"theta={final['coefficient_relerr']:.3e} "
                        f"top-k={final['topk_support_exact']} "
                        f"time={final['training_seconds']:.2f}s"
                    )

    summary_rows = summarize(raw_rows)
    write_csv(RAW_OUTPUT, raw_rows)
    write_csv(SUMMARY_OUTPUT, summary_rows)
    print(f"saved {RAW_OUTPUT}")
    print(f"saved {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
