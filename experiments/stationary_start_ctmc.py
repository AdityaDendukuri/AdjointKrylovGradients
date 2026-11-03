"""
Stationary-start stress test for continuous- and discrete-Arnoldi gradients.

The baseline continuous-time Markov chain (CTMC) has generator ``A0`` and
stationary distribution ``pi``:

    A0 @ pi = 0.

The learned generator is an affine library of conservative directed-edge
perturbations,

    A(theta) = A0 + sum_j theta[j] * directions[j].

At theta=0 and p0=pi, the forward Arnoldi computation has an exact
one-dimensional happy breakdown, even though the physical parameter
sensitivity is generally nonzero:

    d/dtheta_j exp(t A(theta)) pi |_{theta=0}
      = integral exp((t-s) A0) directions[j] pi ds.

This script compares:

* a dense SciPy ``expm_frechet`` reference for the exact-semigroup gradient;
* this repository's two-sided continuous Fréchet/Krylov approximation;
* Matfree 0.6.1's custom VJP of the fixed-depth Arnoldi program; and
* the finite gradient obtained by a simple one-dimensional
  happy-breakdown termination branch.

The near-stationary sweep is essential: it tests whether any discrepancy
persists away from the released Matfree implementation's exact-breakdown
NaNs.  The optional training mode evaluates every method using the same
dense exact-semigroup loss.

Examples
--------

    .venv-matfree/bin/python \
        experiments/stationary_start_ctmc.py --mode gradient

    STATIONARY_EPOCHS=300 .venv-matfree/bin/python \
        experiments/stationary_start_ctmc.py --mode all
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import platform
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg
from matfree import decomp, funm
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[1]
GRADIENT_OUTPUT = (
    ROOT / "paper" / "figures" / "stationary_start_ctmc_gradients.csv"
)
TRAINING_RAW_OUTPUT = (
    ROOT / "paper" / "figures" / "stationary_start_ctmc_training_raw.csv"
)
TRAINING_SUMMARY_OUTPUT = (
    ROOT / "paper" / "figures" / "stationary_start_ctmc_training_summary.csv"
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


def cosine_similarity(value, reference):
    value = np.asarray(value)
    reference = np.asarray(reference)
    denominator = np.linalg.norm(value) * np.linalg.norm(reference)
    if denominator <= 1.0e-30:
        return np.nan
    return float(np.vdot(value, reference).real / denominator)


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty result table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assemble(base, directions, theta):
    return base + np.einsum("m,mij->ij", theta, directions)


def make_problem(n: int, library_size: int | None = None):
    """Construct a symmetric birth-death CTMC and a directed-edge library.

    Matrices use the paper's column-generator convention: every column sums
    to zero and ``p_dot = A p``.
    """
    if n < 10:
        raise ValueError("Use n >= 10 so the four true edges are distinct")

    baseline_rate = 4.0
    base = np.zeros((n, n), dtype=np.float64)
    all_directions = []
    all_direction_labels = []

    for left in range(n - 1):
        right = left + 1

        # Baseline left -> right transition.
        base[right, left] += baseline_rate
        base[left, left] -= baseline_rate
        direction = np.zeros_like(base)
        direction[right, left] = 1.0
        direction[left, left] = -1.0
        all_directions.append(direction)
        all_direction_labels.append(f"{left}->{right}")

        # Baseline right -> left transition.
        base[left, right] += baseline_rate
        base[right, right] -= baseline_rate
        direction = np.zeros_like(base)
        direction[left, right] = 1.0
        direction[right, right] = -1.0
        all_directions.append(direction)
        all_direction_labels.append(f"{right}->{left}")

    all_directions = np.stack(all_directions)
    if library_size is None:
        library_size = len(all_directions)
    if not 8 <= library_size <= len(all_directions):
        raise ValueError(
            f"library_size must lie in [8, {len(all_directions)}]"
        )
    selected_direction_ids = np.linspace(
        0, len(all_directions) - 1, library_size, dtype=np.int32
    )
    if len(np.unique(selected_direction_ids)) != library_size:
        raise AssertionError("Library subsampling produced duplicate edges")
    directions = all_directions[selected_direction_ids]
    direction_labels = [
        all_direction_labels[index] for index in selected_direction_ids
    ]
    theta_true = np.zeros(len(directions), dtype=np.float64)
    true_indices = np.asarray(
        [
            library_size // 7,
            library_size // 3,
            3 * library_size // 5,
            6 * library_size // 7,
        ],
        dtype=np.int32,
    )
    true_indices = np.clip(true_indices, 0, len(directions) - 1)
    if len(np.unique(true_indices)) != len(true_indices):
        raise ValueError("True-support construction produced duplicate edges")
    theta_true[true_indices] = np.asarray((1.25, -1.0, 1.5, -1.2))

    stationary = np.full(n, 1.0 / n)
    coordinates = np.linspace(-1.0, 1.0, n)
    generic = (
        0.65 * np.exp(-0.5 * ((coordinates + 0.45) / 0.16) ** 2)
        + 0.35 * np.exp(-0.5 * ((coordinates - 0.55) / 0.11) ** 2)
    )
    generic /= generic.sum()

    # Defensive validity checks for the CTMC construction.
    if np.linalg.norm(base.sum(axis=0)) > 1.0e-13:
        raise AssertionError("Baseline matrix is not a column generator")
    if np.linalg.norm(base @ stationary) > 1.0e-13:
        raise AssertionError("Constructed distribution is not stationary")
    true_generator = assemble(base, directions, theta_true)
    off_diagonal = true_generator.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if off_diagonal.min() < -1.0e-13:
        raise AssertionError("True perturbation creates a negative rate")

    return {
        "base": base,
        "directions": directions,
        "theta_true": theta_true,
        "true_indices": true_indices,
        "direction_labels": direction_labels,
        "selected_direction_ids": selected_direction_ids,
        "stationary": stationary,
        "generic": generic,
    }


def initial_state(problem, epsilon):
    return (
        (1.0 - epsilon) * problem["stationary"]
        + epsilon * problem["generic"]
    )


def arnoldi_full(matrix, vector, dimension):
    """Full Arnoldi with two MGS passes and explicit happy breakdown."""
    n = len(vector)
    requested = min(dimension, n)
    beta = np.linalg.norm(vector)
    if beta == 0.0:
        raise ValueError("Arnoldi start vector must be nonzero")

    dtype = np.result_type(matrix, vector)
    basis = np.zeros((n, requested), dtype=dtype)
    hessenberg = np.zeros((requested, requested), dtype=dtype)
    basis[:, 0] = vector / beta
    actual = requested
    breakdown_scale = max(np.linalg.norm(matrix, ord=1), 1.0)
    breakdown_scale *= 10.0 * np.finfo(np.float64).eps

    for column in range(requested):
        candidate = matrix @ basis[:, column]
        for _ in range(2):
            coefficients = basis[:, : column + 1].conj().T @ candidate
            hessenberg[: column + 1, column] += coefficients
            candidate -= basis[:, : column + 1] @ coefficients

        if column + 1 < requested:
            next_norm = np.linalg.norm(candidate)
            if next_norm <= breakdown_scale:
                actual = column + 1
                break
            hessenberg[column + 1, column] = next_norm
            basis[:, column + 1] = candidate / next_norm

    return basis[:, :actual], hessenberg[:actual, :actual], beta


def projected_path(basis, hessenberg, beta, times):
    first = np.zeros(len(hessenberg), dtype=hessenberg.dtype)
    first[0] = beta
    return np.column_stack(
        [
            basis @ (scipy.linalg.expm(time_value * hessenberg) @ first)
            for time_value in np.asarray(times)
        ]
    )


def exact_predictions(problem, theta, state, times):
    matrix = assemble(problem["base"], problem["directions"], theta)
    return np.stack(
        [scipy.linalg.expm(time_value * matrix) @ state for time_value in times]
    )


def make_exact_value_and_grad(problem, state, targets, times):
    base = problem["base"]
    directions = problem["directions"]

    def value_and_grad(theta):
        matrix = assemble(base, directions, theta)
        loss = 0.0
        gradient = np.zeros(len(directions))
        for time_value, target in zip(times, targets):
            transition = scipy.linalg.expm(time_value * matrix)
            prediction = transition @ state
            residual = prediction - target
            loss += 0.5 * np.vdot(residual, residual).real
            for index, direction in enumerate(directions):
                frechet = scipy.linalg.expm_frechet(
                    time_value * matrix,
                    time_value * direction,
                    compute_expm=False,
                )
                gradient[index] += np.vdot(
                    residual, frechet @ state
                ).real
        scale = 1.0 / len(times)
        return loss * scale, gradient * scale

    return value_and_grad


def make_continuous_value_and_grad(
    problem,
    state,
    targets,
    times,
    forward_dimension,
    adjoint_dimension,
    quadrature_order,
):
    """Two-sided approximation of the exact-semigroup Fréchet VJP."""
    base = problem["base"]
    directions = problem["directions"]
    nodes, weights = leggauss(quadrature_order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    def value_and_grad(theta):
        matrix = assemble(base, directions, theta)
        loss = 0.0
        gradient = np.zeros(len(directions))

        for time_value, target in zip(times, targets):
            forward_times = np.concatenate(
                (time_value * nodes, np.asarray([time_value]))
            )
            forward_basis, forward_small, forward_beta = arnoldi_full(
                matrix, state, forward_dimension
            )
            forward_values = projected_path(
                forward_basis,
                forward_small,
                forward_beta,
                forward_times,
            )
            prediction = forward_values[:, -1]
            residual = prediction - target
            loss += 0.5 * np.vdot(residual, residual).real

            adjoint_basis, adjoint_small, adjoint_beta = arnoldi_full(
                matrix.T, residual, adjoint_dimension
            )
            adjoint_values = projected_path(
                adjoint_basis,
                adjoint_small,
                adjoint_beta,
                time_value * (1.0 - nodes),
            )
            path_states = forward_values[:, :-1]
            direction_paths = np.einsum(
                "mij,jq->miq", directions, path_states
            )
            integrands = np.einsum(
                "iq,miq->mq", adjoint_values, direction_paths
            )
            gradient += time_value * (integrands @ weights).real

        scale = 1.0 / len(times)
        return loss * scale, gradient * scale

    return value_and_grad


def make_matfree_value_and_grad(
    problem,
    state,
    targets,
    times,
    krylov_dimension,
    *,
    custom_vjp=True,
):
    """Fixed-depth, fully reorthogonalized Matfree Arnoldi loss and VJP."""
    base = jnp.asarray(problem["base"])
    directions = jnp.asarray(problem["directions"])
    state_jax = jnp.asarray(state)
    targets_jax = tuple(jnp.asarray(target) for target in targets)

    hessenberg = decomp.hessenberg(
        krylov_dimension,
        reortho="full",
        custom_vjp=custom_vjp,
        reortho_vjp="match",
    )
    exponential_action = funm.funm_arnoldi(
        funm.dense_funm_pade_exp(), hessenberg
    )

    def parameter_action(vector, theta, time_value):
        matrix = base + jnp.tensordot(theta, directions, axes=1)
        return time_value * (matrix @ vector)

    def loss(theta):
        total = jnp.asarray(0.0, dtype=theta.dtype)
        for time_value, target in zip(times, targets_jax):
            prediction = exponential_action(
                lambda vector, coefficients: parameter_action(
                    vector, coefficients, time_value
                ),
                state_jax,
                theta,
            )
            residual = prediction - target
            total += 0.5 * jnp.vdot(residual, residual).real
        return total / len(times)

    compiled = jax.jit(jax.value_and_grad(loss))

    def value_and_grad(theta):
        value, gradient = compiled(jnp.asarray(theta))
        jax.block_until_ready((value, gradient))
        return float(value), np.asarray(gradient)

    return value_and_grad


def terminated_one_dimensional_gradient(problem, targets, times):
    """Gradient of the simplest finite happy-breakdown branch.

    The start vector is the stationary distribution and is held fixed.
    The branch returns ``q exp(t q* A(theta) q) ||pi||`` without constructing
    a second basis vector.  This is not attributed to Matfree; it records
    what a straightforward early-termination patch would differentiate.
    """
    base = jnp.asarray(problem["base"])
    directions = jnp.asarray(problem["directions"])
    state = jnp.asarray(problem["stationary"])
    targets_jax = tuple(jnp.asarray(target) for target in targets)
    beta = jnp.linalg.norm(state)
    basis_vector = state / beta

    def loss(theta):
        matrix = base + jnp.tensordot(theta, directions, axes=1)
        projected = jnp.vdot(basis_vector, matrix @ basis_vector).real
        total = jnp.asarray(0.0, dtype=theta.dtype)
        for time_value, target in zip(times, targets_jax):
            prediction = basis_vector * jnp.exp(
                time_value * projected
            ) * beta
            residual = prediction - target
            total += 0.5 * jnp.vdot(residual, residual).real
        return total / len(times)

    value, gradient = jax.jit(jax.value_and_grad(loss))(
        jnp.zeros(len(directions))
    )
    jax.block_until_ready((value, gradient))
    return float(value), np.asarray(gradient)


def gradient_row(
    problem,
    epsilon,
    times,
    krylov_dimension,
    adjoint_dimension,
    quadrature_order,
):
    state = initial_state(problem, epsilon)
    targets = exact_predictions(
        problem, problem["theta_true"], state, times
    )
    theta0 = np.zeros(len(problem["directions"]))

    exact = make_exact_value_and_grad(problem, state, targets, times)
    continuous = make_continuous_value_and_grad(
        problem,
        state,
        targets,
        times,
        krylov_dimension,
        adjoint_dimension,
        quadrature_order,
    )
    matfree = make_matfree_value_and_grad(
        problem, state, targets, times, krylov_dimension
    )
    matfree_native = make_matfree_value_and_grad(
        problem,
        state,
        targets,
        times,
        krylov_dimension,
        custom_vjp=False,
    )

    exact_loss, exact_gradient = exact(theta0)
    continuous_loss, continuous_gradient = continuous(theta0)
    matfree_loss, matfree_gradient = matfree(theta0)
    matfree_native_loss, matfree_native_gradient = matfree_native(theta0)
    baseline_prediction = exact_predictions(problem, theta0, state, times)
    forward_norm = np.linalg.norm(baseline_prediction - targets)

    arnoldi_predictions = []
    for time_value in times:
        basis, small_matrix, beta = arnoldi_full(
            problem["base"], state, krylov_dimension
        )
        arnoldi_predictions.append(
            projected_path(
                basis,
                small_matrix,
                beta,
                np.asarray([time_value]),
            )[:, 0]
        )
    arnoldi_predictions = np.stack(arnoldi_predictions)
    finite_forward_relerr = relative_error(
        arnoldi_predictions, baseline_prediction
    )

    matfree_finite = bool(
        np.isfinite(matfree_loss) and np.all(np.isfinite(matfree_gradient))
    )
    matfree_native_finite = bool(
        np.isfinite(matfree_native_loss)
        and np.all(np.isfinite(matfree_native_gradient))
    )
    true_support = set(problem["true_indices"].tolist())
    top_k = len(true_support)

    def topk_overlap(gradient):
        if not np.all(np.isfinite(gradient)):
            return np.nan
        selected = set(
            np.argsort(np.abs(np.asarray(gradient)))[-top_k:].tolist()
        )
        return len(selected & true_support)

    row = {
        "n": len(state),
        "M": len(problem["directions"]),
        "epsilon": epsilon,
        "m_forward": krylov_dimension,
        "m_adjoint": adjoint_dimension,
        "n_quad": quadrature_order,
        "num_times": len(times),
        "max_time": max(times),
        "stationarity_residual": float(
            np.linalg.norm(problem["base"] @ state)
        ),
        "target_difference_norm": float(forward_norm),
        "finite_arnoldi_forward_relerr": finite_forward_relerr,
        "exact_loss": exact_loss,
        "continuous_loss": continuous_loss,
        "matfree_loss": matfree_loss,
        "matfree_native_loss": matfree_native_loss,
        "exact_gradient_norm": float(np.linalg.norm(exact_gradient)),
        "continuous_gradient_norm": float(
            np.linalg.norm(continuous_gradient)
        ),
        "matfree_gradient_norm": float(np.linalg.norm(matfree_gradient)),
        "matfree_native_gradient_norm": float(
            np.linalg.norm(matfree_native_gradient)
        ),
        "continuous_gradient_relerr": relative_error(
            continuous_gradient, exact_gradient
        ),
        "matfree_gradient_relerr": (
            relative_error(matfree_gradient, exact_gradient)
            if matfree_finite
            else np.nan
        ),
        "continuous_gradient_cosine": cosine_similarity(
            continuous_gradient, exact_gradient
        ),
        "matfree_gradient_cosine": (
            cosine_similarity(matfree_gradient, exact_gradient)
            if matfree_finite
            else np.nan
        ),
        "matfree_native_gradient_relerr": (
            relative_error(matfree_native_gradient, exact_gradient)
            if matfree_native_finite
            else np.nan
        ),
        "matfree_custom_vs_native_relerr": (
            relative_error(matfree_gradient, matfree_native_gradient)
            if matfree_finite and matfree_native_finite
            else np.nan
        ),
        "continuous_topk_true_overlap": topk_overlap(continuous_gradient),
        "matfree_topk_true_overlap": topk_overlap(matfree_gradient),
        "matfree_finite": int(matfree_finite),
        "matfree_native_finite": int(matfree_native_finite),
        "jax_version": jax.__version__,
        "matfree_version": importlib.metadata.version("matfree"),
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
    }

    if epsilon == 0.0:
        branch_loss, branch_gradient = terminated_one_dimensional_gradient(
            problem, targets, times
        )
        row.update(
            {
                "terminated_branch_loss": branch_loss,
                "terminated_branch_gradient_norm": float(
                    np.linalg.norm(branch_gradient)
                ),
                "terminated_branch_gradient_relerr": relative_error(
                    branch_gradient, exact_gradient
                ),
                "terminated_branch_gradient_cosine": cosine_similarity(
                    branch_gradient, exact_gradient
                ),
            }
        )
    else:
        row.update(
            {
                "terminated_branch_loss": np.nan,
                "terminated_branch_gradient_norm": np.nan,
                "terminated_branch_gradient_relerr": np.nan,
                "terminated_branch_gradient_cosine": np.nan,
            }
        )
    return row


def run_gradient_sweep():
    n = int(os.getenv("STATIONARY_N", "24"))
    library_size = int(os.getenv("STATIONARY_LIBRARY_SIZE", "16"))
    default_dimensions = os.getenv(
        "STATIONARY_M", "3,4,5,6,8,10,12"
    )
    krylov_dimensions = parse_values(
        "STATIONARY_M_VALUES", default_dimensions, int
    )
    fixed_adjoint_dimension = os.getenv("STATIONARY_M_ADJOINT")
    quadrature_order = int(os.getenv("STATIONARY_N_QUAD", "12"))
    times = parse_values("STATIONARY_TIMES", "0.25,0.5,1.0", float)
    epsilons = parse_values(
        "STATIONARY_EPS",
        "0,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e-1",
        float,
    )
    problem = make_problem(n, library_size)

    print(
        "stationary-start CTMC gradient sweep: "
        f"n={n}, M={len(problem['directions'])}, "
        f"m_f={krylov_dimensions}, "
        f"m_a={fixed_adjoint_dimension or 'matched'}, "
        f"nq={quadrature_order}, times={times}"
    )
    print(f"  true support: {problem['true_indices'].tolist()}")
    print(
        "  epsilon      ||A0 p||    continuous error    Matfree error   "
        "Matfree cosine   finite"
    )
    rows = []
    for krylov_dimension in krylov_dimensions:
        adjoint_dimension = (
            int(fixed_adjoint_dimension)
            if fixed_adjoint_dimension is not None
            else krylov_dimension
        )
        print(
            f"  depth m_f={krylov_dimension}, m_a={adjoint_dimension}"
        )
        for epsilon in epsilons:
            row = gradient_row(
                problem,
                epsilon,
                times,
                krylov_dimension,
                adjoint_dimension,
                quadrature_order,
            )
            rows.append(row)
            print(
                f"  {epsilon:9.1e}  "
                f"{row['stationarity_residual']:10.3e}  "
                f"{row['continuous_gradient_relerr']:16.3e}  "
                f"{row['matfree_gradient_relerr']:13.3e}  "
                f"{row['matfree_gradient_cosine']:15.6f}   "
                f"{row['matfree_finite']}"
            )
            if epsilon == 0.0:
                print(
                    "    one-dimensional termination branch: "
                    f"norm={row['terminated_branch_gradient_norm']:.3e}, "
                    f"relative error="
                    f"{row['terminated_branch_gradient_relerr']:.3e}"
                )

    validation_epsilon = next(
        (epsilon for epsilon in epsilons if epsilon > 0.0),
        0.0,
    )
    validation_state = initial_state(problem, validation_epsilon)
    validation_targets = exact_predictions(
        problem,
        problem["theta_true"],
        validation_state,
        times,
    )
    exact_reference = make_exact_value_and_grad(
        problem, validation_state, validation_targets, times
    )
    theta0 = np.zeros(len(problem["directions"]))
    _, exact_gradient = exact_reference(theta0)

    def dense_loss_only(theta):
        predictions = exact_predictions(
            problem, theta, validation_state, times
        )
        residuals = predictions - validation_targets
        return 0.5 * np.mean(np.sum(residuals**2, axis=1))

    finite_difference = np.empty_like(theta0)
    finite_difference_step = 1.0e-6
    for index in range(len(theta0)):
        perturbation = np.zeros_like(theta0)
        perturbation[index] = finite_difference_step
        finite_difference[index] = (
            dense_loss_only(theta0 + perturbation)
            - dense_loss_only(theta0 - perturbation)
        ) / (2.0 * finite_difference_step)
    dense_reference_fd_relerr = relative_error(
        exact_gradient, finite_difference
    )
    for row in rows:
        row["dense_reference_fd_relerr"] = dense_reference_fd_relerr
        row["dense_reference_fd_step"] = finite_difference_step
    print(
        "  dense expm_frechet / central-FD relative discrepancy: "
        f"{dense_reference_fd_relerr:.3e}"
    )

    write_csv(GRADIENT_OUTPUT, rows)
    print(f"saved {GRADIENT_OUTPUT}")
    return rows


def exact_metrics(problem, theta, state, targets, times):
    predictions = exact_predictions(problem, theta, state, times)
    residuals = predictions - targets
    loss = 0.5 * np.mean(np.sum(residuals**2, axis=1))
    target_energy = np.mean(np.sum(targets**2, axis=1))
    coefficient_error = relative_error(theta, problem["theta_true"])
    true_support = set(problem["true_indices"].tolist())
    top_k = set(
        np.argsort(np.abs(np.asarray(theta)))[-len(true_support) :].tolist()
    )
    return {
        "exact_loss": float(loss),
        "exact_prediction_relerr": float(
            np.sqrt(2.0 * loss / max(target_energy, 1.0e-30))
        ),
        "coefficient_relerr": coefficient_error,
        "topk_support_overlap": len(top_k & true_support),
        "topk_support_exact": int(top_k == true_support),
    }


def train_method(
    method,
    value_and_grad,
    problem,
    state,
    targets,
    times,
    epsilon,
    epochs,
    learning_rate,
    l1_penalty,
    evaluate_every,
    krylov_dimension,
    adjoint_dimension,
    quadrature_order,
):
    theta = np.zeros(len(problem["directions"]))
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    rows = []
    elapsed = 0.0
    failed = False

    for epoch in range(epochs + 1):
        if epoch == 0 or epoch == epochs or epoch % evaluate_every == 0:
            approximate_loss, gradient = value_and_grad(theta)
            finite = bool(
                np.isfinite(approximate_loss)
                and np.all(np.isfinite(gradient))
            )
            rows.append(
                {
                    "method": method,
                    "epsilon": epsilon,
                    "m_forward": krylov_dimension,
                    "m_adjoint": adjoint_dimension,
                    "n_quad": (
                        quadrature_order
                        if method == "continuous_frechet"
                        else ""
                    ),
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "l1_penalty": l1_penalty,
                    "approximate_loss": approximate_loss,
                    "gradient_norm": float(np.linalg.norm(gradient)),
                    "gradient_finite": int(finite),
                    "training_seconds": elapsed,
                    "jax_version": jax.__version__,
                    "matfree_version": importlib.metadata.version("matfree"),
                    "scipy_version": scipy.__version__,
                    "platform": platform.platform(),
                    **exact_metrics(
                        problem, theta, state, targets, times
                    ),
                }
            )
            if not finite:
                failed = True

        if epoch == epochs or failed:
            break

        start = time.perf_counter()
        _, gradient = value_and_grad(theta)
        elapsed += time.perf_counter() - start
        if not np.all(np.isfinite(gradient)):
            failed = True
            continue

        step = epoch + 1
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        theta -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1.0e-8
        )
        if l1_penalty > 0.0:
            threshold = learning_rate * l1_penalty
            theta = np.sign(theta) * np.maximum(
                np.abs(theta) - threshold, 0.0
            )
        theta = np.clip(theta, -3.5, 3.5)

    return rows


def summarize_training(rows, epochs):
    summary = []
    keys = sorted(
        {
            (row["method"], row["epsilon"], row["m_forward"])
            for row in rows
        }
    )
    for method, epsilon, krylov_dimension in keys:
        candidates = [
            row
            for row in rows
            if row["method"] == method
            and row["epsilon"] == epsilon
            and row["m_forward"] == krylov_dimension
        ]
        final = max(candidates, key=lambda row: row["epoch"])
        summary.append(
            {
                "method": method,
                "epsilon": epsilon,
                "m_forward": final["m_forward"],
                "m_adjoint": final["m_adjoint"],
                "n_quad": final["n_quad"],
                "requested_epochs": epochs,
                "completed_epochs": final["epoch"],
                "gradient_finite": final["gradient_finite"],
                "exact_loss": final["exact_loss"],
                "exact_prediction_relerr": final[
                    "exact_prediction_relerr"
                ],
                "coefficient_relerr": final["coefficient_relerr"],
                "topk_support_overlap": final["topk_support_overlap"],
                "topk_support_exact": final["topk_support_exact"],
                "training_seconds": final["training_seconds"],
            }
        )
    return summary


def run_training():
    n = int(os.getenv("STATIONARY_N", "24"))
    library_size = int(os.getenv("STATIONARY_LIBRARY_SIZE", "16"))
    default_dimensions = os.getenv("STATIONARY_M", "3,4,8")
    krylov_dimensions = parse_values(
        "STATIONARY_TRAIN_M_VALUES", default_dimensions, int
    )
    fixed_adjoint_dimension = os.getenv("STATIONARY_M_ADJOINT")
    quadrature_order = int(os.getenv("STATIONARY_N_QUAD", "12"))
    times = parse_values("STATIONARY_TIMES", "0.25,0.5,1.0", float)
    epsilons = parse_values(
        "STATIONARY_TRAIN_EPS", "0,1e-10", float
    )
    epochs = int(os.getenv("STATIONARY_EPOCHS", "300"))
    learning_rate = float(os.getenv("STATIONARY_LR", "0.03"))
    l1_penalty = float(os.getenv("STATIONARY_L1", "2e-4"))
    evaluate_every = int(os.getenv("STATIONARY_EVAL_EVERY", "10"))
    problem = make_problem(n, library_size)
    raw_rows = []

    print(
        "stationary-start CTMC training: "
        f"m={krylov_dimensions}, epochs={epochs}, "
        f"lr={learning_rate:g}, l1={l1_penalty:g}"
    )
    for krylov_dimension in krylov_dimensions:
        adjoint_dimension = (
            int(fixed_adjoint_dimension)
            if fixed_adjoint_dimension is not None
            else krylov_dimension
        )
        for epsilon in epsilons:
            state = initial_state(problem, epsilon)
            targets = exact_predictions(
                problem, problem["theta_true"], state, times
            )
            methods = (
                (
                    "continuous_frechet",
                    make_continuous_value_and_grad(
                        problem,
                        state,
                        targets,
                        times,
                        krylov_dimension,
                        adjoint_dimension,
                        quadrature_order,
                    ),
                ),
                (
                    "discrete_arnoldi",
                    make_matfree_value_and_grad(
                        problem,
                        state,
                        targets,
                        times,
                        krylov_dimension,
                    ),
                ),
                (
                    "dense_exact",
                    make_exact_value_and_grad(
                        problem, state, targets, times
                    ),
                ),
            )
            for method, value_and_grad in methods:
                rows = train_method(
                    method,
                    value_and_grad,
                    problem,
                    state,
                    targets,
                    times,
                    epsilon,
                    epochs,
                    learning_rate,
                    l1_penalty,
                    evaluate_every,
                    krylov_dimension,
                    adjoint_dimension,
                    quadrature_order,
                )
                raw_rows.extend(rows)
                final = rows[-1]
                print(
                    f"  m={krylov_dimension:2d} "
                    f"eps={epsilon:9.1e} {method:20s} "
                    f"epoch={final['epoch']:4d} "
                    f"finite={final['gradient_finite']} "
                    f"loss={final['exact_loss']:.3e} "
                    f"theta={final['coefficient_relerr']:.3e} "
                    f"top-k={final['topk_support_overlap']}/"
                    f"{len(problem['true_indices'])}"
                )

    summary = summarize_training(raw_rows, epochs)
    write_csv(TRAINING_RAW_OUTPUT, raw_rows)
    write_csv(TRAINING_SUMMARY_OUTPUT, summary)
    print(f"saved {TRAINING_RAW_OUTPUT}")
    print(f"saved {TRAINING_SUMMARY_OUTPUT}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("gradient", "train", "all"),
        default="gradient",
    )
    arguments = parser.parse_args()
    if arguments.mode in ("gradient", "all"):
        run_gradient_sweep()
    if arguments.mode in ("train", "all"):
        run_training()


if __name__ == "__main__":
    main()
