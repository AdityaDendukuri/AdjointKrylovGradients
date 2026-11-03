"""
Audit how structure enters the Matfree/Kraemer comparison and benchmark a
regime tailored to sparse local generator libraries.

There are two separate experiments.

1. ``--mode audit`` checks that

   * Matfree's custom Arnoldi VJP agrees with ordinary differentiation through
     the same finite Arnoldi program;
   * assembling the affine library before Arnoldi agrees with evaluating the
     affine library inside every matrix-vector product;
   * the resulting parameter gradient agrees with central differences.

2. ``--mode scaling`` compares, in the same JAX process and with the same
   fully reorthogonalised Arnoldi implementation,

   * Matfree's exact VJP of the finite Arnoldi program; and
   * the structured two-sided projected Frechet core used by this repository.

   The library contains local conservative transition directions.  Each
   direction has two nonzeros, while all directions share a tridiagonal union
   pattern.  This is the regime in which projecting a low-rank sensitivity
   core once onto the library can avoid the m operator-value VJPs required by
   a generic Arnoldi adjoint.

3. ``--mode breakdown`` approaches a happy Arnoldi breakdown.  The forward
   error vanishes with the perturbation of the initial state, but the
   derivative of the finite Arnoldi program need not converge to the
   semigroup state sensitivity.  The experiment compares that program
   derivative with an independent adjoint exponential action at the same
   Krylov depth.

Examples
--------

    .venv-matfree/bin/python experiments/audit_kraemer_structure.py --mode audit

    STRUCTURE_N=2000 STRUCTURE_M_VALUES=100,1000,5000 \
        .venv-matfree/bin/python \
        experiments/audit_kraemer_structure.py --mode scaling
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from statistics import median

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.linalg
from matfree import decomp, funm

from kraemer_comparison import build_fpe_library, diagonal_representation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "paper" / "figures" / "kraemer_structure_scaling.csv"
)


def relative_error(value, reference) -> float:
    value_np = np.asarray(value, dtype=np.float64)
    reference_np = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(value_np - reference_np)
        / max(np.linalg.norm(reference_np), 1.0e-14)
    )


def median_ms(function, repeats: int):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return median(samples)


def tridiagonal_coordinates(n: int):
    rows = np.concatenate(
        (
            np.arange(n),
            np.arange(1, n),
            np.arange(n - 1),
        )
    )
    columns = np.concatenate(
        (
            np.arange(n),
            np.arange(n - 1),
            np.arange(1, n),
        )
    )
    return rows, columns


def dense_from_tridiagonal_values(values, n: int):
    matrix = np.zeros((n, n), dtype=np.asarray(values).dtype)
    main = np.asarray(values[:n])
    lower = np.asarray(values[n : 2 * n - 1])
    upper = np.asarray(values[2 * n - 1 :])
    indices = np.arange(n)
    matrix[indices, indices] = main
    edge_indices = np.arange(n - 1)
    matrix[edge_indices + 1, edge_indices] = lower
    matrix[edge_indices, edge_indices + 1] = upper
    return matrix


def dense_local_library_reference(
    *,
    n,
    num_directions,
    dt,
    base_values,
    support_positions,
    support_direction_ids,
    support_coefficients,
    state,
    upstream,
):
    matrix = dense_from_tridiagonal_values(base_values, n)
    exact_state = scipy.linalg.expm(matrix.T * dt) @ np.asarray(upstream)
    exact_theta = np.zeros(num_directions)
    union_rows, union_columns = tridiagonal_coordinates(n)
    for direction in range(num_directions):
        direction_matrix = np.zeros_like(matrix)
        mask = support_direction_ids == direction
        positions = support_positions[mask]
        direction_matrix[
            union_rows[positions], union_columns[positions]
        ] = support_coefficients[mask]
        frechet = scipy.linalg.expm_frechet(
            matrix * dt, direction_matrix * dt, compute_expm=False
        )
        exact_theta[direction] = np.vdot(
            upstream, frechet @ state
        ).real
    return exact_theta, exact_state


def tridiagonal_action(vector, values, n: int):
    main = values[:n]
    lower = values[n : 2 * n - 1]
    upper = values[2 * n - 1 :]
    upper_action = jnp.concatenate(
        (upper * vector[1:], jnp.zeros((1,), dtype=vector.dtype))
    )
    lower_action = jnp.concatenate(
        (jnp.zeros((1,), dtype=vector.dtype), lower * vector[:-1])
    )
    return main * vector + upper_action + lower_action


def tridiagonal_transpose_action(vector, values, n: int):
    main = values[:n]
    lower = values[n : 2 * n - 1]
    upper = values[2 * n - 1 :]
    transposed_upper_action = jnp.concatenate(
        (lower * vector[1:], jnp.zeros((1,), dtype=vector.dtype))
    )
    transposed_lower_action = jnp.concatenate(
        (jnp.zeros((1,), dtype=vector.dtype), upper * vector[:-1])
    )
    return main * vector + transposed_upper_action + transposed_lower_action


def make_local_generator_library(n: int, num_directions: int):
    """Return a stable base generator and two-entry conservative directions."""
    lower_rate = 35.0
    upper_rate = 65.0
    lower = np.full(n - 1, lower_rate)
    upper = np.full(n - 1, upper_rate)
    main = np.empty(n)
    main[0] = -lower_rate
    main[-1] = -upper_rate
    main[1:-1] = -(lower_rate + upper_rate)
    base_values = np.concatenate((main, lower, upper))

    positions = np.empty(2 * num_directions, dtype=np.int32)
    direction_ids = np.repeat(
        np.arange(num_directions, dtype=np.int32), 2
    )
    coefficients = np.empty(2 * num_directions)

    for direction in range(num_directions):
        edge = (direction // 2) % (n - 1)
        generation = direction // (2 * (n - 1))
        amplitude = 0.5 + 0.05 * ((7 * direction + generation) % 11)
        if direction % 2 == 0:
            # column edge -> row edge+1
            positions[2 * direction] = edge
            positions[2 * direction + 1] = n + edge
        else:
            # column edge+1 -> row edge
            positions[2 * direction] = edge + 1
            positions[2 * direction + 1] = 2 * n - 1 + edge
        coefficients[2 * direction] = -amplitude
        coefficients[2 * direction + 1] = amplitude

    return base_values, positions, direction_ids, coefficients


def assemble_values(
    theta,
    base_values,
    support_positions,
    support_direction_ids,
    support_coefficients,
):
    contributions = (
        support_coefficients * theta[support_direction_ids]
    )
    return base_values.at[support_positions].add(contributions)


def make_matfree_loss_and_grad(
    *,
    n,
    krylov_dimension,
    dt,
    base_values,
    support_positions,
    support_direction_ids,
    support_coefficients,
    upstream,
):
    hessenberg = decomp.hessenberg(
        krylov_dimension,
        reortho="full",
        custom_vjp=True,
        reortho_vjp="match",
    )
    exponential_action = funm.funm_arnoldi(
        funm.dense_funm_pade_exp(), hessenberg
    )

    def scaled_action(vector, values):
        return dt * tridiagonal_action(vector, values, n)

    def predict(theta, state):
        values = assemble_values(
            theta,
            base_values,
            support_positions,
            support_direction_ids,
            support_coefficients,
        )
        return exponential_action(scaled_action, state, values)

    def loss(theta, state):
        return jnp.vdot(upstream, predict(theta, state)).real

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))


def make_structured_core_loss_and_grad(
    *,
    n,
    num_directions,
    krylov_dimension,
    core_rank,
    dt,
    base_values,
    support_positions,
    support_direction_ids,
    support_coefficients,
    direction_union_positions,
    support_to_direction_union,
    upstream,
):
    hessenberg = decomp.hessenberg(
        krylov_dimension,
        reortho="full",
        custom_vjp=False,
        reortho_vjp="match",
    )
    dense_exp = funm.dense_funm_pade_exp()
    operator_rows_np, operator_columns_np = tridiagonal_coordinates(n)
    direction_rows = jnp.asarray(
        operator_rows_np[np.asarray(direction_union_positions)]
    )
    direction_columns = jnp.asarray(
        operator_columns_np[np.asarray(direction_union_positions)]
    )

    def forward_decomposition(values, state):
        return hessenberg(
            lambda vector, operator_values: tridiagonal_action(
                vector, operator_values, n
            ),
            state,
            values,
        )

    def adjoint_decomposition(values):
        return hessenberg(
            lambda vector, operator_values: tridiagonal_transpose_action(
                vector, operator_values, n
            ),
            upstream,
            values,
        )

    def loss_and_grad(theta, state):
        values = assemble_values(
            theta,
            base_values,
            support_positions,
            support_direction_ids,
            support_coefficients,
        )
        basis_f_rows, hessenberg_f, _, inverse_beta_f = (
            forward_decomposition(values, state)
        )
        basis_a_rows, hessenberg_a, _, inverse_beta_a = (
            adjoint_decomposition(values)
        )
        basis_f = basis_f_rows
        basis_a = basis_a_rows
        beta_f = 1.0 / inverse_beta_f
        beta_a = 1.0 / inverse_beta_a

        first = jnp.eye(krylov_dimension, dtype=state.dtype)[:, 0]
        prediction = beta_f * (
            basis_f @ (dense_exp(dt * hessenberg_f) @ first)
        )
        state_vjp = beta_a * (
            basis_a @ (dense_exp(dt * hessenberg_a) @ first)
        )

        block = jnp.zeros(
            (2 * krylov_dimension, 2 * krylov_dimension),
            dtype=state.dtype,
        )
        block = block.at[:krylov_dimension, :krylov_dimension].set(
            hessenberg_a
        )
        block = block.at[krylov_dimension:, krylov_dimension:].set(
            hessenberg_f.T
        )
        block = block.at[0, krylov_dimension].set(beta_a * beta_f)
        core = dense_exp(dt * block)[
            :krylov_dimension, krylov_dimension:
        ]
        left_vectors, singular_values, right_vectors_transposed = (
            jnp.linalg.svd(core, full_matrices=False)
        )
        selected_rank = min(core_rank, krylov_dimension)
        left_factors = (
            basis_a @ left_vectors[:, :selected_rank]
        )
        right_factors = (
            basis_f @ right_vectors_transposed[:selected_rank, :].T
        )
        kernel = jnp.sum(
            left_factors[direction_rows, :]
            * right_factors[direction_columns, :]
            * singular_values[None, :selected_rank],
            axis=1,
        )
        direction_gradient = jnp.zeros(
            (num_directions,), dtype=state.dtype
        ).at[support_direction_ids].add(
            support_coefficients * kernel[support_to_direction_union]
        )
        retained_energy = jnp.linalg.norm(
            singular_values[:selected_rank]
        )
        discarded_energy = jnp.linalg.norm(
            singular_values[selected_rank:]
        )
        relative_tail = discarded_energy / jnp.maximum(
            jnp.linalg.norm(singular_values),
            jnp.finfo(state.dtype).eps,
        )
        loss = jnp.vdot(upstream, prediction).real
        return (
            loss,
            direction_gradient,
            state_vjp,
            relative_tail,
            retained_energy,
        )

    return jax.jit(loss_and_grad)


def run_audit():
    n = 80
    num_directions = 10
    dt = 0.3
    krylov_dimension = 20
    base, directions, _, theta, state = build_fpe_library(
        n=n, num_directions=num_directions
    )
    base_diagonals = diagonal_representation(base)
    direction_diagonals = [
        diagonal_representation(direction) for direction in directions
    ]
    base_values_np = np.concatenate(base_diagonals)
    direction_values_np = np.stack(
        [np.concatenate(diagonals) for diagonals in direction_diagonals]
    )
    base_values = jnp.asarray(base_values_np)
    direction_values = jnp.asarray(direction_values_np)
    theta_jax = jnp.asarray(theta)
    state_jax = jnp.asarray(state)
    coordinates = np.linspace(-1.0, 1.0, n)
    upstream = jnp.asarray(
        np.sin(1.7 * coordinates) + 0.2 * np.cos(3.1 * coordinates)
    )

    def action(vector, values):
        return dt * tridiagonal_action(vector, values, n)

    def make_loss(custom_vjp, assemble_outside):
        hessenberg = decomp.hessenberg(
            krylov_dimension,
            reortho="full",
            custom_vjp=custom_vjp,
            reortho_vjp="match",
        )
        exponential_action = funm.funm_arnoldi(
            funm.dense_funm_pade_exp(), hessenberg
        )

        if assemble_outside:

            def predict(coefficients, initial):
                values = base_values + coefficients @ direction_values
                return exponential_action(action, initial, values)

        else:

            def parameter_action(vector, coefficients):
                values = base_values + coefficients @ direction_values
                return action(vector, values)

            def predict(coefficients, initial):
                return exponential_action(
                    parameter_action, initial, coefficients
                )

        def loss(coefficients, initial):
            return jnp.vdot(upstream, predict(coefficients, initial)).real

        return (
            jax.jit(predict),
            jax.jit(jax.value_and_grad(loss, argnums=(0, 1))),
        )

    predict_custom, custom = make_loss(True, True)
    predict_plain, plain = make_loss(False, True)
    predict_inside, inside = make_loss(True, False)

    outputs = {}
    for name, predict, value_and_grad in (
        ("custom_preassembled", predict_custom, custom),
        ("plain_preassembled", predict_plain, plain),
        ("custom_inside_matvec", predict_inside, inside),
    ):
        prediction = predict(theta_jax, state_jax)
        result = value_and_grad(theta_jax, state_jax)
        jax.block_until_ready((prediction, result))
        outputs[name] = (prediction, result)

    custom_prediction, custom_result = outputs["custom_preassembled"]
    custom_loss, (custom_theta, custom_state) = custom_result
    print("Kraemer reproduction audit")
    for name in ("plain_preassembled", "custom_inside_matvec"):
        prediction, result = outputs[name]
        loss, (theta_gradient, state_gradient) = result
        print(
            f"  {name:24s} "
            f"prediction={relative_error(prediction, custom_prediction):.3e} "
            f"loss={abs(float(loss - custom_loss)):.3e} "
            f"theta={relative_error(theta_gradient, custom_theta):.3e} "
            f"state={relative_error(state_gradient, custom_state):.3e}"
        )

    step = 1.0e-5
    finite_difference = np.empty(num_directions)
    for index in range(num_directions):
        perturbation = np.zeros(num_directions)
        perturbation[index] = step
        plus = custom(theta_jax + perturbation, state_jax)[0]
        minus = custom(theta_jax - perturbation, state_jax)[0]
        finite_difference[index] = float((plus - minus) / (2.0 * step))
    print(
        "  custom versus central difference "
        f"theta={relative_error(custom_theta, finite_difference):.3e}"
    )

    repeats = int(os.getenv("STRUCTURE_REPEATS", "20"))
    print("  median loss+VJP runtime")
    for name, _, value_and_grad in (
        ("custom_preassembled", predict_custom, custom),
        ("plain_preassembled", predict_plain, plain),
        ("custom_inside_matvec", predict_inside, inside),
    ):
        elapsed = median_ms(
            lambda function=value_and_grad: function(theta_jax, state_jax),
            repeats,
        )
        print(f"    {name:24s} {elapsed:9.4f} ms")


def run_scaling():
    n = int(os.getenv("STRUCTURE_N", "2000"))
    krylov_dimension = int(os.getenv("STRUCTURE_M_KRYLOV", "40"))
    core_rank = int(os.getenv("STRUCTURE_CORE_RANK", "8"))
    dt = float(os.getenv("STRUCTURE_DT", "0.3"))
    repeats = int(os.getenv("STRUCTURE_REPEATS", "20"))
    direction_counts = [
        int(value)
        for value in os.getenv(
            "STRUCTURE_M_VALUES", "100,1000,5000"
        ).split(",")
    ]
    output = Path(os.getenv("STRUCTURE_OUTPUT", str(DEFAULT_OUTPUT)))
    dense_reference = os.getenv("STRUCTURE_DENSE_REFERENCE", "0") == "1"
    if dense_reference and n > 200:
        raise ValueError(
            "STRUCTURE_DENSE_REFERENCE is intended only for n <= 200"
        )

    coordinates = np.linspace(-2.0, 2.0, n)
    state_np = np.exp(-0.5 * coordinates**2)
    state_np /= np.linalg.norm(state_np)
    upstream_np = (
        np.sin(1.7 * coordinates) + 0.2 * np.cos(3.1 * coordinates)
    )
    upstream_np /= np.linalg.norm(upstream_np)
    state = jnp.asarray(state_np)
    upstream = jnp.asarray(upstream_np)

    rows = []
    for num_directions in direction_counts:
        (
            base_values_np,
            support_positions_np,
            support_direction_ids_np,
            support_coefficients_np,
        ) = make_local_generator_library(n, num_directions)
        base_values = jnp.asarray(base_values_np)
        support_positions = jnp.asarray(support_positions_np)
        support_direction_ids = jnp.asarray(support_direction_ids_np)
        support_coefficients = jnp.asarray(support_coefficients_np)
        (
            direction_union_positions_np,
            support_to_direction_union_np,
        ) = np.unique(support_positions_np, return_inverse=True)
        direction_union_positions = jnp.asarray(
            direction_union_positions_np
        )
        support_to_direction_union = jnp.asarray(
            support_to_direction_union_np
        )
        theta = jnp.zeros((num_directions,), dtype=state.dtype)

        matfree_value_and_grad = make_matfree_loss_and_grad(
            n=n,
            krylov_dimension=krylov_dimension,
            dt=dt,
            base_values=base_values,
            support_positions=support_positions,
            support_direction_ids=support_direction_ids,
            support_coefficients=support_coefficients,
            upstream=upstream,
        )
        structured_value_and_grad = make_structured_core_loss_and_grad(
            n=n,
            num_directions=num_directions,
            krylov_dimension=krylov_dimension,
            core_rank=core_rank,
            dt=dt,
            base_values=base_values,
            support_positions=support_positions,
            support_direction_ids=support_direction_ids,
            support_coefficients=support_coefficients,
            direction_union_positions=direction_union_positions,
            support_to_direction_union=support_to_direction_union,
            upstream=upstream,
        )

        compile_start = time.perf_counter()
        matfree_result = matfree_value_and_grad(theta, state)
        jax.block_until_ready(matfree_result)
        matfree_compile_seconds = time.perf_counter() - compile_start

        compile_start = time.perf_counter()
        structured_result = structured_value_and_grad(theta, state)
        jax.block_until_ready(structured_result)
        structured_compile_seconds = time.perf_counter() - compile_start

        matfree_loss, (matfree_theta, matfree_state) = matfree_result
        (
            structured_loss,
            structured_theta,
            structured_state,
            relative_tail,
            retained_energy,
        ) = structured_result

        matfree_ms = median_ms(
            lambda: matfree_value_and_grad(theta, state), repeats
        )
        structured_ms = median_ms(
            lambda: structured_value_and_grad(theta, state), repeats
        )
        row = {
            "n": n,
            "M": num_directions,
            "direction_nnz": len(support_positions_np),
            "operator_nnz": 3 * n - 2,
            "direction_union_nnz": len(direction_union_positions_np),
            "m": krylov_dimension,
            "core_rank": min(core_rank, krylov_dimension),
            "dt": dt,
            "matfree_loss_and_vjp_ms": matfree_ms,
            "structured_loss_and_vjp_ms": structured_ms,
            "structured_speedup": matfree_ms / structured_ms,
            "loss_abs_difference": abs(
                float(matfree_loss - structured_loss)
            ),
            "parameter_vjp_rel_difference": relative_error(
                structured_theta, matfree_theta
            ),
            "state_vjp_rel_difference": relative_error(
                structured_state, matfree_state
            ),
            "matfree_parameter_exact_relerr": np.nan,
            "structured_parameter_exact_relerr": np.nan,
            "matfree_state_exact_relerr": np.nan,
            "structured_state_exact_relerr": np.nan,
            "core_relative_tail": float(relative_tail),
            "core_retained_energy": float(retained_energy),
            "matfree_compile_seconds": matfree_compile_seconds,
            "structured_compile_seconds": structured_compile_seconds,
            "repeats": repeats,
            "platform": f"{jax.default_backend()}-{jax.devices()[0].device_kind}",
        }
        if dense_reference:
            exact_theta, exact_state = dense_local_library_reference(
                n=n,
                num_directions=num_directions,
                dt=dt,
                base_values=base_values_np,
                support_positions=support_positions_np,
                support_direction_ids=support_direction_ids_np,
                support_coefficients=support_coefficients_np,
                state=state_np,
                upstream=upstream_np,
            )
            row["matfree_parameter_exact_relerr"] = relative_error(
                matfree_theta, exact_theta
            )
            row["structured_parameter_exact_relerr"] = relative_error(
                structured_theta, exact_theta
            )
            row["matfree_state_exact_relerr"] = relative_error(
                matfree_state, exact_state
            )
            row["structured_state_exact_relerr"] = relative_error(
                structured_state, exact_state
            )
        rows.append(row)
        print(
            f"n={n:5d} M={num_directions:5d} m={krylov_dimension:2d} "
            f"rank={row['core_rank']:2d} "
            f"Matfree={matfree_ms:8.3f} ms "
            f"structured={structured_ms:8.3f} ms "
            f"speedup={row['structured_speedup']:5.2f}x "
            f"theta-diff={row['parameter_vjp_rel_difference']:.2e} "
            f"state-diff={row['state_vjp_rel_difference']:.2e} "
            f"tail={row['core_relative_tail']:.2e}"
        )
        if dense_reference:
            print(
                "  exact errors: "
                f"Matfree theta={row['matfree_parameter_exact_relerr']:.2e} "
                f"structured theta="
                f"{row['structured_parameter_exact_relerr']:.2e} "
                f"Matfree state={row['matfree_state_exact_relerr']:.2e} "
                f"structured state={row['structured_state_exact_relerr']:.2e}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


def run_breakdown():
    n = int(os.getenv("BREAKDOWN_N", "20"))
    krylov_dimension = int(os.getenv("BREAKDOWN_M", "5"))
    dt = float(os.getenv("BREAKDOWN_DT", "4.0"))
    epsilon_values = [
        float(value)
        for value in os.getenv(
            "BREAKDOWN_EPS", "1e-1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12"
        ).split(",")
    ]
    diagonal = jnp.asarray(-np.linspace(0.0, 4.0, n))
    base_state = np.zeros(n)
    base_state[0] = 1.0
    perturbation = np.sin(np.linspace(0.3, 2.7, n))
    perturbation[0] = 0.0
    perturbation /= np.linalg.norm(perturbation)
    upstream_np = np.cos(np.linspace(-0.4, 2.2, n))
    upstream_support = int(
        os.getenv("BREAKDOWN_UPSTREAM_SUPPORT", str(krylov_dimension))
    )
    if upstream_support < n:
        upstream_np[upstream_support:] = 0.0
    upstream_np /= np.linalg.norm(upstream_np)
    upstream = jnp.asarray(upstream_np)

    def scaled_diagonal_action(vector, diagonal_values):
        return dt * diagonal_values * vector

    discrete_hessenberg = decomp.hessenberg(
        krylov_dimension,
        reortho="full",
        custom_vjp=True,
        reortho_vjp="match",
    )
    discrete_exponential = funm.funm_arnoldi(
        funm.dense_funm_pade_exp(), discrete_hessenberg
    )

    def discrete_loss(state):
        prediction = discrete_exponential(
            scaled_diagonal_action, state, diagonal
        )
        return jnp.vdot(upstream, prediction).real, prediction

    discrete_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda state: discrete_loss(state)[0], argnums=0
        )
    )
    discrete_predict = jax.jit(lambda state: discrete_loss(state)[1])

    adjoint_hessenberg = decomp.hessenberg(
        krylov_dimension,
        reortho="full",
        custom_vjp=False,
        reortho_vjp="match",
    )
    dense_exp = funm.dense_funm_pade_exp()

    @jax.jit
    def independent_state_vjp():
        basis_rows, small_matrix, _, inverse_beta = adjoint_hessenberg(
            lambda vector, diagonal_values: diagonal_values * vector,
            upstream,
            diagonal,
        )
        first = jnp.eye(krylov_dimension, dtype=upstream.dtype)[:, 0]
        return (
            (1.0 / inverse_beta)
            * basis_rows
            @ (dense_exp(dt * small_matrix) @ first)
        )

    independent = independent_state_vjp()
    exact_state = np.exp(np.asarray(diagonal) * dt) * upstream_np
    independent_error = relative_error(independent, exact_state)

    print(
        "Near-breakdown state-VJP audit "
        f"(n={n}, m={krylov_dimension}, dt={dt:g})"
    )
    print(
        f"  independent adjoint exact relative error: "
        f"{independent_error:.3e}"
    )
    print(
        "  epsilon      forward-error   discrete-state-error   "
        "discrete-gradient-norm"
    )
    rows = []
    for epsilon in epsilon_values:
        state_np = base_state + epsilon * perturbation
        state = jnp.asarray(state_np)
        prediction = discrete_predict(state)
        _, discrete_state = discrete_value_and_grad(state)
        jax.block_until_ready((prediction, discrete_state))
        exact_prediction = (
            np.exp(np.asarray(diagonal) * dt) * state_np
        )
        forward_error = relative_error(prediction, exact_prediction)
        discrete_error = relative_error(discrete_state, exact_state)
        discrete_norm = float(np.linalg.norm(np.asarray(discrete_state)))
        rows.append(
            {
                "n": n,
                "m": krylov_dimension,
                "dt": dt,
                "upstream_support": upstream_support,
                "epsilon": epsilon,
                "forward_relerr": forward_error,
                "discrete_state_vjp_relerr": discrete_error,
                "independent_state_vjp_relerr": independent_error,
                "discrete_state_vjp_norm": discrete_norm,
            }
        )
        print(
            f"  {epsilon:9.1e}   "
            f"{forward_error:13.3e}   "
            f"{discrete_error:20.3e}   "
            f"{discrete_norm:22.3e}"
        )
    output = Path(
        os.getenv(
            "BREAKDOWN_OUTPUT",
            str(ROOT / "paper" / "figures" / "kraemer_breakdown.csv"),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("audit", "scaling", "breakdown"),
        default="audit",
    )
    arguments = parser.parse_args()
    if arguments.mode == "audit":
        run_audit()
    elif arguments.mode == "scaling":
        run_scaling()
    else:
        run_breakdown()


if __name__ == "__main__":
    main()
