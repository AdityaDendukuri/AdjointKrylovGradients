"""Portable same-JAX iso-accuracy and dimension-scaling benchmark.

The script compares three paths on the JAX backend selected by the process:

* Matfree's VJP of the finite Arnoldi program;
* the two-sided projected semigroup core used in this repository; and
* native dense JAX AD where the requested dimension is small enough.

Both matrix-free methods use Matfree's fully reorthogonalized Hessenberg
implementation.  A SciPy sparse exponential-action trajectory and composite
Simpson integration supply a backend-independent semigroup-VJP reference.
Compilation and warmup are excluded from every timing.

The problem is a sparse birth--death generator with local conservative
two-entry directions.  Its operator has ``3n-2`` nonzeros, and increasing
``n`` does not change the transition rates or stiffness.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import time
from pathlib import Path
from statistics import median

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import scipy
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg

from audit_kraemer_structure import (
    assemble_values,
    make_local_generator_library,
    make_matfree_loss_and_grad,
    make_structured_core_loss_and_grad,
    relative_error,
    tridiagonal_coordinates,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_values(name, default, cast):
    return tuple(
        cast(value) for value in os.getenv(name, default).split(",")
    )


N_VALUES = parse_values("PORTABLE_N_VALUES", "80,160,320", int)
DIRECTION_VALUES = parse_values(
    "PORTABLE_DIRECTION_VALUES", "10,100", int
)
KRYLOV_VALUES = parse_values(
    "PORTABLE_KRYLOV_VALUES", "20,30,40,50,60,80", int
)
CORE_RANK_VALUES = parse_values("PORTABLE_CORE_RANKS", "4,8", int)
QUADRATURE_VALUES = parse_values(
    "PORTABLE_QUADRATURE_VALUES", "8,16,24", int
)
DT = float(os.getenv("PORTABLE_DT", "0.3"))
REPEATS = int(os.getenv("PORTABLE_REPEATS", "31"))
WARMUPS = int(os.getenv("PORTABLE_WARMUPS", "3"))
REFERENCE_STEPS = int(os.getenv("PORTABLE_REFERENCE_STEPS", "513"))
REFERENCE_CHECK_STEPS = int(
    os.getenv("PORTABLE_REFERENCE_CHECK_STEPS", "257")
)
DENSE_MAX_N = int(os.getenv("PORTABLE_DENSE_MAX_N", "320"))
RUN_DENSE = os.getenv("PORTABLE_RUN_DENSE", "1") == "1"
RUN_LABEL = re.sub(
    r"[^A-Za-z0-9_.-]+", "_", os.getenv("PORTABLE_RUN_LABEL", "local")
)
OUTPUT_DIR = Path(
    os.getenv(
        "PORTABLE_OUTPUT_DIR",
        ROOT / "benchmark_results" / "portable_iso",
    )
)


def timing_summary(function):
    for _ in range(WARMUPS):
        result = function()
        jax.block_until_ready(result)
    samples = []
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        result = function()
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    values = np.asarray(samples)
    return result, {
        "runtime_median_ms": float(np.median(values)),
        "runtime_q25_ms": float(np.quantile(values, 0.25)),
        "runtime_q75_ms": float(np.quantile(values, 0.75)),
        "runtime_min_ms": float(np.min(values)),
    }


def sparse_matrix_from_values(values, n):
    main = values[:n]
    lower = values[n : 2 * n - 1]
    upper = values[2 * n - 1 :]
    return scipy.sparse.diags(
        (lower, main, upper), offsets=(-1, 0, 1), format="csc"
    )


def semigroup_reference(
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
    steps,
):
    intervals = steps - 1
    if steps < 3 or intervals & (intervals - 1):
        raise ValueError(
            "reference trajectory requires steps=2^k+1 for Romberg integration"
        )
    matrix = sparse_matrix_from_values(base_values, n)
    times = np.linspace(0.0, dt, steps)
    forward = scipy.sparse.linalg.expm_multiply(
        matrix,
        state,
        start=0.0,
        stop=dt,
        num=steps,
        endpoint=True,
    )
    adjoint = scipy.sparse.linalg.expm_multiply(
        matrix.T,
        upstream,
        start=0.0,
        stop=dt,
        num=steps,
        endpoint=True,
    )
    operator_rows, operator_columns = tridiagonal_coordinates(n)
    union_positions, support_to_union = np.unique(
        support_positions, return_inverse=True
    )
    rows = operator_rows[union_positions]
    columns = operator_columns[union_positions]
    integrand = (
        adjoint[::-1, rows] * forward[:, columns]
    )
    union_integrals = scipy.integrate.romb(
        integrand, dx=dt / intervals, axis=0
    )
    parameter_vjp = np.zeros(num_directions)
    np.add.at(
        parameter_vjp,
        support_direction_ids,
        support_coefficients * union_integrals[support_to_union],
    )
    return {
        "loss": float(np.vdot(upstream, forward[-1]).real),
        "parameter_vjp": parameter_vjp,
        "state_vjp": adjoint[-1],
    }


def make_dense_loss_and_grad(
    *,
    n,
    dt,
    base_values,
    support_positions,
    support_direction_ids,
    support_coefficients,
    upstream,
):
    rows_np, columns_np = tridiagonal_coordinates(n)
    rows = jnp.asarray(rows_np)
    columns = jnp.asarray(columns_np)

    def loss(theta, state):
        values = assemble_values(
            theta,
            base_values,
            support_positions,
            support_direction_ids,
            support_coefficients,
        )
        matrix = jnp.zeros((n, n), dtype=state.dtype)
        matrix = matrix.at[rows, columns].set(values)
        prediction = jsp_linalg.expm(dt * matrix) @ state
        return jnp.vdot(upstream, prediction).real

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))


def make_structured_quadrature_loss_and_grad(
    *,
    n,
    num_directions,
    krylov_dimension,
    n_quad,
    dt,
    base_values,
    support_positions,
    support_direction_ids,
    support_coefficients,
    direction_union_positions,
    support_to_direction_union,
    upstream,
):
    from matfree import decomp, funm

    from audit_kraemer_structure import (
        tridiagonal_action,
        tridiagonal_transpose_action,
    )

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
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = jnp.asarray((raw_nodes + 1.0) / 2.0)
    weights = jnp.asarray(raw_weights / 2.0)

    def loss_and_grad(theta, state):
        values = assemble_values(
            theta,
            base_values,
            support_positions,
            support_direction_ids,
            support_coefficients,
        )
        basis_f_rows, hessenberg_f, _, inverse_beta_f = hessenberg(
            lambda vector, operator_values: tridiagonal_action(
                vector, operator_values, n
            ),
            state,
            values,
        )
        basis_a_rows, hessenberg_a, _, inverse_beta_a = hessenberg(
            lambda vector, operator_values: tridiagonal_transpose_action(
                vector, operator_values, n
            ),
            upstream,
            values,
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

        forward_reduced = jax.vmap(
            lambda node: dense_exp(dt * node * hessenberg_f) @ first
        )(nodes)
        adjoint_reduced = jax.vmap(
            lambda node: dense_exp(dt * (1.0 - node) * hessenberg_a)
            @ first
        )(nodes)
        forward_path = beta_f * (basis_f @ forward_reduced.T)
        adjoint_path = beta_a * (basis_a @ adjoint_reduced.T)
        union_kernel = dt * jnp.sum(
            weights[None, :]
            * adjoint_path[direction_rows, :]
            * forward_path[direction_columns, :],
            axis=1,
        )
        direction_gradient = jnp.zeros(
            (num_directions,), dtype=state.dtype
        ).at[support_direction_ids].add(
            support_coefficients * union_kernel[support_to_direction_union]
        )
        loss = jnp.vdot(upstream, prediction).real
        return loss, direction_gradient, state_vjp

    return jax.jit(loss_and_grad)


def device_metadata():
    devices = []
    for device in jax.devices():
        devices.append(
            {
                "id": int(device.id),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
        )
    return {
        "run_label": RUN_LABEL,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "operating_system": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "scipy_version": scipy.__version__,
        "default_backend": jax.default_backend(),
        "devices": devices,
        "environment": {
            key: os.getenv(key, "")
            for key in (
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
    }


def base_row(
    *,
    n,
    num_directions,
    method,
    krylov_dimension,
    core_rank,
    n_quad,
    result,
    reference,
    reference_discrepancy,
    compile_seconds,
    timings,
):
    loss, parameter_vjp, state_vjp = result
    parameter_error = relative_error(
        parameter_vjp, reference["parameter_vjp"]
    )
    state_error = relative_error(state_vjp, reference["state_vjp"])
    return {
        "run_label": RUN_LABEL,
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "method": method,
        "system": "local_birth_death_generator",
        "precision": "float64",
        "n": n,
        "M": num_directions,
        "operator_nnz": 3 * n - 2,
        "direction_nnz": 2 * num_directions,
        "dt": DT,
        "m": krylov_dimension,
        "core_rank": core_rank,
        "n_quad": n_quad,
        "loss_abs_error": abs(float(loss) - reference["loss"]),
        "parameter_vjp_relerr": parameter_error,
        "state_vjp_relerr": state_error,
        "max_vjp_relerr": max(parameter_error, state_error),
        "reference_self_relerr": reference_discrepancy,
        **timings,
        "compile_seconds": compile_seconds,
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "reference_steps": REFERENCE_STEPS,
    }


def compile_and_time(function, theta, state):
    compile_start = time.perf_counter()
    compiled_result = function(theta, state)
    jax.block_until_ready(compiled_result)
    compile_seconds = time.perf_counter() - compile_start
    result, timings = timing_summary(lambda: function(theta, state))
    return result, compile_seconds, timings


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = device_metadata()
    metadata_path = OUTPUT_DIR / f"device_{RUN_LABEL}.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    rows = []
    for n in N_VALUES:
        coordinates = np.linspace(-2.0, 2.0, n)
        state_np = np.exp(-0.5 * coordinates**2)
        state_np /= np.linalg.norm(state_np)
        upstream_np = (
            np.sin(1.7 * coordinates)
            + 0.2 * np.cos(3.1 * coordinates)
        )
        upstream_np /= np.linalg.norm(upstream_np)
        state = jnp.asarray(state_np)
        upstream = jnp.asarray(upstream_np)

        for num_directions in DIRECTION_VALUES:
            (
                base_values_np,
                support_positions_np,
                support_direction_ids_np,
                support_coefficients_np,
            ) = make_local_generator_library(n, num_directions)
            reference = semigroup_reference(
                n=n,
                num_directions=num_directions,
                dt=DT,
                base_values=base_values_np,
                support_positions=support_positions_np,
                support_direction_ids=support_direction_ids_np,
                support_coefficients=support_coefficients_np,
                state=state_np,
                upstream=upstream_np,
                steps=REFERENCE_STEPS,
            )
            reference_check = semigroup_reference(
                n=n,
                num_directions=num_directions,
                dt=DT,
                base_values=base_values_np,
                support_positions=support_positions_np,
                support_direction_ids=support_direction_ids_np,
                support_coefficients=support_coefficients_np,
                state=state_np,
                upstream=upstream_np,
                steps=REFERENCE_CHECK_STEPS,
            )
            reference_discrepancy = max(
                relative_error(
                    reference_check["parameter_vjp"],
                    reference["parameter_vjp"],
                ),
                relative_error(
                    reference_check["state_vjp"],
                    reference["state_vjp"],
                ),
            )

            base_values = jnp.asarray(base_values_np)
            support_positions = jnp.asarray(support_positions_np)
            support_direction_ids = jnp.asarray(
                support_direction_ids_np
            )
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

            if RUN_DENSE and n <= DENSE_MAX_N:
                dense_function = make_dense_loss_and_grad(
                    n=n,
                    dt=DT,
                    base_values=base_values,
                    support_positions=support_positions,
                    support_direction_ids=support_direction_ids,
                    support_coefficients=support_coefficients,
                    upstream=upstream,
                )
                result, compile_seconds, timings = compile_and_time(
                    dense_function, theta, state
                )
                rows.append(
                    base_row(
                        n=n,
                        num_directions=num_directions,
                        method="JAX_dense_AD",
                        krylov_dimension="",
                        core_rank="",
                        n_quad="",
                        result=(result[0], result[1][0], result[1][1]),
                        reference=reference,
                        reference_discrepancy=reference_discrepancy,
                        compile_seconds=compile_seconds,
                        timings=timings,
                    )
                )
                print(
                    f"n={n:5d} M={num_directions:5d} dense "
                    f"error={rows[-1]['max_vjp_relerr']:.2e} "
                    f"time={rows[-1]['runtime_median_ms']:.3f} ms"
                )

            for krylov_dimension in KRYLOV_VALUES:
                if krylov_dimension > n:
                    continue
                matfree_function = make_matfree_loss_and_grad(
                    n=n,
                    krylov_dimension=krylov_dimension,
                    dt=DT,
                    base_values=base_values,
                    support_positions=support_positions,
                    support_direction_ids=support_direction_ids,
                    support_coefficients=support_coefficients,
                    upstream=upstream,
                )
                result, compile_seconds, timings = compile_and_time(
                    matfree_function, theta, state
                )
                rows.append(
                    base_row(
                        n=n,
                        num_directions=num_directions,
                        method="Matfree_finite_Arnoldi_VJP",
                        krylov_dimension=krylov_dimension,
                        core_rank="",
                        n_quad="",
                        result=(result[0], result[1][0], result[1][1]),
                        reference=reference,
                        reference_discrepancy=reference_discrepancy,
                        compile_seconds=compile_seconds,
                        timings=timings,
                    )
                )
                print(
                    f"n={n:5d} M={num_directions:5d} "
                    f"Matfree m={krylov_dimension:3d} "
                    f"error={rows[-1]['max_vjp_relerr']:.2e} "
                    f"time={rows[-1]['runtime_median_ms']:.3f} ms"
                )

                for requested_rank in CORE_RANK_VALUES:
                    core_rank = min(requested_rank, krylov_dimension)
                    structured_function = make_structured_core_loss_and_grad(
                        n=n,
                        num_directions=num_directions,
                        krylov_dimension=krylov_dimension,
                        core_rank=core_rank,
                        dt=DT,
                        base_values=base_values,
                        support_positions=support_positions,
                        support_direction_ids=support_direction_ids,
                        support_coefficients=support_coefficients,
                        direction_union_positions=direction_union_positions,
                        support_to_direction_union=support_to_direction_union,
                        upstream=upstream,
                    )
                    result, compile_seconds, timings = compile_and_time(
                        structured_function, theta, state
                    )
                    rows.append(
                        base_row(
                            n=n,
                            num_directions=num_directions,
                            method="Structured_semigroup_core",
                            krylov_dimension=krylov_dimension,
                            core_rank=core_rank,
                            n_quad="",
                            result=(result[0], result[1], result[2]),
                            reference=reference,
                            reference_discrepancy=reference_discrepancy,
                            compile_seconds=compile_seconds,
                            timings=timings,
                        )
                    )
                    print(
                        f"n={n:5d} M={num_directions:5d} "
                        f"core m={krylov_dimension:3d} r={core_rank:2d} "
                        f"error={rows[-1]['max_vjp_relerr']:.2e} "
                        f"time={rows[-1]['runtime_median_ms']:.3f} ms"
                    )

                for n_quad in QUADRATURE_VALUES:
                    quadrature_function = (
                        make_structured_quadrature_loss_and_grad(
                            n=n,
                            num_directions=num_directions,
                            krylov_dimension=krylov_dimension,
                            n_quad=n_quad,
                            dt=DT,
                            base_values=base_values,
                            support_positions=support_positions,
                            support_direction_ids=support_direction_ids,
                            support_coefficients=support_coefficients,
                            direction_union_positions=(
                                direction_union_positions
                            ),
                            support_to_direction_union=(
                                support_to_direction_union
                            ),
                            upstream=upstream,
                        )
                    )
                    result, compile_seconds, timings = compile_and_time(
                        quadrature_function, theta, state
                    )
                    rows.append(
                        base_row(
                            n=n,
                            num_directions=num_directions,
                            method="Structured_semigroup_quadrature",
                            krylov_dimension=krylov_dimension,
                            core_rank="",
                            n_quad=n_quad,
                            result=result,
                            reference=reference,
                            reference_discrepancy=reference_discrepancy,
                            compile_seconds=compile_seconds,
                            timings=timings,
                        )
                    )
                    print(
                        f"n={n:5d} M={num_directions:5d} "
                        f"quadrature m={krylov_dimension:3d} "
                        f"nq={n_quad:2d} "
                        f"error={rows[-1]['max_vjp_relerr']:.2e} "
                        f"time={rows[-1]['runtime_median_ms']:.3f} ms"
                    )

    output_path = OUTPUT_DIR / f"portable_iso_{RUN_LABEL}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
