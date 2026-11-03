"""
Matched Matfree/Kraemer et al. benchmark for the structured exponential layer.

This script evaluates Matfree's exact custom VJP of the finite-m Arnoldi
approximation on the same one-dimensional Fokker--Planck generator, objective,
parameter directions, time steps, precisions, and Krylov dimensions as
`kraemer_comparison.jl`.

The dense Float64 reference differentiates the exact matrix exponential with
SciPy's Frechet derivative. JIT compilation is performed before timing.

Run after installing requirements-benchmark.txt:

    .venv-matfree/bin/python experiments/kraemer_comparison.py

Optional environment variables:

    KRAEMER_M_VALUES=5,10,20,30,40,60
    KRAEMER_DT_VALUES=0.3,1.0
    KRAEMER_REPEATS=10
"""

from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import jax

REQUESTED_PRECISION = os.getenv("KRAEMER_PRECISION")
jax.config.update("jax_enable_x64", REQUESTED_PRECISION != "float32")

import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg
from matfree import decomp, funm


ROOT = Path(__file__).resolve().parents[1]
FINAL_OUTPUT = ROOT / "paper" / "figures" / "kraemer_matfree.csv"
OUTPUT = (
    FINAL_OUTPUT.with_name(f"kraemer_matfree_{REQUESTED_PRECISION}.csv")
    if REQUESTED_PRECISION
    else FINAL_OUTPUT
)
COMBINED_OUTPUT = ROOT / "paper" / "figures" / "kraemer_comparison.csv"


def parse_values(name: str, default: str, cast):
    return tuple(cast(value) for value in os.getenv(name, default).split(","))


M_VALUES = parse_values("KRAEMER_M_VALUES", "5,10,20,30,40,60", int)
DT_VALUES = parse_values("KRAEMER_DT_VALUES", "0.3,1.0", float)
REPEATS = int(os.getenv("KRAEMER_REPEATS", "10"))


def relative_error(value, reference) -> float:
    value64 = np.asarray(value, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(value64 - reference64)
        / max(np.linalg.norm(reference64), 1.0e-14)
    )


def build_fpe_library(n: int = 80, num_directions: int = 10):
    """Match src/fpe.jl exactly, including the reflecting boundary terms."""
    length = 2.0
    diffusion = 0.5
    xs = np.linspace(-length, length, n)
    h = (xs[-1] - xs[0]) / (n - 1)

    base = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        if col > 0:
            base[col - 1, col] += diffusion / h**2
        if col < n - 1:
            base[col + 1, col] += diffusion / h**2
        base[col, col] += (
            -diffusion / h**2
            if col in (0, n - 1)
            else -2.0 * diffusion / h**2
        )

    directions = []
    for power in range(1, num_directions + 1):
        drift = (xs / length) ** power
        direction = np.zeros((n, n), dtype=np.float64)
        for col in range(n):
            if col < n - 1:
                direction[col + 1, col] += drift[col] / (2.0 * h)
            if col > 0:
                direction[col - 1, col] -= drift[col] / (2.0 * h)
        direction[0, 0] -= drift[0] / (2.0 * h)
        direction[-1, -1] += drift[-1] / (2.0 * h)
        directions.append(direction)

    theta_true = np.zeros(num_directions, dtype=np.float64)
    theta_true[0] = -4.0
    theta_true[2] = 4.0
    theta_eval = theta_true + 0.08 * np.linspace(-1.0, 1.0, num_directions)
    initial = np.exp(-0.5 * xs**2)
    initial /= initial.sum()
    return base, np.stack(directions), theta_true, theta_eval, initial


def exact_reference(base, directions, theta_true, theta_eval, initial, dt):
    matrix_true = base + np.einsum("m,mij->ij", theta_true, directions)
    matrix_eval = base + np.einsum("m,mij->ij", theta_eval, directions)
    target = scipy.linalg.expm(matrix_true * dt) @ initial
    prediction = scipy.linalg.expm(matrix_eval * dt) @ initial
    upstream = prediction - target

    parameter_vjp = np.empty(len(directions), dtype=np.float64)
    for index, direction in enumerate(directions):
        frechet = scipy.linalg.expm_frechet(
            matrix_eval * dt, direction * dt, compute_expm=False
        )
        parameter_vjp[index] = upstream @ (frechet @ initial)
    state_vjp = scipy.linalg.expm(matrix_eval.T * dt) @ upstream
    return target, prediction, parameter_vjp, state_vjp


def diagonal_representation(matrix):
    return (
        np.diag(matrix).copy(),
        np.diag(matrix, k=-1).copy(),
        np.diag(matrix, k=1).copy(),
    )


def benchmark_configuration(
    *,
    base,
    directions,
    theta_eval,
    initial,
    target,
    prediction_reference,
    parameter_reference,
    state_reference,
    dt,
    krylov_dimension,
    dtype,
):
    dtype_name = "float64" if dtype == jnp.float64 else "float32"
    base_main, base_lower, base_upper = diagonal_representation(base)
    direction_diagonals = [diagonal_representation(d) for d in directions]
    directions_main = np.stack([item[0] for item in direction_diagonals])
    directions_lower = np.stack([item[1] for item in direction_diagonals])
    directions_upper = np.stack([item[2] for item in direction_diagonals])

    main0 = jnp.asarray(base_main, dtype=dtype)
    lower0 = jnp.asarray(base_lower, dtype=dtype)
    upper0 = jnp.asarray(base_upper, dtype=dtype)
    main_dirs = jnp.asarray(directions_main, dtype=dtype)
    lower_dirs = jnp.asarray(directions_lower, dtype=dtype)
    upper_dirs = jnp.asarray(directions_upper, dtype=dtype)
    theta = jnp.asarray(theta_eval, dtype=dtype)
    p0 = jnp.asarray(initial, dtype=dtype)
    target_jax = jnp.asarray(target, dtype=dtype)
    dt_jax = jnp.asarray(dt, dtype=dtype)

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

    def predict(coefficients, state):
        # Assemble the three diagonals once outside the Arnoldi loop. The
        # custom VJP differentiates with respect to these matrix-free operator
        # parameters, after which ordinary JAX AD projects onto coefficients.
        main = main0 + coefficients @ main_dirs
        lower = lower0 + coefficients @ lower_dirs
        upper = upper0 + coefficients @ upper_dirs
        return exponential_action(
            matrix_vector_product, state, main, lower, upper
        )

    def loss(coefficients, state):
        residual = predict(coefficients, state) - target_jax
        return 0.5 * jnp.vdot(residual, residual).real

    predict_compiled = jax.jit(predict)
    loss_vjp_compiled = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

    compile_start = time.perf_counter()
    prediction = predict_compiled(theta, p0)
    jax.block_until_ready(prediction)
    value_and_gradients = loss_vjp_compiled(theta, p0)
    jax.block_until_ready(value_and_gradients)
    compile_seconds = time.perf_counter() - compile_start

    forward_times = []
    vjp_times = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        prediction = predict_compiled(theta, p0)
        jax.block_until_ready(prediction)
        forward_times.append((time.perf_counter_ns() - start) / 1.0e6)

        start = time.perf_counter_ns()
        value_and_gradients = loss_vjp_compiled(theta, p0)
        jax.block_until_ready(value_and_gradients)
        vjp_times.append((time.perf_counter_ns() - start) / 1.0e6)

    _, (parameter_vjp, state_vjp) = value_and_gradients
    return {
        "backend": "Matfree_JAX",
        "system": "FPE",
        "n": len(initial),
        "M": len(theta_eval),
        "dt": dt,
        "m": krylov_dimension,
        "n_quad": "",
        "precision": dtype_name,
        "reorthogonalization": "full_twice",
        "gradient_semantics": "exact_VJP_of_finite_m_Arnoldi",
        "forward_relerr": relative_error(prediction, prediction_reference),
        "parameter_vjp_relerr": relative_error(
            parameter_vjp, parameter_reference
        ),
        "state_vjp_relerr": relative_error(state_vjp, state_reference),
        "forward_ms": median(forward_times),
        "loss_and_full_vjp_ms": median(vjp_times),
        "compile_seconds": compile_seconds,
        "repeats": REPEATS,
        "jax_version": jax.__version__,
        "matfree_version": importlib.metadata.version("matfree"),
        "scipy_version": scipy.__version__,
        "platform": f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}",
    }


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def combine_if_available():
    julia_output = FINAL_OUTPUT.with_name("kraemer_shared_krylov.csv")
    if not julia_output.exists():
        return
    with julia_output.open(newline="") as handle:
        julia_rows = list(csv.DictReader(handle))
    with FINAL_OUTPUT.open(newline="") as handle:
        matfree_rows = list(csv.DictReader(handle))
    fieldnames = list(
        dict.fromkeys([*julia_rows[0].keys(), *matfree_rows[0].keys()])
    )
    with COMBINED_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(julia_rows)
        writer.writerows(matfree_rows)
    print(f"Saved {COMBINED_OUTPUT}")


def run_both_precisions():
    rows = []
    for precision in ("float64", "float32"):
        environment = os.environ.copy()
        environment["KRAEMER_PRECISION"] = precision
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            check=True,
            env=environment,
        )
        child_output = FINAL_OUTPUT.with_name(
            f"kraemer_matfree_{precision}.csv"
        )
        with child_output.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
        child_output.unlink()
    write_rows(FINAL_OUTPUT, rows)
    print(f"Saved {FINAL_OUTPUT}")
    combine_if_available()


def main():
    # Matfree 0.6.1 creates some internal arrays in JAX's default dtype. Run
    # Float64 and Float32 in separate processes so each custom VJP is
    # consistently typed.
    if REQUESTED_PRECISION is None:
        run_both_precisions()
        return

    base, directions, theta_true, theta_eval, initial = build_fpe_library()
    rows = []
    print(
        f"JAX {jax.__version__}; Matfree "
        f"{importlib.metadata.version('matfree')}; devices={jax.devices()}"
    )
    for dt in DT_VALUES:
        references = exact_reference(
            base, directions, theta_true, theta_eval, initial, dt
        )
        target, prediction_ref, parameter_ref, state_ref = references
        dtype = jnp.float64 if REQUESTED_PRECISION == "float64" else jnp.float32
        for krylov_dimension in M_VALUES:
            row = benchmark_configuration(
                base=base,
                directions=directions,
                theta_eval=theta_eval,
                initial=initial,
                target=target,
                prediction_reference=prediction_ref,
                parameter_reference=parameter_ref,
                state_reference=state_ref,
                dt=dt,
                krylov_dimension=krylov_dimension,
                dtype=dtype,
            )
            rows.append(row)
            print(
                f"Matfree {row['precision']:7s} dt={dt:3.1f} "
                f"m={krylov_dimension:2d} "
                f"forward={row['forward_relerr']:.2e} "
                f"theta={row['parameter_vjp_relerr']:.2e} "
                f"state={row['state_vjp_relerr']:.2e} "
                f"VJP={row['loss_and_full_vjp_ms']:.3f} ms"
            )
    write_rows(OUTPUT, rows)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
