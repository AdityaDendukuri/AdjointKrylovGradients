"""
Audit linear-invariant preservation in physical and finite-program VJPs.

For a common left invariant ``c`` of an affine generator library,

    c.T @ A(theta) = 0  and  c.T @ A_j = 0,

the exact semigroup preserves ``c.T @ p`` and its parameter derivative is
identically zero.  The exact input-state cotangent of the invariant loss is
``c``.  A finite Arnoldi approximation need not preserve the invariant, so
the exact VJP of that numerical program can contain a spurious parameter
gradient and a nonphysical state cotangent.

This script compares:

* a dense exact-semigroup reference;
* the two-sided continuous Fréchet VJP;
* Matfree's custom Arnoldi VJP;
* ordinary JAX differentiation through the same Arnoldi program; and
* Matfree with an explicit invariant-preserving output projection.

The explicit projection is the strongest obvious control: it demonstrates
that the discrete program can be repaired when all invariants are known and
encoded.  The claim tested here is not that Matfree can never preserve an
invariant, but that a direct matrix-function derivative inherits common
generator invariants independently of the forward algorithm.

Two systems are included:

1. a birth-death CTMC, with total probability as the invariant;
2. a qutrit Lindbladian represented as an 18-dimensional real-doubled
   operator, with the real trace as the invariant.

Run
---

    .venv-matfree/bin/python experiments/invariant_vjp_audit.py

Environment variables
---------------------

    INVARIANT_M_VALUES=2,3,4,5,6,8,10,12
    INVARIANT_PRECISIONS=float64,float32
    INVARIANT_DT=1.0
    INVARIANT_N_QUAD=12
    INVARIANT_OUTPUT=paper/figures/invariant_vjp_audit.csv
"""

from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg
from matfree import decomp, funm
from numpy.polynomial.legendre import leggauss

from stationary_start_ctmc import initial_state, make_problem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "paper" / "figures" / "invariant_vjp_audit.csv"
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


def real_double(matrix):
    """Map a complex linear operator to its real-doubled representation."""
    matrix = np.asarray(matrix)
    return np.block(
        [
            [matrix.real, -matrix.imag],
            [matrix.imag, matrix.real],
        ]
    )


def hamiltonian_superoperator(hamiltonian):
    dimension = len(hamiltonian)
    identity = np.eye(dimension, dtype=np.complex128)
    return -1.0j * (
        np.kron(identity, hamiltonian)
        - np.kron(hamiltonian.T, identity)
    )


def lindblad_superoperator(operator):
    dimension = len(operator)
    identity = np.eye(dimension, dtype=np.complex128)
    gram = operator.conj().T @ operator
    return (
        np.kron(operator.conj(), operator)
        - 0.5 * np.kron(identity, gram)
        - 0.5 * np.kron(gram.T, identity)
    )


def make_ctmc_problem():
    source = make_problem(24, 16)
    state = initial_state(source, 0.2)
    invariant = np.ones(len(state))
    return {
        "system": "ctmc_mass",
        "matrix": source["base"],
        "directions": source["directions"],
        "state": state,
        "invariant": invariant,
        "physical_dimension": len(state),
        "representation": "real",
    }


def make_lindblad_problem():
    """Construct a trace-preserving qutrit generator and direction library."""
    dimension = 3
    identity = np.eye(dimension, dtype=np.complex128)

    x01 = np.zeros((dimension, dimension), dtype=np.complex128)
    x01[0, 1] = 1.0
    x01[1, 0] = 1.0
    x12 = np.zeros_like(x01)
    x12[1, 2] = 1.0
    x12[2, 1] = 1.0
    y02 = np.zeros_like(x01)
    y02[0, 2] = -1.0j
    y02[2, 0] = 1.0j

    hamiltonian = (
        np.diag(np.asarray((0.0, 0.7, 1.45)))
        + 0.22 * x01
        + 0.17 * x12
    )

    lower10 = np.zeros_like(x01)
    lower10[0, 1] = 1.0
    lower21 = np.zeros_like(x01)
    lower21[1, 2] = 1.0
    dephase = np.diag(np.asarray((1.0, -0.4, -0.6)))

    generator = (
        hamiltonian_superoperator(hamiltonian)
        + 0.35 * lindblad_superoperator(lower10)
        + 0.28 * lindblad_superoperator(lower21)
        + 0.12 * lindblad_superoperator(dephase)
    )
    complex_directions = np.stack(
        [
            hamiltonian_superoperator(x01),
            hamiltonian_superoperator(x12),
            hamiltonian_superoperator(y02),
            lindblad_superoperator(lower10),
            lindblad_superoperator(lower21),
            lindblad_superoperator(dephase),
        ]
    )

    amplitudes = np.asarray((1.0, 0.55j, -0.35), dtype=np.complex128)
    amplitudes /= np.linalg.norm(amplitudes)
    density = 0.82 * np.outer(amplitudes, amplitudes.conj())
    density += 0.18 * identity / dimension
    state_complex = density.reshape(-1, order="F")
    state = np.concatenate((state_complex.real, state_complex.imag))

    trace_complex = np.zeros(dimension**2)
    trace_complex[np.arange(dimension) * (dimension + 1)] = 1.0
    invariant = np.concatenate(
        (trace_complex, np.zeros_like(trace_complex))
    )

    matrix = real_double(generator)
    directions = np.stack(
        [real_double(direction) for direction in complex_directions]
    )
    return {
        "system": "lindblad_trace",
        "matrix": matrix,
        "directions": directions,
        "state": state,
        "invariant": invariant,
        "physical_dimension": dimension,
        "representation": "real_doubled_complex",
    }


def validate_problem(problem):
    matrix = problem["matrix"]
    directions = problem["directions"]
    invariant = problem["invariant"]
    state = problem["state"]

    baseline_residual = np.linalg.norm(invariant @ matrix)
    direction_residual = max(
        np.linalg.norm(invariant @ direction)
        for direction in directions
    )
    initial_invariant = float(np.vdot(invariant, state).real)
    if baseline_residual > 2.0e-13:
        raise AssertionError(
            f"{problem['system']} baseline invariant residual "
            f"{baseline_residual:.3e}"
        )
    if direction_residual > 2.0e-13:
        raise AssertionError(
            f"{problem['system']} direction invariant residual "
            f"{direction_residual:.3e}"
        )
    if abs(initial_invariant - 1.0) > 2.0e-13:
        raise AssertionError(
            f"{problem['system']} initial invariant is {initial_invariant}"
        )
    return baseline_residual, direction_residual


def arnoldi_full(matrix, vector, dimension):
    """Two-pass MGS Arnoldi with dtype-aware happy-breakdown detection."""
    requested = min(dimension, len(vector))
    dtype = np.result_type(matrix, vector)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    beta = np.linalg.norm(vector)
    if beta == 0.0:
        raise ValueError("Arnoldi start vector must be nonzero")

    basis = np.zeros((len(vector), requested), dtype=dtype)
    hessenberg = np.zeros((requested, requested), dtype=dtype)
    basis[:, 0] = vector / beta
    actual = requested
    tolerance = (
        20.0
        * np.finfo(real_dtype).eps
        * max(np.linalg.norm(matrix, ord=1), 1.0)
    )

    for column in range(requested):
        candidate = matrix @ basis[:, column]
        for _ in range(2):
            coefficients = basis[:, : column + 1].T @ candidate
            hessenberg[: column + 1, column] += coefficients
            candidate -= basis[:, : column + 1] @ coefficients

        if column + 1 < requested:
            next_norm = np.linalg.norm(candidate)
            if next_norm <= tolerance:
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
            basis
            @ (
                scipy.linalg.expm(
                    np.asarray(time_value, dtype=hessenberg.dtype)
                    * hessenberg
                )
                @ first
            )
            for time_value in times
        ]
    )


def continuous_invariant_vjp(
    matrix,
    directions,
    state,
    invariant,
    dt,
    dimension,
    quadrature_order,
):
    nodes, weights = leggauss(quadrature_order)
    nodes = nodes.astype(state.dtype)
    weights = weights.astype(state.dtype)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    forward_basis, forward_small, forward_beta = arnoldi_full(
        matrix, state, dimension
    )
    forward_path = projected_path(
        forward_basis,
        forward_small,
        forward_beta,
        dt * nodes,
    )
    adjoint_basis, adjoint_small, adjoint_beta = arnoldi_full(
        matrix.T, invariant, dimension
    )
    adjoint_path = projected_path(
        adjoint_basis,
        adjoint_small,
        adjoint_beta,
        dt * (1.0 - nodes),
    )
    state_vjp = projected_path(
        adjoint_basis,
        adjoint_small,
        adjoint_beta,
        np.asarray([dt], dtype=state.dtype),
    )[:, 0]

    direction_paths = np.einsum(
        "mij,jq->miq", directions, forward_path
    )
    integrands = np.einsum(
        "iq,miq->mq", adjoint_path, direction_paths
    )
    parameter_vjp = dt * (integrands @ weights)
    prediction = projected_path(
        forward_basis,
        forward_small,
        forward_beta,
        np.asarray([dt], dtype=state.dtype),
    )[:, 0]
    return prediction, parameter_vjp, state_vjp


def dense_exact_reference(problem, dt):
    matrix = np.asarray(problem["matrix"], dtype=np.float64)
    directions = np.asarray(problem["directions"], dtype=np.float64)
    state = np.asarray(problem["state"], dtype=np.float64)
    invariant = np.asarray(problem["invariant"], dtype=np.float64)

    prediction = scipy.linalg.expm(dt * matrix) @ state
    state_vjp = scipy.linalg.expm(dt * matrix.T) @ invariant
    parameter_vjp = np.empty(len(directions))
    for index, direction in enumerate(directions):
        frechet = scipy.linalg.expm_frechet(
            dt * matrix, dt * direction, compute_expm=False
        )
        parameter_vjp[index] = np.vdot(
            invariant, frechet @ state
        ).real
    return prediction, parameter_vjp, state_vjp


def make_matfree_invariant_vjp(
    problem,
    dt,
    dimension,
    dtype,
    *,
    custom_vjp,
    project_invariant,
):
    matrix = jnp.asarray(problem["matrix"], dtype=dtype)
    directions = jnp.asarray(problem["directions"], dtype=dtype)
    invariant = jnp.asarray(problem["invariant"], dtype=dtype)
    correction = invariant / jnp.vdot(invariant, invariant).real
    dt_jax = jnp.asarray(dt, dtype=dtype)

    hessenberg = decomp.hessenberg(
        dimension,
        reortho="full",
        custom_vjp=custom_vjp,
        reortho_vjp="match",
    )
    exponential_action = funm.funm_arnoldi(
        funm.dense_funm_pade_exp(), hessenberg
    )

    def action(vector, theta):
        operator = matrix + jnp.tensordot(theta, directions, axes=1)
        return dt_jax * (operator @ vector)

    def predict(theta, state):
        prediction = exponential_action(action, state, theta)
        if project_invariant:
            initial_value = jnp.vdot(invariant, state).real
            predicted_value = jnp.vdot(invariant, prediction).real
            prediction = prediction + correction * (
                initial_value - predicted_value
            )
        return prediction

    def loss(theta, state):
        return jnp.vdot(invariant, predict(theta, state)).real

    compiled_prediction = jax.jit(predict)
    compiled_value_and_grad = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1))
    )

    def evaluate(theta, state):
        theta_jax = jnp.asarray(theta, dtype=dtype)
        state_jax = jnp.asarray(state, dtype=dtype)
        prediction = compiled_prediction(theta_jax, state_jax)
        value, gradients = compiled_value_and_grad(theta_jax, state_jax)
        jax.block_until_ready((prediction, value, gradients))
        parameter_vjp, state_vjp = gradients
        return (
            np.asarray(prediction),
            float(value),
            np.asarray(parameter_vjp),
            np.asarray(state_vjp),
        )

    return evaluate


def audit_row(problem, dt, dimension, precision, quadrature_order):
    np_dtype = np.float64 if precision == "float64" else np.float32
    jax_dtype = jnp.float64 if precision == "float64" else jnp.float32
    matrix = np.asarray(problem["matrix"], dtype=np_dtype)
    directions = np.asarray(problem["directions"], dtype=np_dtype)
    state = np.asarray(problem["state"], dtype=np_dtype)
    invariant = np.asarray(problem["invariant"], dtype=np_dtype)
    theta = np.zeros(len(directions), dtype=np_dtype)
    invariant_value = float(np.vdot(invariant, state).real)

    exact_prediction, exact_parameter, exact_state = dense_exact_reference(
        problem, dt
    )
    continuous_prediction, continuous_parameter, continuous_state = (
        continuous_invariant_vjp(
            matrix,
            directions,
            state,
            invariant,
            np_dtype(dt),
            dimension,
            quadrature_order,
        )
    )
    correction = invariant / np.vdot(invariant, invariant).real
    continuous_projected_prediction = (
        continuous_prediction
        + correction
        * (
            invariant_value
            - np.vdot(invariant, continuous_prediction).real
        )
    )

    matfree = make_matfree_invariant_vjp(
        problem,
        dt,
        dimension,
        jax_dtype,
        custom_vjp=True,
        project_invariant=False,
    )
    matfree_native = make_matfree_invariant_vjp(
        problem,
        dt,
        dimension,
        jax_dtype,
        custom_vjp=False,
        project_invariant=False,
    )
    matfree_projected = make_matfree_invariant_vjp(
        problem,
        dt,
        dimension,
        jax_dtype,
        custom_vjp=True,
        project_invariant=True,
    )

    (
        matfree_prediction,
        matfree_value,
        matfree_parameter,
        matfree_state,
    ) = matfree(theta, state)
    (
        native_prediction,
        native_value,
        native_parameter,
        native_state,
    ) = matfree_native(theta, state)
    (
        projected_prediction,
        projected_value,
        projected_parameter,
        projected_state,
    ) = matfree_projected(theta, state)

    def finite(*arrays):
        return int(all(np.all(np.isfinite(array)) for array in arrays))

    baseline_residual, direction_residual = validate_problem(problem)
    return {
        "system": problem["system"],
        "representation": problem["representation"],
        "state_dimension": len(state),
        "physical_dimension": problem["physical_dimension"],
        "M": len(directions),
        "precision": precision,
        "dt": dt,
        "m": dimension,
        "n_quad": quadrature_order,
        "baseline_invariant_residual": baseline_residual,
        "max_direction_invariant_residual": direction_residual,
        "exact_forward_invariant_defect": abs(
            np.vdot(invariant, exact_prediction).real - invariant_value
        ),
        "exact_parameter_vjp_norm": float(
            np.linalg.norm(exact_parameter)
        ),
        "exact_state_vjp_relerr": relative_error(
            exact_state, problem["invariant"]
        ),
        "continuous_forward_relerr": relative_error(
            continuous_prediction, exact_prediction
        ),
        "continuous_forward_invariant_defect": abs(
            np.vdot(invariant, continuous_prediction).real
            - invariant_value
        ),
        "continuous_projected_forward_relerr": relative_error(
            continuous_projected_prediction, exact_prediction
        ),
        "continuous_projected_forward_invariant_defect": abs(
            np.vdot(invariant, continuous_projected_prediction).real
            - invariant_value
        ),
        "continuous_parameter_vjp_norm": float(
            np.linalg.norm(continuous_parameter)
        ),
        "continuous_state_vjp_relerr": relative_error(
            continuous_state, problem["invariant"]
        ),
        "matfree_forward_relerr": relative_error(
            matfree_prediction, exact_prediction
        ),
        "matfree_forward_invariant_defect": abs(
            matfree_value - invariant_value
        ),
        "matfree_parameter_vjp_norm": float(
            np.linalg.norm(matfree_parameter)
        ),
        "matfree_state_vjp_relerr": relative_error(
            matfree_state, problem["invariant"]
        ),
        "matfree_finite": finite(
            matfree_prediction, matfree_parameter, matfree_state
        ),
        "native_forward_value_difference": abs(
            native_value - matfree_value
        ),
        "custom_vs_native_parameter_relerr": (
            relative_error(matfree_parameter, native_parameter)
            if finite(
                matfree_parameter, native_parameter
            )
            else np.nan
        ),
        "custom_vs_native_state_relerr": (
            relative_error(matfree_state, native_state)
            if finite(matfree_state, native_state)
            else np.nan
        ),
        "native_finite": finite(
            native_prediction, native_parameter, native_state
        ),
        "projected_forward_relerr": relative_error(
            projected_prediction, exact_prediction
        ),
        "projected_forward_invariant_defect": abs(
            projected_value - invariant_value
        ),
        "projected_parameter_vjp_norm": float(
            np.linalg.norm(projected_parameter)
        ),
        "projected_state_vjp_relerr": relative_error(
            projected_state, problem["invariant"]
        ),
        "projected_finite": finite(
            projected_prediction, projected_parameter, projected_state
        ),
        "jax_version": jax.__version__,
        "matfree_version": importlib.metadata.version("matfree"),
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
    }


def main():
    dimensions = parse_values(
        "INVARIANT_M_VALUES", "2,3,4,5,6,8,10,12", int
    )
    precisions = parse_values(
        "INVARIANT_PRECISIONS", "float64,float32", str
    )
    dt = float(os.getenv("INVARIANT_DT", "1.0"))
    quadrature_order = int(os.getenv("INVARIANT_N_QUAD", "12"))
    output = Path(os.getenv("INVARIANT_OUTPUT", str(DEFAULT_OUTPUT)))
    problems = (make_ctmc_problem(), make_lindblad_problem())

    rows = []
    print(
        "linear-invariant VJP audit: "
        f"m={dimensions}, precision={precisions}, dt={dt:g}"
    )
    print(
        "  system             precision  m   forward defect   "
        "continuous |g|    Matfree |g|    projected |g|"
    )
    for problem in problems:
        validate_problem(problem)
        for precision in precisions:
            # Matfree's Arnoldi adjoint constructs a few untyped scalar/array
            # constants.  Match the package's own Float32 benchmark setup by
            # disabling JAX x64 globally for Float32 runs, so those constants
            # are not promoted to Float64 inside the custom VJP.
            jax.config.update("jax_enable_x64", precision == "float64")
            for dimension in dimensions:
                if dimension > len(problem["state"]):
                    continue
                row = audit_row(
                    problem,
                    dt,
                    dimension,
                    precision,
                    quadrature_order,
                )
                rows.append(row)
                print(
                    f"  {problem['system']:18s} "
                    f"{precision:9s} {dimension:2d}  "
                    f"{row['matfree_forward_invariant_defect']:14.3e}  "
                    f"{row['continuous_parameter_vjp_norm']:14.3e}  "
                    f"{row['matfree_parameter_vjp_norm']:12.3e}  "
                    f"{row['projected_parameter_vjp_norm']:13.3e}"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
