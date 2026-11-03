"""Audit dense JAX and PyTorch AD against two Float64 references.

The problem matches the 80-state Fokker--Planck comparison used by
``kraemer_comparison.py``.  SciPy's ``expm_frechet`` supplies the primary
parameter-gradient reference.  An independent 160-by-160 block exponential
checks that reference.  The script writes only a new audit CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import scipy.linalg
import torch

import kraemer_comparison as comparison


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "dense_ad_accuracy_audit.csv"


def relative_error(value, reference):
    value = np.asarray(value, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(value - reference)
        / max(np.linalg.norm(reference), 1.0e-14)
    )


def block_frechet(matrix, direction):
    n = matrix.shape[0]
    block = np.block(
        [[matrix, direction], [np.zeros_like(matrix), matrix]]
    )
    return scipy.linalg.expm(block)[:n, n:]


def jax_dense_vjp(base, directions, theta, initial, target, dt):
    base_jax = jnp.asarray(base)
    directions_jax = jnp.asarray(directions)
    theta_jax = jnp.asarray(theta)
    initial_jax = jnp.asarray(initial)
    target_jax = jnp.asarray(target)

    def loss(coefficients, state):
        matrix = base_jax + jnp.einsum(
            "m,mij->ij", coefficients, directions_jax
        )
        residual = jsp_linalg.expm(matrix * dt) @ state - target_jax
        return 0.5 * jnp.vdot(residual, residual).real

    _, gradients = jax.value_and_grad(loss, argnums=(0, 1))(
        theta_jax, initial_jax
    )
    return tuple(np.asarray(item) for item in gradients)


def torch_dense_vjp(base, directions, theta, initial, target, dt):
    base_torch = torch.tensor(base, dtype=torch.float64)
    directions_torch = torch.tensor(directions, dtype=torch.float64)
    theta_torch = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    initial_torch = torch.tensor(
        initial, dtype=torch.float64, requires_grad=True
    )
    target_torch = torch.tensor(target, dtype=torch.float64)
    matrix = base_torch + torch.einsum(
        "m,mij->ij", theta_torch, directions_torch
    )
    residual = torch.linalg.matrix_exp(matrix * dt) @ initial_torch - target_torch
    loss = 0.5 * torch.vdot(residual, residual).real
    parameter_vjp, state_vjp = torch.autograd.grad(
        loss, (theta_torch, initial_torch)
    )
    return parameter_vjp.detach().numpy(), state_vjp.detach().numpy()


def main():
    base, directions, theta_true, theta_eval, initial = (
        comparison.build_fpe_library()
    )
    rows = []
    for dt in (0.3, 1.0):
        target, _, parameter_reference, state_reference = (
            comparison.exact_reference(
                base,
                directions,
                theta_true,
                theta_eval,
                initial,
                dt,
            )
        )
        matrix = base + np.einsum(
            "m,mij->ij", theta_eval, directions
        )
        prediction = scipy.linalg.expm(matrix * dt) @ initial
        upstream = prediction - target
        block_parameter_reference = np.asarray(
            [
                upstream
                @ (block_frechet(matrix * dt, direction * dt) @ initial)
                for direction in directions
            ]
        )
        block_state_reference = (
            scipy.linalg.expm(matrix.T * dt) @ upstream
        )

        methods = {
            "SciPy expm_frechet vs block expm": (
                parameter_reference,
                state_reference,
                block_parameter_reference,
                block_state_reference,
            ),
            "JAX dense AD": (
                *jax_dense_vjp(
                    base, directions, theta_eval, initial, target, dt
                ),
                parameter_reference,
                state_reference,
            ),
            "PyTorch dense AD": (
                *torch_dense_vjp(
                    base, directions, theta_eval, initial, target, dt
                ),
                parameter_reference,
                state_reference,
            ),
        }
        for method, (
            parameter_value,
            state_value,
            parameter_target,
            state_target,
        ) in methods.items():
            rows.append(
                {
                    "dt": dt,
                    "method": method,
                    "parameter_vjp_relerr": relative_error(
                        parameter_value, parameter_target
                    ),
                    "state_vjp_relerr": relative_error(
                        state_value, state_target
                    ),
                    "parameter_reference_norm": float(
                        np.linalg.norm(parameter_reference)
                    ),
                    "state_reference_norm": float(
                        np.linalg.norm(state_reference)
                    ),
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"dt={row['dt']:.1f} {row['method']:<34s} "
            f"parameter={row['parameter_vjp_relerr']:.3e} "
            f"state={row['state_vjp_relerr']:.3e}"
        )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
