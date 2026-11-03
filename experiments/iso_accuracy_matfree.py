"""Fine-grained Matfree CPU sweep for the iso-accuracy comparison.

This imports the already-audited matched benchmark implementation from
`kraemer_comparison.py` and writes to a separate file.  Compilation and warmup
are excluded from each steady-state full-loss/VJP timing.
"""

from __future__ import annotations

import csv
import os
import platform
from pathlib import Path

import jax.numpy as jnp

import kraemer_comparison as comparison


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    os.environ.get(
        "ISO_MATFREE_OUTPUT",
        ROOT / "paper" / "figures" / "iso_accuracy_matfree_raw.csv",
    )
)


def parse_values(name: str, default: str, cast):
    return tuple(cast(value) for value in os.getenv(name, default).split(","))


M_VALUES = parse_values(
    "ISO_M_VALUES",
    "5,10,15,20,25,30,35,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,79,80",
    int,
)
DT_VALUES = parse_values("ISO_DT_VALUES", "0.3,1.0", float)
REPEATS = int(os.getenv("ISO_REPEATS", "21"))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    comparison.REPEATS = REPEATS
    base, directions, theta_true, theta_eval, initial = (
        comparison.build_fpe_library()
    )
    rows = []
    for dt in DT_VALUES:
        target, prediction_reference, parameter_reference, state_reference = (
            comparison.exact_reference(
                base, directions, theta_true, theta_eval, initial, dt
            )
        )
        for krylov_dimension in M_VALUES:
            source = comparison.benchmark_configuration(
                base=base,
                directions=directions,
                theta_eval=theta_eval,
                initial=initial,
                target=target,
                prediction_reference=prediction_reference,
                parameter_reference=parameter_reference,
                state_reference=state_reference,
                dt=dt,
                krylov_dimension=krylov_dimension,
                dtype=jnp.float64,
            )
            parameter_error = float(source["parameter_vjp_relerr"])
            state_error = float(source["state_vjp_relerr"])
            forward_error = float(source["forward_relerr"])
            row = {
                "backend": "Matfree_JAX_CPU",
                "method": "matfree_finite_arnoldi",
                "variant": "Matfree",
                "system": "FPE",
                "precision": "float64",
                "n": source["n"],
                "M": source["M"],
                "dt": dt,
                "requested_m": krylov_dimension,
                "forward_m": krylov_dimension,
                "adjoint_m": krylov_dimension,
                "n_quad": "",
                "core_rtol": "",
                "core_rank": "",
                "core_full_rank": "",
                "core_relative_tail": "",
                "forward_relerr": forward_error,
                "parameter_vjp_relerr": parameter_error,
                "state_vjp_relerr": state_error,
                "max_vjp_relerr": max(parameter_error, state_error),
                "overall_relerr": max(
                    forward_error, parameter_error, state_error
                ),
                "loss_and_full_vjp_ms": source["loss_and_full_vjp_ms"],
                "repeats": REPEATS,
                "reorthogonalization": source["reorthogonalization"],
                "gradient_semantics": source["gradient_semantics"],
                "platform": source["platform"],
                "cpu_name": platform.processor() or platform.machine(),
            }
            rows.append(row)
            print(
                f"Matfree dt={dt:3.1f} m={krylov_dimension:2d} "
                f"theta={parameter_error:9.2e} "
                f"state={state_error:9.2e} "
                f"time={source['loss_and_full_vjp_ms']:7.3f} ms"
            )
    write_rows(OUTPUT, rows)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
