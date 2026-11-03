"""
Accuracy/rank/runtime audit for the quadrature-free compressed sensitivity
core versus the submitted fixed-node Gauss--Legendre VJP.

Outputs:
  paper/figures/compressed_core_benchmark.csv

Environment variables:
  CORE_BENCH_REPEATS  timing repetitions after warmup (default: 5)
  CORE_BENCH_OUTPUT   output CSV path
"""

using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames

include("lindblad_structured.jl")
include("../src/fpe.jl")

const CORE_REPEATS = parse(Int, get(ENV, "CORE_BENCH_REPEATS", "5"))
const CORE_OUTPUT = get(
    ENV,
    "CORE_BENCH_OUTPUT",
    joinpath(
        @__DIR__, "..", "paper", "figures",
        "compressed_core_benchmark.csv",
    ),
)

relative_error(value, reference) =
    norm(value - reference) / max(norm(reference), 1e-14)

function median_time_ms(f)
    f()
    samples = Float64[]
    for _ in 1:CORE_REPEATS
        push!(samples, 1e3 * @elapsed f())
    end
    median(samples)
end

function dense_parameter_reference(A, dirs, p0, dt, upstream)
    n = length(p0)
    adjoint_block = Matrix(A') * dt
    direction_seed = dt .* (upstream * p0')
    zero_block = zeros(eltype(adjoint_block), n, n)
    frechet_adjoint = exp([
        adjoint_block direction_seed
        zero_block adjoint_block
    ])[1:n, (n + 1):(2n)]
    [real(dot(frechet_adjoint, Matrix(direction))) for direction in dirs]
end

function benchmark_case!(
    rows, system, A0, dirs, θ_true, θ_eval, p0, dt, m,
)
    A_true = A0 + sum(
        θ_true[index] * dirs[index] for index in eachindex(dirs)
    )
    A_eval = A0 + sum(
        θ_eval[index] * dirs[index] for index in eachindex(dirs)
    )
    target = exp(Matrix(A_true) * dt) * p0
    prediction = exp(Matrix(A_eval) * dt) * p0
    exact_upstream = prediction - target
    reference = dense_parameter_reference(
        A_eval, dirs, p0, dt, exact_upstream,
    )
    forward_workspace = ArnoldiWorkspace(eltype(p0), length(p0), m)
    forward_basis = arnoldi_basis!(
        forward_workspace, A_eval, p0; m=m,
    )
    approximate_prediction = _eval_krylov_batch(
        forward_basis, [dt], forward_basis.beta,
    )[:, 1]
    upstream = approximate_prediction - target

    for n_quad in (5, 8)
        gradient = krylov_frechet_vjp(
            A_eval, dirs, p0, dt, upstream;
            m_krylov=m,
            n_quad=n_quad,
        )
        runtime = median_time_ms() do
            krylov_frechet_vjp(
                A_eval, dirs, p0, dt, upstream;
                m_krylov=m,
                n_quad=n_quad,
            )
        end
        push!(rows, (
            system=system,
            n=length(p0),
            M=length(dirs),
            dt=dt,
            m=m,
            method="gauss_legendre",
            n_quad=n_quad,
            core_rtol=NaN,
            core_rank=missing,
            core_full_rank=missing,
            core_relative_tail=NaN,
            max_direction_error_bound=NaN,
            parameter_vjp_relerr=relative_error(gradient, reference),
            runtime_ms=runtime,
        ))
    end

    for core_rtol in (1e-2, 1e-3, 1e-5, 1e-7, 0.0)
        diagnostics = compressed_core_frechet_vjp(
            A_eval, dirs, p0, dt, upstream;
            m_krylov=m,
            core_rtol=core_rtol,
            return_diagnostics=true,
        )
        runtime = median_time_ms() do
            compressed_core_frechet_vjp(
                A_eval, dirs, p0, dt, upstream;
                m_krylov=m,
                core_rtol=core_rtol,
            )
        end
        push!(rows, (
            system=system,
            n=length(p0),
            M=length(dirs),
            dt=dt,
            m=m,
            method="compressed_core",
            n_quad=missing,
            core_rtol=core_rtol,
            core_rank=diagnostics.core_rank,
            core_full_rank=diagnostics.core_full_rank,
            core_relative_tail=diagnostics.core_relative_tail,
            max_direction_error_bound=maximum(
                diagnostics.direction_error_bounds,
            ),
            parameter_vjp_relerr=relative_error(
                diagnostics.gradient, reference,
            ),
            runtime_ms=runtime,
        ))
    end

    subset = rows[(end - 6):end]
    for row in subset
        if row.method == "gauss_legendre"
            @printf(
                "%-9s dt=%3.1f m=%2d GL%d       err=%9.2e time=%7.3f ms\n",
                system, dt, m, row.n_quad,
                row.parameter_vjp_relerr, row.runtime_ms,
            )
        else
            @printf(
                "%-9s dt=%3.1f m=%2d core %2d/%2d err=%9.2e time=%7.3f ms rtol=%5.0e\n",
                system, dt, m,
                row.core_rank, row.core_full_rank,
                row.parameter_vjp_relerr, row.runtime_ms,
                row.core_rtol,
            )
        end
    end
end

rows = NamedTuple[]

let
    n = 80
    M = 10
    domain_half_length = 2.0
    xs = collect(range(-domain_half_length, domain_half_length; length=n))
    drift_functions = [
        let power=index, scale=domain_half_length
            x -> (x / scale)^power
        end for index in 1:M
    ]
    A0, dirs = fpe_library_1d(xs, drift_functions; D=0.5)
    θ_true = zeros(M)
    θ_true[1] = -4.0
    θ_true[3] = 4.0
    θ_eval =
        θ_true + 0.08 .* collect(range(-1, 1; length=M))
    p0 = exp.(-0.5 .* xs .^ 2)
    p0 ./= sum(p0)
    for dt in (0.3, 1.0), m in (30, 40, 60)
        benchmark_case!(
            rows, "FPE", A0, dirs, θ_true, θ_eval, p0, dt, m,
        )
    end
end

let
    n_qubits = 2
    A0, dirs = build_ising_library(n_qubits)
    θ_true = true_params(0.5, n_qubits)
    θ_eval = max.(
        θ_true .* (1 .+ 0.2 .* sin.(1:length(θ_true))) .+ 0.03,
        1e-4,
    )
    p0 = initial_density_vec(n_qubits)
    for dt in (1.0, 3.0), m in (10, 15)
        benchmark_case!(
            rows, "Lindblad", A0, dirs, θ_true, θ_eval, p0, dt, m,
        )
    end
end

mkpath(dirname(CORE_OUTPUT))
CSV.write(CORE_OUTPUT, DataFrame(rows))
println("\nSaved $(CORE_OUTPUT)")
