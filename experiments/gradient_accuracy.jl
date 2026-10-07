"""
Gradient-accuracy audit for the shared-Krylov pullback.

Produces paper/figures/gradient_accuracy.csv with parameter- and state-VJP
errors against dense block-Fréchet references over Krylov dimensions,
quadrature orders, time steps, and both real FPE and complex Lindblad systems.

Run:
    julia --project=. experiments/gradient_accuracy.jl
"""

using LinearAlgebra, SparseArrays, Printf
using CSV, DataFrames
using ExponentialUtilities: expv

include("lindblad_library.jl")
include("../src/fpe.jl")

relative_error(x, ref) = norm(x - ref) / max(norm(ref), 1e-14)

function audit_case!(
    rows, system, A0, dirs, θ_true, θ_eval, p0, dt;
    m_values, n_quad_values,
)
    M = length(dirs)
    A_true = A0 + sum(θ_true[j] * dirs[j] for j in 1:M)
    A_eval = A0 + sum(θ_eval[j] * dirs[j] for j in 1:M)
    target = exp(Matrix(A_true) * dt) * p0
    pred = exp(Matrix(A_eval) * dt) * p0
    upstream = pred - target

    # One dense 2n×2n block exponential gives the exact adjoint Fréchet
    # matrix. Projecting it onto every sparse direction produces a reference
    # VJP without finite-difference step-size error.
    n = length(p0)
    AdjointBlock = Matrix(A_eval') * dt
    X = dt .* (upstream * p0')
    Z = zeros(eltype(AdjointBlock), n, n)
    G = exp([AdjointBlock X; Z AdjointBlock])[1:n, n+1:2n]
    θ_ref = [real(dot(G, Matrix(dirs[j]))) for j in 1:M]
    p_ref = exp(Matrix(A_eval)' * dt) * upstream

    for n_quad in n_quad_values, m in m_values
        θ_vjp = krylov_frechet_vjp(
            A_eval, dirs, p0, dt, upstream;
            m_krylov=m, n_quad=n_quad,
        )
        p_vjp = expv(dt, A_eval', upstream; m=m)
        row = (
            system=system,
            n=length(p0),
            M=M,
            dt=dt,
            m=m,
            n_quad=n_quad,
            reference="dense_block_frechet",
            parameter_vjp_relerr=relative_error(θ_vjp, θ_ref),
            state_vjp_relerr=relative_error(p_vjp, p_ref),
        )
        push!(rows, row)
        @printf(
            "%-12s dt=%4.1f m=%2d nq=%d θerr=%9.2e perr=%9.2e\n",
            system, dt, m, n_quad,
            row.parameter_vjp_relerr, row.state_vjp_relerr,
        )
    end
end

rows = NamedTuple[]

# Real, stiff, nonnormal Fokker--Planck generator.
let
    n = 80
    M = 10
    L = 2.0
    D = 0.5
    xs = collect(range(-L, L; length=n))
    drift_fns = [let k=j, L=L; x -> (x / L)^k end for j in 1:M]
    A0, dirs = fpe_library_1d(xs, drift_fns; D=D)
    θ_true = zeros(M)
    θ_true[1] = -4.0
    θ_true[3] = 4.0
    θ_eval = θ_true + 0.08 .* collect(range(-1, 1; length=M))
    p0 = exp.(-0.5 .* xs .^ 2)
    p0 ./= sum(p0)

    for dt in (0.1, 0.3, 1.0)
        audit_case!(
            rows, "FPE", A0, dirs, θ_true, θ_eval, p0, dt;
            m_values=(5, 10, 20, 30, 40, 60),
            n_quad_values=(3, 5, 8, 12, 16, 24, 32),
        )
    end
end

# Complex-valued open-system generator using the paper's two-qubit library.
let
    n_qubits = 2
    A0, dirs = build_ising_library(n_qubits)
    θ_true = true_params(0.5, n_qubits)
    θ_eval = max.(θ_true .* (1 .+ 0.2 .* sin.(1:length(θ_true))) .+ 0.03, 1e-4)
    p0 = initial_density_vec(n_qubits)

    for dt in (0.3, 1.0, 3.0)
        audit_case!(
            rows, "Lindblad", A0, dirs, θ_true, θ_eval, p0, dt;
            m_values=(5, 10, 15),
            n_quad_values=(3, 5, 8, 12, 16, 24, 32),
        )
    end
end

out_path = joinpath(@__DIR__, "..", "paper", "figures", "gradient_accuracy.csv")
CSV.write(out_path, DataFrame(rows))
println("\nSaved $(out_path)")
