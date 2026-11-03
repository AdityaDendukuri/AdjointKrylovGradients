"""
Float32/Float64 forward- and backward-accuracy comparison.

The reference is a Float64 dense exponential plus a dense 2n×2n block-Fréchet
exponential. The tested path includes the Krylov forward approximation, so the
reported VJP error captures both forward-state and backward approximation
effects.

Run:
    julia --project=experiments experiments/gradient_precision.jl
"""

using LinearAlgebra, SparseArrays, Printf
using CSV, DataFrames
using ExponentialUtilities: expv

include("lindblad_structured.jl")
include("../src/fpe.jl")

relative_error(x, ref) = norm(x - ref) / max(norm(ref), 1e-14)

function precision_type(T, precision)
    precision == :float64 && return T <: Complex ? ComplexF64 : Float64
    precision == :float32 && return T <: Complex ? ComplexF32 : Float32
    error("Unsupported precision $precision")
end

function audit_precision!(
    rows, system, A0, dirs, θ_true, θ_eval, p0, dt, m, n_quad,
)
    M = length(dirs)
    A_true64 = Matrix(A0 + sum(θ_true[j] * dirs[j] for j in 1:M))
    A_eval64 = Matrix(A0 + sum(θ_eval[j] * dirs[j] for j in 1:M))
    p064 = eltype(p0) <: Complex ? ComplexF64.(p0) : Float64.(p0)
    target64 = exp(A_true64 * dt) * p064
    pred64 = exp(A_eval64 * dt) * p064
    upstream64 = pred64 - target64

    n = length(p0)
    Aadj = A_eval64' * dt
    X = dt .* (upstream64 * p064')
    Z = zeros(eltype(Aadj), n, n)
    G = exp([Aadj X; Z Aadj])[1:n, n+1:2n]
    θ_ref = [real(dot(G, Matrix(dirs[j]))) for j in 1:M]
    p_ref = exp(A_eval64' * dt) * upstream64

    for precision in (:float64, :float32)
        T = precision_type(eltype(p0), precision)
        RT = real_scalar_type(T)
        A0p = convert(SparseMatrixCSC{T,Int}, sparse(A0))
        dirsp = SparseMatrixCSC{T,Int}[
            convert(SparseMatrixCSC{T,Int}, sparse(dir)) for dir in dirs
        ]
        θp = RT.(θ_eval)
        p0p = T.(p0)
        targetp = T.(target64)
        dtp = RT(dt)
        A_evalp = A0p + sum(θp[j] * dirsp[j] for j in 1:M)
        predp = expv(dtp, A_evalp, p0p; m=m)
        upstreamp = predp - targetp
        θ_vjp = krylov_frechet_vjp(
            A_evalp, dirsp, p0p, dtp, upstreamp;
            m_krylov=m, n_quad=n_quad,
        )
        p_vjp = expv(dtp, A_evalp', upstreamp; m=m)

        row = (
            system=system,
            n=n,
            M=M,
            dt=dt,
            m=m,
            n_quad=n_quad,
            precision=String(precision),
            reference="Float64_dense_block_frechet",
            forward_relerr=relative_error(predp, pred64),
            parameter_vjp_relerr=relative_error(θ_vjp, θ_ref),
            state_vjp_relerr=relative_error(p_vjp, p_ref),
        )
        push!(rows, row)
        @printf(
            "%-10s %-7s dt=%3.1f m=%2d forward=%9.2e θ=%9.2e state=%9.2e\n",
            system, String(precision), dt, m,
            row.forward_relerr,
            row.parameter_vjp_relerr,
            row.state_vjp_relerr,
        )
    end
end

rows = NamedTuple[]

let
    n = 80
    M = 10
    xs = collect(range(-2.0, 2.0; length=n))
    drift_fns = [let k=j; x -> (x / 2.0)^k end for j in 1:M]
    A0, dirs = fpe_library_1d(xs, drift_fns; D=0.5)
    θ_true = zeros(M)
    θ_true[1] = -4.0
    θ_true[3] = 4.0
    θ_eval = θ_true + 0.08 .* collect(range(-1, 1; length=M))
    p0 = exp.(-0.5 .* xs .^ 2)
    p0 ./= sum(p0)

    audit_precision!(rows, "FPE", A0, dirs, θ_true, θ_eval, p0, 0.3, 40, 8)
    audit_precision!(rows, "FPE", A0, dirs, θ_true, θ_eval, p0, 1.0, 60, 8)
end

let
    A0, dirs = build_ising_library(2; sparse_operators=true)
    θ_true = true_params(0.5, 2)
    θ_eval = max.(θ_true .* (1 .+ 0.2 .* sin.(1:length(θ_true))) .+ 0.03, 1e-4)
    p0 = initial_density_vec(2)
    audit_precision!(rows, "Lindblad", A0, dirs, θ_true, θ_eval, p0, 1.0, 10, 8)
end

out_path = joinpath(
    @__DIR__, "..", "paper", "figures", "gradient_precision.csv",
)
CSV.write(out_path, DataFrame(rows))
println("\nSaved $out_path")
