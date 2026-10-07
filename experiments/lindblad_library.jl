# Flux-free Lindblad library helpers, split out of lindblad_structured.jl so that
# accuracy audits can use them without pulling in the (dropped) neural encoder.

using LinearAlgebra, Printf, Random, Statistics
using ExponentialUtilities: expv

include("../src/shared_krylov.jl")
include("../src/lindblad.jl")

const H_FIELD_LINDBLAD = 0.3
const DTS_LINDBLAD = [0.3, 1.0, 3.0]
const TRAIN_CTXS_LINDBLAD = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
const TEST_CTXS_LINDBLAD  = [0.10, 0.30, 0.50, 0.70, 0.90]

J_of(c, i) = 0.5 * (1 + 0.4 * sin(i * π * c))
γ_of(c, i) = 0.1 * (1 + 0.5 * cos(i * π * c)^2)

true_params(c, n) = vcat([J_of(c, i) for i in 1:n-1], [γ_of(c, i) for i in 1:n])

param_labels(n) = vcat(["J$(i)" for i in 1:n-1], ["gamma$(i)" for i in 1:n])

function build_ising_library(n::Int; sparse_operators=false)
    site_op = sparse_operators ? nsite_op_sparse : nsite_op
    h_superop = sparse_operators ? hamiltonian_superop_sparse : hamiltonian_superop
    d_superop = sparse_operators ? lindblad_superop_sparse : lindblad_superop

    H_field = sum(site_op(σz, i, n) for i in 1:n)
    A_const = H_FIELD_LINDBLAD * h_superop(H_field)

    dirs = AbstractMatrix{ComplexF64}[]
    for i in 1:n-1
        H_i = site_op(σx, i, n) * site_op(σx, i + 1, n) +
              site_op(σy, i, n) * site_op(σy, i + 1, n)
        push!(dirs, h_superop(H_i))
    end
    for i in 1:n
        push!(dirs, d_superop(site_op(σz, i, n)))
    end

    return A_const, dirs
end

function initial_density_vec(n::Int)
    plus = ComplexF64[1, 1] / sqrt(2)
    ψ0 = foldl(kron, [plus for _ in 1:n])
    vec(ψ0 * ψ0')
end

interval_dts(abs_times::AbstractVector) = vcat(abs_times[1], diff(abs_times))
