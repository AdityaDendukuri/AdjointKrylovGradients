"""
Shared-Krylov amortization on a large-domain Kuramoto--Sivashinsky operator.

The grid spacing is held at the n=64, L=22 experiment value while the domain
and state dimension grow together.  Candidate affine directions modulate the
second- and fourth-derivative terms by distinct Fourier coefficient fields.
The benchmark compares one VJP over the complete library with M one-direction
VJPs, which rebuild the same forward and adjoint Arnoldi spaces M times.

Environment variables:
  KS_AMORT_N              state dimension (default: 512)
  KS_AMORT_LIBRARIES      comma-separated library sizes (default: 16,64,256)
  KS_AMORT_M_KRYLOV       Krylov depth (default: 40)
  KS_AMORT_N_QUAD         quadrature nodes (default: 8)
  KS_AMORT_REPEATS        timed repetitions (default: 3)
  KS_AMORT_OUTPUT_DIR     output directory (default: paper/figures)
  KS_AMORT_SMOKE          small fast configuration when set to 1
"""

using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames

include("../src/shared_krylov.jl")

parse_list(::Type{T}, name, default) where {T} =
    parse.(T, split(get(ENV, name, default), ","))

const SMOKE = get(ENV, "KS_AMORT_SMOKE", "0") == "1"
const N = parse(Int, get(ENV, "KS_AMORT_N", SMOKE ? "64" : "512"))
const LIBRARY_SIZES = parse_list(
    Int,
    "KS_AMORT_LIBRARIES",
    SMOKE ? "4,8" : "16,64,256",
)
const M_KRYLOV = parse(
    Int,
    get(ENV, "KS_AMORT_M_KRYLOV", SMOKE ? "15" : "40"),
)
const N_QUAD = parse(Int, get(ENV, "KS_AMORT_N_QUAD", SMOKE ? "4" : "8"))
const REPEATS = parse(Int, get(ENV, "KS_AMORT_REPEATS", SMOKE ? "1" : "3"))
const OUTPUT_DIR = get(
    ENV,
    "KS_AMORT_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)
const BASE_DX = 22.0 / 64
const DOMAIN_LENGTH = BASE_DX * N
const DT = 0.0125

function periodic_derivatives(n, domain_length)
    dx = domain_length / n
    first = spdiagm(
        -1 => fill(-1.0 / (2dx), n - 1),
         1 => fill(1.0 / (2dx), n - 1),
    )
    first[1, n] = -1.0 / (2dx)
    first[n, 1] = 1.0 / (2dx)
    second = spdiagm(
        -1 => fill(1.0 / dx^2, n - 1),
         0 => fill(-2.0 / dx^2, n),
         1 => fill(1.0 / dx^2, n - 1),
    )
    second[1, n] = 1.0 / dx^2
    second[n, 1] = 1.0 / dx^2
    first, second, sparse(second * second)
end

function coefficient_field(xs, index)
    mode = fld(index - 1, 2) + 1
    isodd(index) ? cos.(2π .* mode .* xs ./ DOMAIN_LENGTH) :
                   sin.(2π .* mode .* xs ./ DOMAIN_LENGTH)
end

function direction_library(xs, second, fourth, count)
    directions = SparseMatrixCSC{Float64,Int}[]
    for index in 1:count
        field = coefficient_field(xs, index)
        derivative = isodd(fld(index - 1, 2)) ? second : fourth
        scale = derivative === second ? 1.0 : BASE_DX^2
        push!(
            directions,
            sparse(spdiagm(0 => field) * (-scale .* derivative)),
        )
    end
    directions
end

function best_time(f; repeats=REPEATS)
    minimum(@elapsed f() for _ in 1:repeats)
end

N >= 16 || error("KS_AMORT_N must be at least 16")
all(>(0), LIBRARY_SIZES) || error("library sizes must be positive")

xs = collect(range(0, DOMAIN_LENGTH; length=N + 1))[1:end-1]
_, second, fourth = periodic_derivatives(N, DOMAIN_LENGTH)
A = sparse(-second - fourth)
phase = 2π .* xs ./ DOMAIN_LENGTH
state = 0.72 .* cos.(8 .* phase) .+
        0.22 .* sin.(13 .* phase) .-
        0.10 .* cos.(21 .* phase)
cotangent = 0.55 .* sin.(7 .* phase .+ 0.2) .+
            0.31 .* cos.(17 .* phase .- 0.4)

println("KS shared-Krylov amortization")
@printf(
    "  n=%d L=%.1f dx=%.5f dt=%.4f m=%d nq=%d repeats=%d\n",
    N,
    DOMAIN_LENGTH,
    BASE_DX,
    DT,
    M_KRYLOV,
    N_QUAD,
    REPEATS,
)

rows = NamedTuple[]
for library_size in LIBRARY_SIZES
    directions = direction_library(xs, second, fourth, library_size)
    shared() = krylov_frechet_vjp(
        A,
        directions,
        state,
        DT,
        cotangent;
        m_krylov=M_KRYLOV,
        n_quad=N_QUAD,
    )
    repeated() = [
        only(krylov_frechet_vjp(
            A,
            [direction],
            state,
            DT,
            cotangent;
            m_krylov=M_KRYLOV,
            n_quad=N_QUAD,
        ))
        for direction in directions
    ]

    shared_gradient = shared()
    repeated_gradient = repeated()
    relative_difference = norm(shared_gradient - repeated_gradient) /
                          max(norm(repeated_gradient), eps())
    shared_seconds = best_time(shared)
    repeated_seconds = best_time(repeated)
    speedup = repeated_seconds / shared_seconds
    push!(rows, (
        state_dimension=N,
        domain_length=DOMAIN_LENGTH,
        grid_spacing=BASE_DX,
        library_size=library_size,
        krylov_depth=M_KRYLOV,
        quadrature_nodes=N_QUAD,
        shared_seconds=shared_seconds,
        repeated_seconds=repeated_seconds,
        speedup=speedup,
        gradient_relative_difference=relative_difference,
    ))
    @printf(
        "  M=%4d shared=%8.3f ms repeated=%8.3f ms speedup=%7.2fx rel.diff=%.2e\n",
        library_size,
        1e3 * shared_seconds,
        1e3 * repeated_seconds,
        speedup,
        relative_difference,
    )
end

mkpath(OUTPUT_DIR)
CSV.write(
    joinpath(OUTPUT_DIR, "kuramoto_sivashinsky_amortization.csv"),
    DataFrame(rows),
)
