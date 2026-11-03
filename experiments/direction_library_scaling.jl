"""
Scaling audit for pattern-fused compressed-core library contractions.

The forward and adjoint Arnoldi bases are fixed.  We compare:

  1. direct sparse contractions, one `A_j * v_k` per rank/direction;
  2. a precompiled union-pattern kernel followed by one sparse transpose
     multiplication.

Cache construction is reported separately and excluded from VJP timings,
matching its intended use as persistent `SemigroupLayer` metadata.

Environment variables:
  LIBRARY_SCALING_M=10,50,200,1000
  LIBRARY_SCALING_REPEATS=10
  LIBRARY_SCALING_OUTPUT=paper/figures/direction_library_scaling.csv
"""

using CSV, DataFrames
using LinearAlgebra, SparseArrays, Statistics, Printf

include("../src/shared_krylov.jl")

const DIRECTION_COUNTS = parse.(
    Int,
    split(get(ENV, "LIBRARY_SCALING_M", "10,50,200,1000"), ","),
)
const SCALING_REPEATS =
    parse(Int, get(ENV, "LIBRARY_SCALING_REPEATS", "10"))
const SCALING_OUTPUT = get(
    ENV,
    "LIBRARY_SCALING_OUTPUT",
    joinpath(
        @__DIR__,
        "..",
        "paper",
        "figures",
        "direction_library_scaling.csv",
    ),
)

function median_ms(f)
    f()
    samples = Float64[]
    for _ in 1:SCALING_REPEATS
        start = time_ns()
        f()
        push!(samples, (time_ns() - start) / 1e6)
    end
    median(samples)
end

function structured_directions(n, count)
    coordinates = collect(range(-1.0, 1.0; length=n))
    SparseMatrixCSC{Float64,Int}[
        let
            phase = 0.017 * direction
            lower = @. 0.015 + 0.004 * sin(
                phase + 2.1 * coordinates[1:(end - 1)],
            )
            upper = @. 0.012 + 0.003 * cos(
                1.3 * phase - 1.7 * coordinates[2:end],
            )
            diagonal = @. 0.01 * sin(
                phase + (1 + mod(direction, 7)) * coordinates,
            )
            spdiagm(-1 => lower, 0 => diagonal, 1 => upper)
        end for direction in 1:count
    ]
end

BLAS.set_num_threads(1)
n = 400
m_krylov = 40
dt = 0.4
core_rtol = 1e-7
A = spdiagm(
    -1 => fill(0.35, n - 1),
     0 => fill(-1.15, n),
     1 => fill(0.65, n - 1),
)
coordinates = collect(range(-2.0, 2.0; length=n))
p0 = exp.(-0.5 .* coordinates .^ 2)
p0 ./= norm(p0)
upstream = @. sin(1.7 * coordinates) + 0.2 * cos(3.1 * coordinates)

forward_workspace = ArnoldiWorkspace(Float64, n, m_krylov)
adjoint_workspace = ArnoldiWorkspace(Float64, n, m_krylov)
forward_basis = arnoldi_basis!(
    forward_workspace, A, p0; m=m_krylov,
)
adjoint_basis = arnoldi_basis!(
    adjoint_workspace, A', upstream; m=m_krylov,
)

rows = NamedTuple[]
for direction_count in DIRECTION_COUNTS
    dirs = structured_directions(n, direction_count)
    cache_start = time_ns()
    cache = SparseDirectionCache(dirs)
    cache_ms = (time_ns() - cache_start) / 1e6

    direct = _compressed_core_from_bases(
        forward_basis,
        adjoint_basis,
        dirs,
        dt;
        core_rtol=core_rtol,
        return_diagnostics=true,
    )
    fused = _compressed_core_from_bases(
        forward_basis,
        adjoint_basis,
        dirs,
        dt;
        core_rtol=core_rtol,
        direction_cache=cache,
        return_diagnostics=true,
    )
    direct_ms = median_ms() do
        _compressed_core_from_bases(
            forward_basis,
            adjoint_basis,
            dirs,
            dt;
            core_rtol=core_rtol,
        )
    end
    fused_ms = median_ms() do
        _compressed_core_from_bases(
            forward_basis,
            adjoint_basis,
            dirs,
            dt;
            core_rtol=core_rtol,
            direction_cache=cache,
        )
    end
    relative_difference =
        norm(fused.gradient - direct.gradient) /
        max(norm(direct.gradient), 1e-14)
    total_direction_nnz = sum(nnz, dirs)
    union_nnz = length(cache.rows)

    push!(rows, (
        n=n,
        M=direction_count,
        m=m_krylov,
        dt=dt,
        core_rtol=core_rtol,
        core_rank=direct.core_rank,
        union_nnz=union_nnz,
        total_direction_nnz=total_direction_nnz,
        cache_build_ms=cache_ms,
        direct_vjp_ms=direct_ms,
        fused_vjp_ms=fused_ms,
        speedup=direct_ms / fused_ms,
        gradient_relative_difference=relative_difference,
        repeats=SCALING_REPEATS,
    ))
    @printf(
        "M=%4d rank=%2d union=%6d total=%8d direct=%8.3f ms fused=%8.3f ms speedup=%5.2fx diff=%8.1e\n",
        direction_count,
        direct.core_rank,
        union_nnz,
        total_direction_nnz,
        direct_ms,
        fused_ms,
        direct_ms / fused_ms,
        relative_difference,
    )
end

mkpath(dirname(SCALING_OUTPUT))
CSV.write(SCALING_OUTPUT, DataFrame(rows))
println("Saved $(SCALING_OUTPUT)")
