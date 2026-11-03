"""
Large explicit-sparse Lindblad forward/VJP scaling benchmark.

Unlike `lindblad_scaling.jl`, which trains small dense models, this benchmark
isolates the numerical layer and grows the vectorized density dimension to
4^n. Dense finite-difference checks are retained for n ≤ 4.

Environment variables:
  LINDBLAD_SPARSE_NS       comma-separated qubit counts (default: 2,...,9)
  LINDBLAD_SPARSE_M        Krylov dimension (default: 40)
  LINDBLAD_SPARSE_NQ       quadrature nodes, 5 or 8 (default: 8)
  LINDBLAD_SPARSE_DT       propagation interval (default: 1.0)
  LINDBLAD_SPARSE_TRIALS   timing trials after warmup (default: 3)
  LINDBLAD_SPARSE_OUTPUT   output CSV path
"""

using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames

include("lindblad_structured.jl")

const SPARSE_NS = parse.(
    Int,
    split(get(ENV, "LINDBLAD_SPARSE_NS", "2,3,4,5,6,7,8,9"), ","),
)
const SPARSE_M = parse(Int, get(ENV, "LINDBLAD_SPARSE_M", "40"))
const SPARSE_NQ = parse(Int, get(ENV, "LINDBLAD_SPARSE_NQ", "8"))
const SPARSE_DT = parse(Float64, get(ENV, "LINDBLAD_SPARSE_DT", "1.0"))
const SPARSE_TRIALS = parse(Int, get(ENV, "LINDBLAD_SPARSE_TRIALS", "3"))
const SPARSE_OUTPUT = get(
    ENV,
    "LINDBLAD_SPARSE_OUTPUT",
    joinpath(@__DIR__, "..", "paper", "figures", "lindblad_sparse_scaling.csv"),
)

SPARSE_NQ in (5, 8) || error("LINDBLAD_SPARSE_NQ must be 5 or 8")

function perturbed_params(c, n)
    θ = true_params(c, n)
    max.(θ .* (1 .+ 0.2 .* sin.(1:length(θ))) .+ 0.03, 1e-4)
end

relative_error(x, ref) = norm(x - ref) / max(norm(ref), 1e-14)

function median_time_ms(f, n_trials)
    f() # compile and warm caches
    samples = Float64[]
    for _ in 1:n_trials
        push!(samples, 1e3 * @elapsed f())
    end
    median(samples)
end

function fd_parameter_vjp(A0, dirs, θ, p0, target, dt, m_ref; h=2e-6)
    function loss(θ_arg)
        A = A0 + sum(θ_arg[j] * dirs[j] for j in eachindex(θ_arg))
        pred = expv(dt, A, p0; m=m_ref)
        0.5 * norm(pred - target)^2
    end

    grad = zeros(length(θ))
    for j in eachindex(θ)
        θp = copy(θ)
        θm = copy(θ)
        θp[j] += h
        θm[j] -= h
        grad[j] = (loss(θp) - loss(θm)) / (2h)
    end
    grad
end

rows = NamedTuple[]
println("Explicit-sparse Lindblad matrix-exponential scaling")
@printf "  dt=%.2f  Krylov m=%d  quadrature=%d  trials=%d\n" SPARSE_DT SPARSE_M SPARSE_NQ SPARSE_TRIALS

for n in SPARSE_NS
    build_s = @elapsed A0, dirs = build_ising_library(n; sparse_operators=true)
    @assert issparse(A0)
    @assert all(issparse, dirs)

    N = 4^n
    M = length(dirs)
    m = min(SPARSE_M, N - 1)
    θ_true = true_params(0.5, n)
    θ_eval = perturbed_params(0.5, n)
    A_true = A0 + sum(θ_true[j] * dirs[j] for j in 1:M)
    A_eval = A0 + sum(θ_eval[j] * dirs[j] for j in 1:M)
    p0 = initial_density_vec(n)

    # A larger reference Krylov space is affordable only for the small systems
    # where a finite-difference gradient check is reported.
    m_ref = n <= 4 ? min(max(2m, 80), N - 1) : m
    target = expv(SPARSE_DT, A_true, p0; m=m_ref)
    pred = expv(SPARSE_DT, A_eval, p0; m=m)
    upstream = pred - target

    forward_ms = median_time_ms(
        () -> expv(SPARSE_DT, A_eval, p0; m=m),
        SPARSE_TRIALS,
    )
    vjp_ms = median_time_ms(
        () -> krylov_frechet_vjp(
            A_eval, dirs, p0, SPARSE_DT, upstream;
            m_krylov=m,
            n_quad=SPARSE_NQ,
        ),
        SPARSE_TRIALS,
    )
    grad = krylov_frechet_vjp(
        A_eval, dirs, p0, SPARSE_DT, upstream;
        m_krylov=m,
        n_quad=SPARSE_NQ,
    )

    grad_relerr = NaN
    if n <= 4
        grad_fd = fd_parameter_vjp(
            A0, dirs, θ_eval, p0, target, SPARSE_DT, m_ref,
        )
        grad_relerr = relative_error(grad, grad_fd)
    end

    operator_bytes = Base.summarysize(A0) +
                     sum(Base.summarysize, dirs; init=0)
    # Two Arnoldi bases, their work vectors, the two quadrature path batches,
    # and one sparse-direction work vector. This is a transparent dominant-term
    # estimate rather than a process-level peak-RSS measurement.
    workspace_bytes = sizeof(ComplexF64) * (
        2N * m + 2m^2 + 2N + 2N * SPARSE_NQ + N
    )
    dense_library_bytes = sizeof(ComplexF64) * (M + 1) * Int128(N)^2

    row = (
        n=n,
        hilbert_dim=2^n,
        liouville_dim=N,
        M=M,
        dt=SPARSE_DT,
        krylov_m=m,
        n_quad=SPARSE_NQ,
        timing_trials=SPARSE_TRIALS,
        nnz_generator=nnz(A_eval),
        nnz_directions=sum(nnz, dirs; init=0),
        build_seconds=build_s,
        forward_ms=forward_ms,
        parameter_vjp_ms=vjp_ms,
        parameter_vjp_relerr=grad_relerr,
        sparse_operator_mb=operator_bytes / 2.0^20,
        krylov_workspace_mb=workspace_bytes / 2.0^20,
        dense_library_gb=Float64(dense_library_bytes) / 2.0^30,
    )
    push!(rows, row)
    @printf(
        "  n=%d N=%7d M=%2d nnz(A)=%8d sparse=%8.1f MB forward=%8.2f ms VJP=%8.2f ms err=%9.2e\n",
        row.n, row.liouville_dim, row.M, row.nnz_generator,
        row.sparse_operator_mb, row.forward_ms, row.parameter_vjp_ms,
        row.parameter_vjp_relerr,
    )
    flush(stdout)
end

mkpath(dirname(SPARSE_OUTPUT))
CSV.write(SPARSE_OUTPUT, DataFrame(rows))
println("\nSaved $(SPARSE_OUTPUT)")
