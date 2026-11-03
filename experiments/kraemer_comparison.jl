"""
Matched shared-Krylov/Kraemer et al. benchmark.

This script evaluates the paper's fused two-basis quadrature pullback on the
same one-dimensional Fokker--Planck generator, objective, time steps,
precisions, and Krylov dimensions as `kraemer_comparison.py`.

The timed VJP returns cotangents for both the structured parameters and the
input state. The forward Arnoldi basis is reused inside the pullback, matching
Algorithm 1 rather than timing the unfused public rrule.

Run:

    julia --project=experiments experiments/kraemer_comparison.jl

Optional environment variables:

    KRAEMER_M_VALUES=5,10,20,30,40,60
    KRAEMER_DT_VALUES=0.3,1.0
    KRAEMER_REPEATS=10
    KRAEMER_CORE_RTOL=1e-7
"""

using CSV, DataFrames
using LinearAlgebra, SparseArrays, Printf, Statistics

include("../src/fpe.jl")
include("../src/shared_krylov.jl")

const KRAEMER_M_VALUES = parse.(
    Int, split(get(ENV, "KRAEMER_M_VALUES", "5,10,20,30,40,60"), ","),
)
const KRAEMER_DT_VALUES = parse.(
    Float64, split(get(ENV, "KRAEMER_DT_VALUES", "0.3,1.0"), ","),
)
const KRAEMER_REPEATS = parse(Int, get(ENV, "KRAEMER_REPEATS", "10"))
const KRAEMER_CORE_RTOL =
    parse(Float64, get(ENV, "KRAEMER_CORE_RTOL", "1e-7"))

relative_error(value, reference) =
    norm(Float64.(value) - Float64.(reference)) / max(norm(reference), 1e-14)

function exact_reference(A0, dirs, θ_true, θ_eval, p0, dt)
    A_true = Matrix(A0 + sum(θ_true[j] * dirs[j] for j in eachindex(dirs)))
    A_eval = Matrix(A0 + sum(θ_eval[j] * dirs[j] for j in eachindex(dirs)))
    target = exp(A_true * dt) * p0
    prediction = exp(A_eval * dt) * p0
    upstream = prediction - target
    n = length(p0)
    Z = zeros(Float64, n, n)
    parameter_vjp = [
        dot(
            upstream,
            exp([A_eval * dt Matrix(dirs[j]) * dt; Z A_eval * dt])[1:n, n+1:2n] * p0,
        )
        for j in eachindex(dirs)
    ]
    state_vjp = exp(A_eval' * dt) * upstream
    (; target, prediction, parameter_vjp, state_vjp)
end

function fused_loss_and_vjp(
    A0, dirs, θ, p0, target, dt; m_krylov, n_quad,
)
    A = A0 + sum(θ[j] * dirs[j] for j in eachindex(dirs))
    T = eltype(p0)
    RT = real_scalar_type(T)
    raw_nodes, raw_weights = n_quad == 5 ?
        (GL_NODES_5, GL_WEIGHTS_5) : (GL_NODES_8, GL_WEIGHTS_8)
    nodes = RT.(raw_nodes)
    weights = RT.(raw_weights)

    ws_f = ArnoldiWorkspace(T, length(p0), m_krylov)
    ws_a = ArnoldiWorkspace(T, length(p0), m_krylov)
    ks_f = arnoldi_basis!(ws_f, A, p0; m=m_krylov)
    prediction = _eval_krylov_batch(ks_f, RT[dt], ks_f.beta)[:, 1]
    upstream = prediction - target
    loss = RT(0.5) * sum(abs2, upstream)

    ks_a = arnoldi_basis!(ws_a, A', upstream; m=m_krylov)
    state_vjp = _eval_krylov_batch(ks_a, RT[dt], ks_a.beta)[:, 1]
    forward_path =
        _eval_krylov_batch(ks_f, nodes .* dt, ks_f.beta)
    adjoint_path =
        _eval_krylov_batch(ks_a, (one(RT) .- nodes) .* dt, ks_a.beta)

    buffer = zeros(T, length(p0))
    parameter_vjp = zeros(RT, length(dirs))
    for j in eachindex(dirs)
        value = zero(T)
        @inbounds for q in eachindex(nodes)
            mul!(buffer, dirs[j], view(forward_path, :, q))
            value += weights[q] * dot(view(adjoint_path, :, q), buffer)
        end
        parameter_vjp[j] = real(dt * value)
    end
    (; loss, prediction, parameter_vjp, state_vjp)
end

function fused_loss_and_core_vjp(
    A0, dirs, θ, p0, target, dt;
    m_krylov,
    core_rtol,
    direction_cache,
)
    A = A0 + sum(θ[j] * dirs[j] for j in eachindex(dirs))
    T = eltype(p0)
    RT = real_scalar_type(T)
    ws_f = ArnoldiWorkspace(T, length(p0), m_krylov)
    ws_a = ArnoldiWorkspace(T, length(p0), m_krylov)

    ks_f = arnoldi_basis!(ws_f, A, p0; m=m_krylov)
    prediction = _eval_krylov_batch(ks_f, RT[dt], ks_f.beta)[:, 1]
    upstream = prediction - target
    loss = RT(0.5) * sum(abs2, upstream)

    ks_a = arnoldi_basis!(ws_a, A', upstream; m=m_krylov)
    state_vjp = _eval_krylov_batch(ks_a, RT[dt], ks_a.beta)[:, 1]
    diagnostics = _compressed_core_from_bases(
        ks_f,
        ks_a,
        dirs,
        dt;
        core_rtol=core_rtol,
        direction_cache=direction_cache,
        return_diagnostics=true,
    )
    (
        loss=loss,
        prediction=prediction,
        parameter_vjp=diagnostics.gradient,
        state_vjp=state_vjp,
        core_rank=diagnostics.core_rank,
        core_full_rank=diagnostics.core_full_rank,
        core_relative_tail=diagnostics.core_relative_tail,
    )
end

function forward_only(A0, dirs, θ, p0, dt; m_krylov)
    A = A0 + sum(θ[j] * dirs[j] for j in eachindex(dirs))
    workspace = ArnoldiWorkspace(eltype(p0), length(p0), m_krylov)
    basis = arnoldi_basis!(workspace, A, p0; m=m_krylov)
    _eval_krylov_batch(basis, [dt], basis.beta)[:, 1]
end

function benchmark_configuration(
    A0, dirs, θ_eval, p0, reference, dt, m, precision, method,
)
    T = precision == :float64 ? Float64 : Float32
    A0p = convert(SparseMatrixCSC{T,Int}, A0)
    dirsp = SparseMatrixCSC{T,Int}[
        convert(SparseMatrixCSC{T,Int}, direction) for direction in dirs
    ]
    θp = T.(θ_eval)
    p0p = T.(p0)
    targetp = T.(reference.target)
    dtp = T(dt)
    n_quad = 8
    direction_cache = method === :compressed_core ?
        SparseDirectionCache(dirsp) : nothing

    # Compile and warm allocation paths before timing.
    prediction = forward_only(
        A0p, dirsp, θp, p0p, dtp; m_krylov=m,
    )
    result = if method === :compressed_core
        fused_loss_and_core_vjp(
            A0p, dirsp, θp, p0p, targetp, dtp;
            m_krylov=m,
            core_rtol=KRAEMER_CORE_RTOL,
            direction_cache=direction_cache,
        )
    else
        fused_loss_and_vjp(
            A0p, dirsp, θp, p0p, targetp, dtp;
            m_krylov=m, n_quad=n_quad,
        )
    end
    GC.gc()

    forward_times = Float64[]
    vjp_times = Float64[]
    for _ in 1:KRAEMER_REPEATS
        start = time_ns()
        prediction = forward_only(
            A0p, dirsp, θp, p0p, dtp; m_krylov=m,
        )
        push!(forward_times, (time_ns() - start) / 1e6)

        start = time_ns()
        result = if method === :compressed_core
            fused_loss_and_core_vjp(
                A0p, dirsp, θp, p0p, targetp, dtp;
                m_krylov=m,
                core_rtol=KRAEMER_CORE_RTOL,
                direction_cache=direction_cache,
            )
        else
            fused_loss_and_vjp(
                A0p, dirsp, θp, p0p, targetp, dtp;
                m_krylov=m, n_quad=n_quad,
            )
        end
        push!(vjp_times, (time_ns() - start) / 1e6)
    end

    (
        backend=method === :compressed_core ?
            "Compressed_Core_Julia" : "Shared_Krylov_Julia",
        system="FPE",
        n=length(p0),
        M=length(θ_eval),
        dt=dt,
        m=m,
        n_quad=method === :compressed_core ? missing : n_quad,
        core_rtol=method === :compressed_core ?
            KRAEMER_CORE_RTOL : NaN,
        core_rank=method === :compressed_core ?
            result.core_rank : missing,
        core_full_rank=method === :compressed_core ?
            result.core_full_rank : missing,
        core_relative_tail=method === :compressed_core ?
            result.core_relative_tail : NaN,
        precision=String(precision),
        reorthogonalization="single_MGS_pass",
        gradient_semantics=method === :compressed_core ?
            "compressed_projected_exact_integral_VJP" :
            "quadrature_VJP_of_exact_exponential",
        forward_relerr=relative_error(
            result.prediction, reference.prediction,
        ),
        parameter_vjp_relerr=relative_error(
            result.parameter_vjp, reference.parameter_vjp,
        ),
        state_vjp_relerr=relative_error(
            result.state_vjp, reference.state_vjp,
        ),
        forward_ms=median(forward_times),
        loss_and_full_vjp_ms=median(vjp_times),
        compile_seconds=NaN,
        repeats=KRAEMER_REPEATS,
        jax_version="",
        matfree_version="",
        scipy_version="",
        platform="$(Sys.KERNEL)-$(Sys.ARCH)-cpu",
    )
end

BLAS.set_num_threads(1)
n = 80
M = 10
xs = collect(range(-2.0, 2.0; length=n))
drift_fns = [let k=j; x -> (x / 2.0)^k end for j in 1:M]
A0, dirs = fpe_library_1d(xs, drift_fns; D=0.5)
θ_true = zeros(M)
θ_true[1] = -4.0
θ_true[3] = 4.0
θ_eval = θ_true + 0.08 .* collect(range(-1.0, 1.0; length=M))
p0 = exp.(-0.5 .* xs .^ 2)
p0 ./= sum(p0)

rows = NamedTuple[]
for dt in KRAEMER_DT_VALUES
    reference = exact_reference(A0, dirs, θ_true, θ_eval, p0, dt)
    for precision in (:float64, :float32), m in KRAEMER_M_VALUES,
        method in (:quadrature, :compressed_core)
        row = benchmark_configuration(
            A0, dirs, θ_eval, p0, reference, dt, m, precision, method,
        )
        push!(rows, row)
        @printf(
            "%-10s %-7s dt=%3.1f m=%2d forward=%9.2e θ=%9.2e state=%9.2e VJP=%7.3f ms\n",
            method === :compressed_core ? "Core" : "Shared",
            row.precision, dt, m, row.forward_relerr,
            row.parameter_vjp_relerr, row.state_vjp_relerr,
            row.loss_and_full_vjp_ms,
        )
    end
end

output = joinpath(@__DIR__, "..", "paper", "figures", "kraemer_shared_krylov.csv")
CSV.write(output, DataFrame(rows))
println("Saved $output")
