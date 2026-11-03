"""Sparse-direct versus Krylov gradient pilot on the stiff Brusselator CME."""

using LinearAlgebra, SparseArrays, Statistics, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "shared_krylov.jl"))
include(joinpath(ROOT, "src", "sparse_direct_rational.jl"))
include(joinpath(@__DIR__, "optimized_shared_krylov_utils.jl"))
include(joinpath(@__DIR__, "brusselator_error_control.jl"))
BLAS.set_num_threads(1)

const REPEATS_DIRECT = parse(Int, get(ENV, "BRUSS_DIRECT_REPEATS", "3"))
const PILOT_REFERENCE_DEPTH = parse(Int, get(ENV, "BRUSS_DIRECT_REFERENCE_DEPTH", "420"))

relative_error(x, reference) =
    norm(x - reference) / max(norm(reference), eps(Float64))

function median_time(f, repeats)
    f()
    median([(@elapsed f()) * 1e3 for _ in 1:repeats])
end

function main_pilot()
    directions, _ = build_library()
    n = size(first(directions), 1)
    theta_true = rates()
    theta_iterate = theta_true .* [
        j == 1 ? 0.78 : j == 13 ? 1.22 : j == 23 ? 0.74 :
        j == 39 ? 1.28 : 1.0 for j in eachindex(theta_true)
    ]
    Atrue = sum(theta_true[j] .* directions[j] for j in eachindex(directions))
    A = sum(theta_iterate[j] .* directions[j] for j in eachindex(directions))
    initial = zeros(n); initial[lin(3, 2)] = 1.0
    state = deep_action(Atrue, initial, 0.25, PILOT_REFERENCE_DEPTH)
    target = deep_action(Atrue, state, DT, PILOT_REFERENCE_DEPTH)
    prediction = deep_action(A, state, DT, PILOT_REFERENCE_DEPTH)
    upstream = target - prediction

    @printf("building depth-%d projected reference...\n", PILOT_REFERENCE_DEPTH)
    reference_forward = arnoldi_with_residual(
        A, state, PILOT_REFERENCE_DEPTH,
    )
    reference_adjoint = arnoldi_with_residual(
        A', upstream, PILOT_REFERENCE_DEPTH,
    )
    reference_gradient = gradient_from_bases(
        reference_forward, reference_adjoint, directions, DT,
    )
    reference_prediction = action(reference_forward, DT)
    reference_state = action(reference_adjoint, DT)

    cache = SparseDirectionCache(directions)
    direct = () -> sparse_direct_exponential_vjp(
        A, cache, state, DT, upstream,
    )
    direct_result = direct()
    direct_ms = median_time(direct, REPEATS_DIRECT)
    @printf(
        "sparse direct: %.1f ms  grad %.3e  forward %.3e  adjoint %.3e\n",
        direct_ms,
        relative_error(direct_result.parameter_vjp, reference_gradient),
        relative_error(direct_result.prediction, reference_prediction),
        relative_error(direct_result.state_vjp, reference_state),
    )

    for depth in (60, 120, 180), order in (24, 32)
        workspace = OptimizedQuadratureWorkspace(
            directions, n, depth, order,
        )
        nodes, weights = gauss_legendre_01(order)
        evaluate = () -> optimized_linear_loss_and_vjp!(
            workspace, A, state, upstream, DT, nodes, weights;
            m_krylov=depth,
        )
        result = evaluate()
        elapsed = median_time(evaluate, REPEATS_DIRECT)
        @printf(
            "Krylov m=%3d nq=%2d: %.1f ms  grad %.3e  forward %.3e  adjoint %.3e\n",
            depth, order, elapsed,
            relative_error(result.parameter_vjp, reference_gradient),
            relative_error(result.prediction, reference_prediction),
            relative_error(result.state_vjp, reference_state),
        )
    end
end

main_pilot()
