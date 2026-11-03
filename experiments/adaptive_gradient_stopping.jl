"""
Prototype an additive, library-aware adaptive stopping rule for the shared
forward--adjoint Krylov gradient.

At a candidate `(m_f,m_a,n_q)`, the controller evaluates three embedded
changes: refine only the forward space, refine only the adjoint space, and
double the Gauss--Legendre order.  It refines the largest contribution.  A
mixed forward--adjoint difference is included in the stopping estimate, so
the rule remains meaningful when either Arnoldi process terminates exactly.

This is an empirical a posteriori controller, not a rigorous error bound.  The
experiment records the true error against a deep projected-core reference and
the contractive CTMC residual bound separately.

Run from the repository root:

    julia --project=experiments experiments/adaptive_gradient_stopping.jl
"""

using LinearAlgebra, SparseArrays, Printf
using CSV, DataFrames
using FastGaussQuadrature: gausslegendre

include("brusselator_error_control.jl")

const ADAPT_INITIAL_DEPTH = parse(Int, get(ENV, "ADAPT_INITIAL_DEPTH", "4"))
const ADAPT_DEPTH_STEP = parse(Int, get(ENV, "ADAPT_DEPTH_STEP", "4"))
const ADAPT_MAX_DEPTH = parse(Int, get(ENV, "ADAPT_MAX_DEPTH", "180"))
const ADAPT_INITIAL_QUAD = parse(Int, get(ENV, "ADAPT_INITIAL_QUAD", "2"))
const ADAPT_MAX_QUAD = parse(Int, get(ENV, "ADAPT_MAX_QUAD", "32"))
const ADAPT_TOL = parse(Float64, get(ENV, "ADAPT_TOL", "1e-7"))
const ADAPT_SAFETY = parse(Float64, get(ENV, "ADAPT_SAFETY", "4.0"))
const ADAPT_GUARD_SPAN = parse(Int, get(ENV, "ADAPT_GUARD_SPAN", "20"))
const ADAPT_MAX_CONTRACTION = parse(Float64, get(ENV, "ADAPT_MAX_CONTRACTION", "0.9"))
const ADAPT_OUTPUT = get(
    ENV,
    "ADAPT_OUTPUT",
    joinpath(@__DIR__, "..", "benchmark_results", "adaptive_gradient_stopping.csv"),
)

"""Return a prefix of one long Arnoldi factorization, including its defect."""
function basis_prefix(basis, depth)
    depth <= basis.depth || error("requested depth $depth exceeds available depth $(basis.depth)")
    residual_coefficient = if depth < basis.depth
        basis.H[depth + 1, depth]
    else
        basis.residual_coefficient
    end
    residual_vector = if depth < basis.depth
        basis.V[:, depth + 1]
    else
        basis.residual_vector
    end
    (
        V=basis.V[:, 1:depth],
        H=basis.H[1:depth, 1:depth],
        beta=basis.beta,
        depth=depth,
        residual_coefficient=residual_coefficient,
        residual_vector=residual_vector,
    )
end

function quadrature_gradient(forward, adjoint, projected_directions, time, order)
    nodes, weights = gausslegendre(order)
    nodes = (nodes .+ 1) ./ 2
    weights ./= 2
    ef = zeros(forward.depth); ef[1] = forward.beta
    ea = zeros(adjoint.depth); ea[1] = adjoint.beta
    gradient = zeros(length(projected_directions))
    for (s, weight) in zip(nodes, weights)
        f = exp((s * time) .* forward.H) * ef
        a = exp(((1 - s) * time) .* adjoint.H) * ea
        for j in eachindex(projected_directions)
            gradient[j] += time * weight * real(dot(a, projected_directions[j] * f))
        end
    end
    gradient
end

function gradient_at(forward, adjoint, directions, time, order)
    projected = [adjoint.V' * direction * forward.V for direction in directions]
    quadrature_gradient(forward, adjoint, projected, time, order)
end

scaled_maximum(values, scales) = maximum(abs.(values) ./ scales)

"""
Estimate the unresolved tail over two wide, simultaneous refinement windows.

The estimate is deliberately not called a bound: it assumes that the second
window contracts relative to the first.  If no contraction is observed, the
guard is infinite and stopping is forbidden.  This prevents a short local
plateau from being mistaken for convergence.
"""
function contraction_guard(full_forward, full_adjoint, directions, scales,
                           time, mf, ma, nq, span, max_quad)
    mf1 = min(mf + span, full_forward.depth)
    ma1 = min(ma + span, full_adjoint.depth)
    mf2 = min(mf + 2span, full_forward.depth)
    ma2 = min(ma + 2span, full_adjoint.depth)
    guard_q = min(2nq, max_quad)
    f0, a0 = basis_prefix(full_forward, mf), basis_prefix(full_adjoint, ma)
    f1, a1 = basis_prefix(full_forward, mf1), basis_prefix(full_adjoint, ma1)
    f2, a2 = basis_prefix(full_forward, mf2), basis_prefix(full_adjoint, ma2)
    g0 = gradient_at(f0, a0, directions, time, guard_q)
    g1 = gradient_at(f1, a1, directions, time, guard_q)
    g2 = gradient_at(f2, a2, directions, time, guard_q)
    first_change = scaled_maximum(g1 - g0, scales)
    second_change = scaled_maximum(g2 - g1, scales)
    if first_change <= eps()
        ratio = second_change <= eps() ? 0.0 : Inf
    else
        ratio = second_change / first_change
    end
    tail_estimate = if ratio < 1
        first_change + second_change / max(1 - ratio, eps())
    else
        Inf
    end
    (
        first_change=first_change,
        second_change=second_change,
        contraction=ratio,
        tail_estimate=tail_estimate,
    )
end

function embedded_indicators(full_forward, full_adjoint, directions, scales,
                             time, mf, ma, nq, depth_step, max_depth, max_quad)
    mf2 = min(mf + depth_step, max_depth)
    ma2 = min(ma + depth_step, max_depth)
    nq2 = min(2nq, max_quad)
    f = basis_prefix(full_forward, mf)
    a = basis_prefix(full_adjoint, ma)
    f2 = basis_prefix(full_forward, mf2)
    a2 = basis_prefix(full_adjoint, ma2)

    g = gradient_at(f, a, directions, time, nq)
    gf = mf2 == mf ? g : gradient_at(f2, a, directions, time, nq)
    ga = ma2 == ma ? g : gradient_at(f, a2, directions, time, nq)
    gfa = (mf2 == mf && ma2 == ma) ? g :
        gradient_at(f2, a2, directions, time, nq)
    gq = nq2 == nq ? g : gradient_at(f, a, directions, time, nq2)

    df = gf - g
    da = ga - g
    interaction = gfa - gf - ga + g
    dq = gq - g
    eta_f = scaled_maximum(df, scales)
    eta_a = scaled_maximum(da, scales)
    eta_interaction = scaled_maximum(interaction, scales)
    eta_q = scaled_maximum(dq, scales)
    eta_total = eta_f + eta_a + eta_interaction + eta_q
    (
        gradient=g,
        eta_forward=eta_f,
        eta_adjoint=eta_a,
        eta_interaction=eta_interaction,
        eta_quadrature=eta_q,
        eta_total=eta_total,
        refined_gradient=gfa + dq,
        next_forward=mf2,
        next_adjoint=ma2,
        next_quadrature=nq2,
    )
end

function build_problem()
    directions, labels = build_library()
    theta_true = rates()
    perturbation = [j == 1 ? 0.78 : j == 13 ? 1.22 :
                    j == 23 ? 0.74 : j == 39 ? 1.28 : 1.0
                    for j in eachindex(theta_true)]
    theta_iterate = theta_true .* perturbation
    Atrue = sum(theta_true[j] .* directions[j] for j in eachindex(directions))
    A = sum(theta_iterate[j] .* directions[j] for j in eachindex(directions))
    n = size(A, 1)
    initial = zeros(n); initial[lin(3, 2)] = 1.0
    state = deep_action(Atrue, initial, 0.25, REFERENCE_DEPTH)
    target = deep_action(Atrue, state, DT, REFERENCE_DEPTH)
    prediction = deep_action(A, state, DT, REFERENCE_DEPTH)
    terminal = target - prediction
    A, directions, labels, state, terminal
end

function main_adaptive()
    A, directions, labels, state, terminal = build_problem()
    available_depth = min(ADAPT_MAX_DEPTH + 2ADAPT_GUARD_SPAN, size(A, 1) - 1)
    full_forward = arnoldi_with_residual(A, state, available_depth)
    full_adjoint = arnoldi_with_residual(A', terminal, available_depth)

    reference_forward = arnoldi_with_residual(A, state, REFERENCE_DEPTH)
    reference_adjoint = arnoldi_with_residual(A', terminal, REFERENCE_DEPTH)
    reference = gradient_from_bases(reference_forward, reference_adjoint, directions, DT)
    scales = max.(norm.(directions), eps())

    mf = min(ADAPT_INITIAL_DEPTH, ADAPT_MAX_DEPTH)
    ma = min(ADAPT_INITIAL_DEPTH, ADAPT_MAX_DEPTH)
    nq = ADAPT_INITIAL_QUAD
    rows = NamedTuple[]
    stopped = false
    for iteration in 1:200
        indicator = embedded_indicators(
            full_forward, full_adjoint, directions, scales, DT,
            mf, ma, nq, ADAPT_DEPTH_STEP, ADAPT_MAX_DEPTH, ADAPT_MAX_QUAD,
        )
        true_error = scaled_maximum(indicator.gradient - reference, scales)
        refined_error = scaled_maximum(indicator.refined_gradient - reference, scales)
        guard = contraction_guard(
            full_forward, full_adjoint, directions, scales, DT,
            mf, ma, nq, ADAPT_GUARD_SPAN, ADAPT_MAX_QUAD,
        )
        push!(rows, (
            iteration=iteration,
            forward_depth=mf,
            adjoint_depth=ma,
            quadrature_order=nq,
            eta_forward=indicator.eta_forward,
            eta_adjoint=indicator.eta_adjoint,
            eta_interaction=indicator.eta_interaction,
            eta_quadrature=indicator.eta_quadrature,
            eta_total=indicator.eta_total,
            safety_factor=ADAPT_SAFETY,
            guarded_estimate=ADAPT_SAFETY * indicator.eta_total,
            guard_first_change=guard.first_change,
            guard_second_change=guard.second_change,
            guard_contraction=guard.contraction,
            guard_tail_estimate=guard.tail_estimate,
            true_scaled_error=true_error,
            refined_true_scaled_error=refined_error,
            tolerance=ADAPT_TOL,
        ))
        @printf(
            "it=%2d mf=%3d ma=%3d q=%2d eta=(%.2e, %.2e, %.2e, %.2e) total=%.2e true=%.2e\n",
            iteration, mf, ma, nq, indicator.eta_forward,
            indicator.eta_adjoint, indicator.eta_interaction,
            indicator.eta_quadrature, indicator.eta_total, true_error,
        )

        # Require both the decomposed estimate and the simultaneous look-ahead
        # change to pass.  The latter catches cancellation among components.
        lookahead = scaled_maximum(indicator.refined_gradient - indicator.gradient, scales)
        if ADAPT_SAFETY * indicator.eta_total <= ADAPT_TOL &&
           ADAPT_SAFETY * lookahead <= ADAPT_TOL &&
           guard.contraction <= ADAPT_MAX_CONTRACTION &&
           guard.tail_estimate <= ADAPT_TOL
            stopped = true
            break
        end

        candidates = (
            indicator.eta_forward,
            indicator.eta_adjoint,
            indicator.eta_quadrature,
        )
        order = sortperm(collect(candidates); rev=true)
        advanced = false
        for choice in order
            if choice == 1 && mf < ADAPT_MAX_DEPTH
                mf = indicator.next_forward; advanced = true; break
            elseif choice == 2 && ma < ADAPT_MAX_DEPTH
                ma = indicator.next_adjoint; advanced = true; break
            elseif choice == 3 && nq < ADAPT_MAX_QUAD
                nq = indicator.next_quadrature; advanced = true; break
            end
        end
        advanced || break
    end

    data = DataFrame(rows)
    mkpath(dirname(ADAPT_OUTPUT))
    CSV.write(ADAPT_OUTPUT, data)
    final_error = data.true_scaled_error[end]
    @printf(
        "stopped=%s at (mf,ma,q)=(%d,%d,%d): estimate %.3e, true error %.3e, target %.3e\n",
        stopped, mf, ma, nq, data.eta_total[end], final_error, ADAPT_TOL,
    )
    println("saved ", ADAPT_OUTPUT)
    stopped || error("controller exhausted its refinement limits")
    final_error <= ADAPT_TOL || error(
        "embedded controller stopped too early: true error $final_error > $(ADAPT_TOL)",
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_adaptive()
