"""
Error-control audit on a stiff Brusselator CME snapshot.

The target snapshot is generated with the true rates.  Gradients are evaluated
at a perturbed-rate iterate on the longest observation interval.  For equal
forward/adjoint Krylov depths, the script compares

  * the true library-coordinate gradient error (against a deep-core reference),
  * Kandolf et al.'s multiplicative estimator (59), converted to the same
    normalized library-coordinate scale, and
  * the additive CTMC residual bound in the paired 1/infinity norms.

Run from the repository root:

    julia --project=experiments experiments/brusselator_error_control.jl
"""

using LinearAlgebra, SparseArrays, Printf
using CSV, DataFrames
using FastGaussQuadrature: gausslegendre

const XMAX = parse(Int, get(ENV, "BRUSS_CERT_XMAX", "40"))
const YMAX = parse(Int, get(ENV, "BRUSS_CERT_YMAX", "40"))
const DT = parse(Float64, get(ENV, "BRUSS_CERT_DT", "0.5"))
const REFERENCE_DEPTH = parse(Int, get(ENV, "BRUSS_CERT_REFERENCE_DEPTH", "420"))
const MAX_DEPTH = parse(Int, get(ENV, "BRUSS_CERT_MAX_DEPTH", "60"))
const DEPTH_STEP = parse(Int, get(ENV, "BRUSS_CERT_DEPTH_STEP", "2"))
const QUAD_ORDER = parse(Int, get(ENV, "BRUSS_CERT_QUAD_ORDER", "24"))
const OUTPUT = get(
    ENV,
    "BRUSS_CERT_OUTPUT",
    joinpath(@__DIR__, "..", "paper", "figures", "brusselator_error_control.csv"),
)

lin(x, y) = x * (YMAX + 1) + y + 1
combination(x, a) = a == 0 ? 1.0 : x < a ? 0.0 : prod(x-k for k in 0:a-1) / factorial(a)

function build_library()
    channels = [(1, 0), (-1, 0), (-1, 1), (1, -1)]
    monomials = [(a, degree-a) for degree in 0:3 for a in 0:degree]
    n = (XMAX + 1) * (YMAX + 1)
    directions = SparseMatrixCSC{Float64,Int}[]
    labels = String[]
    for (dx, dy) in channels, (a, b) in monomials
        rows = Int[]; columns = Int[]; values = Float64[]
        for x in 0:XMAX, y in 0:YMAX
            x < a && continue
            y < b && continue
            xn, yn = x + dx, y + dy
            (0 <= xn <= XMAX && 0 <= yn <= YMAX) || continue
            propensity = combination(x, a) * combination(y, b)
            iszero(propensity) && continue
            column, destination = lin(x, y), lin(xn, yn)
            push!(rows, destination, column)
            push!(columns, column, column)
            push!(values, propensity, -propensity)
        end
        push!(directions, sparse(rows, columns, values, n, n))
        push!(labels, "($dx,$dy) C(x,$a) C(y,$b)")
    end
    directions, labels
end

function rates(A_context=1.0, B_context=3.0)
    theta = zeros(40)
    theta[1] = A_context
    theta[13] = 1.0
    theta[23] = B_context
    theta[39] = 1.0
    theta
end

function arnoldi_with_residual(A, start, depth; tolerance=1e-14)
    n = length(start)
    requested = min(depth, n-1)
    V = zeros(eltype(start), n, requested + 1)
    H = zeros(eltype(start), requested + 1, requested)
    beta = norm(start)
    beta > tolerance || error("Arnoldi start must be nonzero")
    V[:, 1] .= start ./ beta
    used = requested
    for column in 1:requested
        work = A * view(V, :, column)
        # Two-pass modified Gram--Schmidt keeps the deep reference reliable.
        for _ in 1:2
            for row in 1:column
                coefficient = dot(view(V, :, row), work)
                H[row, column] += coefficient
                work .-= coefficient .* view(V, :, row)
            end
        end
        H[column + 1, column] = norm(work)
        if H[column + 1, column] <= tolerance
            used = column
            break
        end
        V[:, column + 1] .= work ./ H[column + 1, column]
    end
    (
        V=V[:, 1:used], H=H[1:used, 1:used], beta=beta, depth=used,
        residual_coefficient=H[used + 1, used],
        residual_vector=V[:, used + 1],
    )
end

function action(basis, time)
    e1 = zeros(basis.depth); e1[1] = 1.0
    basis.beta .* (basis.V * (exp(time .* basis.H) * e1))
end

function projected_core(forward, adjoint, time)
    mf, ma = forward.depth, adjoint.depth
    block = zeros(Float64, ma + mf, ma + mf)
    block[1:ma, 1:ma] .= adjoint.H
    block[(ma + 1):end, (ma + 1):end] .= forward.H'
    block[1, ma + 1] = adjoint.beta * forward.beta
    exp(time .* block)[1:ma, (ma + 1):end]
end

function gradient_from_bases(forward, adjoint, directions, time)
    core = projected_core(forward, adjoint, time)
    [real(dot(core, adjoint.V' * direction * forward.V)) for direction in directions]
end

function kandolf_59(forward, adjoint, time)
    m = forward.depth
    adjoint.depth == m || error("Equation (59) uses equal depths")
    e1e1 = zeros(m, m); e1e1[1, 1] = 1.0
    augmented = zeros(Float64, 4m, 4m)
    augmented[1:m, 1:m] .= time .* adjoint.H
    augmented[1:m, (m + 1):(2m)] .= -e1e1
    augmented[(m + 1):(2m), (m + 1):(2m)] .= time .* forward.H'
    augmented[(m + 1):(2m), (2m + 1):(3m)] .= Matrix{Float64}(I, m, m)
    augmented[(2m + 1):(3m), (3m + 1):(4m)] .= Matrix{Float64}(I, m, m)
    scale = forward.beta * adjoint.beta *
        abs(time * forward.residual_coefficient) *
        abs(time * adjoint.residual_coefficient)
    scale * abs(exp(augmented)[m, 3m + 1])
end

function residual_envelope(basis, time, vector_norm)
    time <= 0 && return 0.0
    nodes, weights = gausslegendre(QUAD_ORDER)
    nodes = (nodes .+ 1) .* (time / 2)
    weights .*= time / 2
    e1 = zeros(basis.depth); e1[1] = 1.0
    coefficient = basis.beta * abs(basis.residual_coefficient) * vector_norm(basis.residual_vector)
    coefficient * sum(weights .* [abs((exp(t .* basis.H) * e1)[end]) for t in nodes])
end

function induced_one_norm(A)
    maximum(vec(sum(abs, A; dims=1)))
end

function additive_bounds(forward, adjoint, directions, time)
    nodes, weights = gausslegendre(QUAD_ORDER)
    nodes = (nodes .+ 1) ./ 2
    weights ./= 2
    result = zeros(length(directions))
    direction_norms = induced_one_norm.(directions)
    for (s, weight) in zip(nodes, weights)
        t = time * s
        tau = time - t
        ftilde = action(forward, t)
        lambdatilde = action(adjoint, tau)
        delta_f = residual_envelope(forward, t, x -> norm(x, 1))
        delta_a = residual_envelope(adjoint, tau, x -> norm(x, Inf))
        for j in eachindex(directions)
            direction = directions[j]
            result[j] += time * weight * (
                delta_a * norm(direction * ftilde, 1) +
                delta_f * norm(direction' * lambdatilde, Inf) +
                direction_norms[j] * delta_f * delta_a
            )
        end
    end
    result
end

function deep_action(A, start, time, depth)
    action(arnoldi_with_residual(A, start, depth), time)
end

function main()
    directions, labels = build_library()
    n = size(first(directions), 1)
    theta_true = rates()
    theta_iterate = theta_true .* [j == 1 ? 0.78 : j == 13 ? 1.22 : j == 23 ? 0.74 : j == 39 ? 1.28 : 1.0 for j in eachindex(theta_true)]
    Atrue = sum(theta_true[j] .* directions[j] for j in eachindex(directions))
    A = sum(theta_iterate[j] .* directions[j] for j in eachindex(directions))

    initial = zeros(n); initial[lin(3, 2)] = 1.0
    # Use the final, longest interval in the paper's 0.05, 0.2, 0.5 schedule.
    state = deep_action(Atrue, initial, 0.25, REFERENCE_DEPTH)
    target = deep_action(Atrue, state, DT, REFERENCE_DEPTH)
    prediction = deep_action(A, state, DT, REFERENCE_DEPTH)
    terminal = target - prediction

    reference_forward = arnoldi_with_residual(A, state, REFERENCE_DEPTH)
    reference_adjoint = arnoldi_with_residual(A', terminal, REFERENCE_DEPTH)
    reference_gradient = gradient_from_bases(reference_forward, reference_adjoint, directions, DT)

    # Verify that increasing the reference depth does not change the result.
    check_depth = min(REFERENCE_DEPTH + 20, n - 1)
    check_forward = arnoldi_with_residual(A, state, check_depth)
    check_adjoint = arnoldi_with_residual(A', terminal, check_depth)
    check_gradient = gradient_from_bases(check_forward, check_adjoint, directions, DT)
    reference_disagreement = maximum(abs.(check_gradient - reference_gradient)) /
        max(maximum(abs.(check_gradient)), eps())
    reference_disagreement < 1e-9 || error("reference depth is not converged: $reference_disagreement")

    frobenius_norms = norm.(directions)
    rows = NamedTuple[]
    previous_gradient = nothing
    for depth in 2:DEPTH_STEP:MAX_DEPTH
        forward = arnoldi_with_residual(A, state, depth)
        adjoint = arnoldi_with_residual(A', terminal, depth)
        gradient = gradient_from_bases(forward, adjoint, directions, DT)
        coordinate_errors = abs.(gradient - reference_gradient)
        normalized_errors = coordinate_errors ./ frobenius_norms
        additive = additive_bounds(forward, adjoint, directions, DT)
        normalized_additive = additive ./ frobenius_norms
        eta59 = kandolf_59(forward, adjoint, DT)
        successive = previous_gradient === nothing ? NaN :
            maximum(abs.(gradient - previous_gradient) ./ frobenius_norms)
        worst = argmax(normalized_errors)
        active = (1, 13, 23, 39)
        active_worst = active[argmax(normalized_errors[collect(active)])]
        push!(rows, (
            depth=depth,
            true_max_normalized_coordinate_error=maximum(normalized_errors),
            additive_max_normalized_coordinate_bound=maximum(normalized_additive),
            additive_bound_at_worst_error_coordinate=normalized_additive[worst],
            true_active_max_normalized_coordinate_error=maximum(normalized_errors[collect(active)]),
            additive_active_max_normalized_coordinate_bound=maximum(normalized_additive[collect(active)]),
            kandolf_59_frobenius_estimate=eta59,
            successive_max_normalized_coordinate_difference=successive,
            forward_residual_coefficient=forward.residual_coefficient,
            adjoint_residual_coefficient=adjoint.residual_coefficient,
            worst_direction=labels[worst],
            worst_direction_index=worst,
            worst_active_direction_index=active_worst,
            state_dimension=n,
            interval_length=DT,
            reference_depth=REFERENCE_DEPTH,
            reference_relative_disagreement=reference_disagreement,
        ))
        @printf(
            "m=%3d true=%9.2e add=%9.2e K59=%9.2e f-res=%9.2e a-res=%9.2e dir=%d\n",
            depth, maximum(normalized_errors), maximum(normalized_additive), eta59,
            forward.residual_coefficient, adjoint.residual_coefficient, worst,
        )
        previous_gradient = gradient
    end
    data = DataFrame(rows)
    all(data.additive_max_normalized_coordinate_bound .+ 1e-11 .>=
        data.true_max_normalized_coordinate_error) ||
        error("additive bound fell below the true normalized coordinate error")
    mkpath(dirname(OUTPUT))
    CSV.write(OUTPUT, data)
    @printf("reference relative disagreement: %.3e\n", reference_disagreement)
    println("saved ", OUTPUT)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
