"""Trajectory-objective benchmark for partial Lindblad observations.

Compare three mathematically equivalent gradient organizations:

1. independent endpoint Frechet cores from the initial state;
2. a chained discrete adjoint with one projected core per interval;
3. the same chained adjoint with endpoint-reusing Lobatto quadrature.

All methods use the same Arnoldi depth and partial-observation residuals.  The
full chained core is retained as the matched-basis accuracy reference, while
the timed core uses SVD compression.
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, SparseDirectionCache,
    arnoldi_basis!, _eval_krylov_batch, _compressed_core_from_bases
using ExponentialUtilities: expv
using FastGaussQuadrature: gausslobatto
using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames

include("../src/lindblad.jl")

const Q = parse(Int, get(ENV, "LINDBLAD_TRAJ_QUBITS", "4"))
const HORIZON = parse(Float64, get(ENV, "LINDBLAD_TRAJ_HORIZON", "3.0"))
const COUNTS = parse.(Int, split(get(ENV, "LINDBLAD_TRAJ_COUNTS", "1,2,4,8,16,32,64"), ","))
const DEPTH = parse(Int, get(ENV, "LINDBLAD_TRAJ_DEPTH", "40"))
const LOBATTO_ORDERS = parse.(Int, split(get(ENV, "LINDBLAD_TRAJ_ORDERS", "2,3,4,5"), ","))
const CORE_RTOL = parse(Float64, get(ENV, "LINDBLAD_TRAJ_CORE_RTOL", "1e-8"))
const REPEATS = parse(Int, get(ENV, "LINDBLAD_TRAJ_REPEATS", "5"))
const OUTPUT = get(ENV, "LINDBLAD_TRAJ_OUTPUT",
    joinpath(@__DIR__, "..", "benchmark_results", "lindblad_trajectory_objective.csv"))

function ising_problem(q)
    site = (op, i) -> nsite_op_sparse(op, i, q)
    Hfield = sum(site(σz, i) for i in 1:q)
    constant = 0.3 * hamiltonian_superop_sparse(Hfield)
    directions = SparseMatrixCSC{ComplexF64,Int}[]
    for i in 1:q-1
        H = site(σx, i) * site(σx, i+1) + site(σy, i) * site(σy, i+1)
        push!(directions, hamiltonian_superop_sparse(H))
    end
    for i in 1:q
        push!(directions, lindblad_superop_sparse(site(σz, i)))
    end
    truth = vcat([0.5 * (1 + 0.4sin(i*pi*0.35)) for i in 1:q-1],
                 [0.1 * (1 + 0.5cos(i*pi*0.35)^2) for i in 1:q])
    iterate = truth .* [1 + 0.18sin(1.7i) for i in eachindex(truth)]
    Atrue = constant + sum(truth[j] .* directions[j] for j in eachindex(directions))
    Aeval = constant + sum(iterate[j] .* directions[j] for j in eachindex(directions))
    plus = ComplexF64[1, 1] / sqrt(2)
    psi = foldl(kron, [plus for _ in 1:q])
    state0 = vec(psi * psi')
    Atrue, Aeval, directions, state0
end

function observation_matrix(q)
    site = (op, i) -> nsite_op_sparse(op, i, q)
    d = 2^q
    operators = SparseMatrixCSC{ComplexF64,Int}[spdiagm(0 => ones(ComplexF64, d))]
    for op in (σx, σy, σz), i in 1:q
        push!(operators, site(op, i))
    end
    for op in (σx, σy, σz), i in 1:q-1
        push!(operators, site(op, i) * site(op, i+1))
    end
    C = Matrix{ComplexF64}(undef, length(operators), d^2)
    for (row, operator) in enumerate(operators)
        C[row, :] .= conj.(vec(Matrix(operator)))
    end
    C
end

function observations(Atrue, state0, C, times)
    [C * expv(time, Atrue, state0; m=max(DEPTH, 60), tol=1e-14) for time in times]
end

function model_trajectory(A, state0, dt, count)
    states = Vector{Vector{ComplexF64}}(undef, count + 1)
    states[1] = copy(state0)
    for k in 1:count
        states[k+1] = expv(dt, A, states[k]; m=DEPTH, tol=1e-14)
    end
    states
end

function observation_pullbacks(C, states, targets)
    [C' * (C * states[k+1] - targets[k]) for k in eachindex(targets)]
end

function endpoint_core_gradient(A, dirs, state0, C, targets, times;
                                core_rtol=CORE_RTOL)
    n, M = length(state0), length(dirs)
    cache = SparseDirectionCache(dirs)
    fw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    aw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    gradient = zeros(Float64, M)
    for k in eachindex(times)
        fb = arnoldi_basis!(fw, A, state0; m=DEPTH)
        prediction = view(_eval_krylov_batch(fb, [times[k]], fb.beta), :, 1)
        delta = C' * (C * prediction - targets[k])
        ab = arnoldi_basis!(aw, A', delta; m=DEPTH)
        gradient .+= _compressed_core_from_bases(fb, ab, dirs, times[k];
            core_rtol=core_rtol, core_atol=0.0, direction_cache=cache)
    end
    gradient
end

function shared_forward_endpoint_core_gradient(A, dirs, state0, C, targets,
                                               times; core_rtol=CORE_RTOL)
    n, M = length(state0), length(dirs)
    cache = SparseDirectionCache(dirs)
    fw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    aw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    fb = arnoldi_basis!(fw, A, state0; m=DEPTH)
    predictions = _eval_krylov_batch(fb, times, fb.beta)
    gradient = zeros(Float64, M)
    for k in eachindex(times)
        delta = C' * (C * view(predictions, :, k) - targets[k])
        ab = arnoldi_basis!(aw, A', delta; m=DEPTH)
        gradient .+= _compressed_core_from_bases(fb, ab, dirs, times[k];
            core_rtol=core_rtol, core_atol=0.0, direction_cache=cache)
    end
    gradient
end

function chained_core_gradient(A, dirs, state0, C, targets, dt;
                               core_rtol=CORE_RTOL)
    count, n, M = length(targets), length(state0), length(dirs)
    states = model_trajectory(A, state0, dt, count)
    jumps = observation_pullbacks(C, states, targets)
    cache = SparseDirectionCache(dirs)
    fw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    aw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    gradient = zeros(Float64, M)
    lambda = zeros(ComplexF64, n)
    for k in count:-1:1
        lambda .+= jumps[k]
        fb = arnoldi_basis!(fw, A, states[k]; m=DEPTH)
        ab = arnoldi_basis!(aw, A', lambda; m=DEPTH)
        gradient .+= _compressed_core_from_bases(fb, ab, dirs, dt;
            core_rtol=core_rtol, core_atol=0.0, direction_cache=cache)
        lambda .= view(_eval_krylov_batch(ab, [dt], ab.beta), :, 1)
    end
    gradient
end

function lobatto_interval_gradient!(gradient, kernel, projected, cache,
                                    fb, ab, f0, f1, lambda1, dt, order)
    x, weights = gausslobatto(order)
    nodes = (x .+ 1) ./ 2
    weights ./= 2
    lambda0 = view(_eval_krylov_batch(ab, [dt], ab.beta), :, 1)
    fill!(kernel, 0)
    @inbounds for coordinate in eachindex(kernel)
        row, column = cache.rows[coordinate], cache.cols[coordinate]
        kernel[coordinate] = weights[1] * conj(lambda0[row]) * f0[column] +
                             weights[end] * conj(lambda1[row]) * f1[column]
    end
    if order > 2
        interior = nodes[2:end-1]
        forward = _eval_krylov_batch(fb, dt .* interior, fb.beta)
        adjoint = _eval_krylov_batch(ab, dt .* (1 .- interior), ab.beta)
        @inbounds for coordinate in eachindex(kernel)
            row, column = cache.rows[coordinate], cache.cols[coordinate]
            for ell in eachindex(interior)
                kernel[coordinate] += weights[ell+1] *
                    conj(adjoint[row, ell]) * forward[column, ell]
            end
        end
    end
    mul!(projected, transpose(cache.coefficients), kernel)
    gradient .+= real.(dt .* projected)
    lambda0
end

function chained_lobatto_gradient(A, dirs, state0, C, targets, dt, order)
    count, n, M = length(targets), length(state0), length(dirs)
    states = model_trajectory(A, state0, dt, count)
    jumps = observation_pullbacks(C, states, targets)
    cache = SparseDirectionCache(dirs)
    fw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    aw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    gradient = zeros(Float64, M)
    kernel = zeros(ComplexF64, length(cache.rows))
    projected = zeros(ComplexF64, M)
    lambda = zeros(ComplexF64, n)
    for k in count:-1:1
        lambda .+= jumps[k]
        fb = arnoldi_basis!(fw, A, states[k]; m=DEPTH)
        ab = arnoldi_basis!(aw, A', lambda; m=DEPTH)
        lambda .= lobatto_interval_gradient!(gradient, kernel, projected, cache,
            fb, ab, states[k], states[k+1], lambda, dt, order)
    end
    gradient
end

function global_forward_lobatto_gradient(A, dirs, state0, C, targets, dt, order)
    count, n, M = length(targets), length(state0), length(dirs)
    x, weights = gausslobatto(order)
    nodes = (x .+ 1) ./ 2
    weights ./= 2

    # One forward Arnoldi decomposition supplies every observation checkpoint
    # and every interior quadrature value over the complete trajectory.
    times = Float64[0.0]
    for k in 1:count
        append!(times, ((k-1) .+ nodes[2:end]) .* dt)
    end
    fw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    fb = arnoldi_basis!(fw, A, state0; m=DEPTH)
    forward = _eval_krylov_batch(fb, times, fb.beta)
    checkpoint_column(k) = 1 + k * (order - 1)
    jumps = [C' * (C * view(forward, :, checkpoint_column(k)) - targets[k])
             for k in 1:count]

    cache = SparseDirectionCache(dirs)
    aw = ArnoldiWorkspace(ComplexF64, n, DEPTH)
    gradient = zeros(Float64, M)
    kernel = zeros(ComplexF64, length(cache.rows))
    projected = zeros(ComplexF64, M)
    lambda = zeros(ComplexF64, n)
    for k in count:-1:1
        lambda .+= jumps[k]
        ab = arnoldi_basis!(aw, A', lambda; m=DEPTH)
        lambda0 = view(_eval_krylov_batch(ab, [dt], ab.beta), :, 1)
        adjoint_interior = order > 2 ?
            _eval_krylov_batch(ab, dt .* (1 .- nodes[2:end-1]), ab.beta) :
            zeros(ComplexF64, n, 0)
        first_column = 1 + (k-1) * (order-1)
        fill!(kernel, 0)
        @inbounds for coordinate in eachindex(kernel)
            row, column = cache.rows[coordinate], cache.cols[coordinate]
            kernel[coordinate] =
                weights[1] * conj(lambda0[row]) * forward[column, first_column] +
                weights[end] * conj(lambda[row]) *
                    forward[column, first_column + order - 1]
            for ell in 1:(order-2)
                kernel[coordinate] += weights[ell+1] *
                    conj(adjoint_interior[row, ell]) *
                    forward[column, first_column + ell]
            end
        end
        mul!(projected, transpose(cache.coefficients), kernel)
        gradient .+= real.(dt .* projected)
        lambda .= lambda0
    end
    gradient
end

function median_ms(f)
    f()
    median((@elapsed f()) * 1000 for _ in 1:REPEATS)
end

Atrue, Aeval, directions, state0 = ising_problem(Q)
C = observation_matrix(Q)
rows = NamedTuple[]

@printf("Lindblad trajectory objective: q=%d n=%d M=%d T=%.2f m=%d\n",
    Q, length(state0), length(directions), HORIZON, DEPTH)

for count in COUNTS
    times = collect(range(HORIZON / count, HORIZON; length=count))
    dt = HORIZON / count
    targets = observations(Atrue, state0, C, times)

    reference = chained_core_gradient(Aeval, directions, state0, C, targets, dt;
        core_rtol=0.0)
    compressed = () -> chained_core_gradient(Aeval, directions, state0, C,
        targets, dt; core_rtol=CORE_RTOL)
    endpoint = () -> endpoint_core_gradient(Aeval, directions, state0, C,
        targets, times; core_rtol=CORE_RTOL)
    shared_endpoint = () -> shared_forward_endpoint_core_gradient(Aeval,
        directions, state0, C, targets, times; core_rtol=CORE_RTOL)
    compressed_value = compressed()
    endpoint_value = endpoint()
    shared_endpoint_value = shared_endpoint()
    compressed_ms = median_ms(compressed)
    endpoint_ms = median_ms(endpoint)
    shared_endpoint_ms = median_ms(shared_endpoint)
    compressed_error = norm(compressed_value-reference) / max(norm(reference), eps())
    endpoint_error = norm(endpoint_value-reference) / max(norm(reference), eps())
    shared_endpoint_error = norm(shared_endpoint_value-reference) /
                            max(norm(reference), eps())

    for order in LOBATTO_ORDERS
        quadrature = () -> chained_lobatto_gradient(Aeval, directions, state0,
            C, targets, dt, order)
        global_quadrature = () -> global_forward_lobatto_gradient(Aeval,
            directions, state0, C, targets, dt, order)
        qvalue = quadrature()
        global_qvalue = global_quadrature()
        qtime = median_ms(quadrature)
        global_qtime = median_ms(global_quadrature)
        qerror = norm(qvalue-reference) / max(norm(reference), eps())
        global_qerror = norm(global_qvalue-reference) / max(norm(reference), eps())
        push!(rows, (qubits=Q, liouville_dimension=length(state0),
            directions=length(directions), horizon=HORIZON,
            observation_count=count, interval=dt, krylov_depth=DEPTH,
            lobatto_order=order, repeats=REPEATS,
            endpoint_core_ms=endpoint_ms,
            shared_forward_endpoint_core_ms=shared_endpoint_ms,
            chained_compressed_core_ms=compressed_ms,
            chained_lobatto_ms=qtime,
            global_forward_lobatto_ms=global_qtime,
            endpoint_core_over_lobatto=endpoint_ms/qtime,
            shared_endpoint_core_over_lobatto=shared_endpoint_ms/qtime,
            chained_core_over_lobatto=compressed_ms/qtime,
            shared_endpoint_core_over_global_lobatto=shared_endpoint_ms/global_qtime,
            endpoint_core_relative_difference=endpoint_error,
            shared_endpoint_core_relative_difference=shared_endpoint_error,
            compressed_core_relative_difference=compressed_error,
            lobatto_relative_difference=qerror,
            global_lobatto_relative_difference=global_qerror))
        @printf("K=%2d dt=%6.3f q=%d endpoint=%8.2f shared=%8.2f core=%8.2f localQ=%8.2f globalQ=%8.2f ms shared/globalQ=%5.2f err=(%.1e, %.1e, %.1e, %.1e, %.1e)\n",
            count, dt, order, endpoint_ms, shared_endpoint_ms, compressed_ms,
            qtime, global_qtime, shared_endpoint_ms/global_qtime,
            endpoint_error, shared_endpoint_error, compressed_error, qerror,
            global_qerror)
    end
end

mkpath(dirname(OUTPUT))
CSV.write(OUTPUT, DataFrame(rows))
println("Saved ", OUTPUT)
