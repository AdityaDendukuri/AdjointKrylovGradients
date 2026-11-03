"""
Peak-memory comparison between streamed trajectory quadrature and the
two-sided projected core.

The generator is a matrix-free local Lindblad model with site-dependent
Z fields and dephasing.  Its Liouville-space diagonal is stored in O(n)
memory, where n = 4^q.  Two noncommuting local directions are used for the
gradient contractions.

The quadrature path constructs the forward Arnoldi basis, stores the forward
path at the quadrature nodes, and then overwrites the same Arnoldi workspace
with the adjoint basis.  Adjoint values are lifted one node at a time.

The projected-core path follows Kandolf et al., Algorithm 2.  It constructs
and retains both Arnoldi bases, forms the reduced core, and streams the
singular-vector contractions one factor at a time.  This avoids charging the
core method for full-space factor matrices that are not required.

Run each method in a fresh process and measure peak RSS with `/usr/bin/time`.

Environment variables:
  LINDBLAD_MEMORY_METHOD   quadrature, core, or
                           core_recompute            (default quadrature)
  LINDBLAD_MEMORY_Q        qubit count              (default 9)
  LINDBLAD_MEMORY_M        Krylov depth             (default 24)
  LINDBLAD_MEMORY_NQ       Gauss--Legendre nodes    (default 6)
  LINDBLAD_MEMORY_DT       interval length          (default 0.4)
  LINDBLAD_MEMORY_RTOL     core truncation tolerance(default 1e-12)
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, arnoldi_basis!,
    _eval_krylov_batch
using LinearAlgebra, Printf, SparseArrays

const METHOD = Symbol(lowercase(get(ENV, "LINDBLAD_MEMORY_METHOD", "quadrature")))
const Q = parse(Int, get(ENV, "LINDBLAD_MEMORY_Q", "9"))
const DEPTH = parse(Int, get(ENV, "LINDBLAD_MEMORY_M", "24"))
const NQ = parse(Int, get(ENV, "LINDBLAD_MEMORY_NQ", "6"))
const DT = parse(Float64, get(ENV, "LINDBLAD_MEMORY_DT", "0.4"))
const CORE_RTOL = parse(Float64, get(ENV, "LINDBLAD_MEMORY_RTOL", "1e-12"))

METHOD in (:quadrature, :core, :core_recompute) ||
    error("LINDBLAD_MEMORY_METHOD must be quadrature, core, or core_recompute")

struct LocalDirection{T}
    site::Int
    rows::Vector{Int}
    columns::Vector{Int}
    values::Vector{T}
end

function LocalDirection(site::Int, matrix::AbstractMatrix{T}) where {T}
    sparse_matrix = sparse(matrix)
    rows = Int[]
    columns = Int[]
    values = T[]
    matrix_rows = rowvals(sparse_matrix)
    matrix_values = nonzeros(sparse_matrix)
    for column in axes(sparse_matrix, 2)
        for pointer in nzrange(sparse_matrix, column)
            push!(rows, matrix_rows[pointer])
            push!(columns, column)
            push!(values, matrix_values[pointer])
        end
    end
    LocalDirection{T}(site, rows, columns, values)
end

function apply_direction!(output, direction::LocalDirection, input, q)
    fill!(output, zero(eltype(output)))
    stride = 4^(direction.site - 1)
    block = 4stride
    outer_count = length(input) ÷ block
    @inbounds for outer in 0:(outer_count - 1)
        base = outer * block
        for inner in 0:(stride - 1)
            offset = base + inner
            for entry in eachindex(direction.values)
                row = direction.rows[entry] - 1
                column = direction.columns[entry] - 1
                output[offset + row * stride + 1] +=
                    direction.values[entry] *
                    input[offset + column * stride + 1]
            end
        end
    end
    output
end

function lindblad_diagonal(q)
    n = 4^q
    diagonal = zeros(ComplexF64, n)
    # Local vectorization order is (00, 10, 01, 11).
    z_difference = (0.0, -2.0, 2.0, 0.0)
    z_product = (1.0, -1.0, -1.0, 1.0)
    for site in 1:q
        frequency = 0.20 + 0.017site
        dephasing = 0.035 + 0.003site
        local_values = ntuple(4) do digit
            -im * frequency * z_difference[digit] +
            dephasing * (z_product[digit] - 1)
        end
        stride = 4^(site - 1)
        block = 4stride
        outer_count = n ÷ block
        @inbounds for outer in 0:(outer_count - 1)
            base = outer * block
            for digit in 0:3
                value = local_values[digit + 1]
                start = base + digit * stride + 1
                stop = start + stride - 1
                for index in start:stop
                    diagonal[index] += value
                end
            end
        end
    end
    diagonal
end

function product_vector(local_vectors)
    value = ComplexF64[1]
    for local_vector in local_vectors
        value = kron(local_vector, value)
    end
    value
end

function initial_state(q)
    factors = Vector{Vector{ComplexF64}}(undef, q)
    for site in 1:q
        phase = 0.19site
        amplitude = 0.55 + 0.02sin(0.7site)
        psi = ComplexF64[1, amplitude * cis(phase)]
        psi ./= norm(psi)
        factors[site] = vec(psi * psi')
    end
    product_vector(factors)
end

function cotangent(q)
    factors = Vector{Vector{ComplexF64}}(undef, q)
    for site in 1:q
        phase = 0.31site
        observable = ComplexF64[
            0.7 + 0.01site       0.23cis(-phase)
            0.23cis(phase)      -0.4 + 0.005site
        ]
        factors[site] = vec(observable) / norm(observable)
    end
    value = product_vector(factors)
    value / norm(value)
end

function directions(q)
    site = max(1, q ÷ 2)
    x_direction = hamiltonian_superop(ComplexF64[0 1; 1 0])
    lowering = ComplexF64[0 1; 0 0]
    decay_direction = lindblad_superop(lowering)
    [LocalDirection(site, x_direction),
     LocalDirection(site, decay_direction)]
end

function contract_directions!(gradient, directions, left, right, buffer, q, scale)
    for (index, direction) in pairs(directions)
        apply_direction!(buffer, direction, right, q)
        gradient[index] += real(scale * dot(left, buffer))
    end
    gradient
end

function exponential_divided_difference(left_value, right_value, dt)
    difference = right_value - left_value
    scale = max(abs(left_value), abs(right_value), 1.0)
    if abs(difference) <= 32eps(Float64) * scale
        return dt * exp(dt * left_value)
    end
    exp(dt * left_value) * expm1(dt * difference) / difference
end

function exact_gradient(diagonal, state, upstream, directions, q, dt)
    gradient = zeros(Float64, length(directions))
    n = length(state)
    for (direction_index, direction) in pairs(directions)
        value = zero(ComplexF64)
        stride = 4^(direction.site - 1)
        block = 4stride
        outer_count = n ÷ block
        @inbounds for outer in 0:(outer_count - 1)
            base = outer * block
            for inner in 0:(stride - 1)
                offset = base + inner
                for entry in eachindex(direction.values)
                    row = offset + (direction.rows[entry] - 1) * stride + 1
                    column = offset +
                             (direction.columns[entry] - 1) * stride + 1
                    divided_difference = exponential_divided_difference(
                        diagonal[row], diagonal[column], dt,
                    )
                    value += conj(upstream[row]) * direction.values[entry] *
                             state[column] * divided_difference
                end
            end
        end
        gradient[direction_index] = real(value)
    end
    gradient
end

function reduced_action!(output, basis, time, reduced_matrix, beta, work)
    fill!(work, zero(eltype(work)))
    work[1] = one(eltype(work))
    coefficients = exp(time .* reduced_matrix) * work
    mul!(output, view(basis.V, :, 1:basis.m), coefficients)
    output .*= beta
    output
end

function streamed_quadrature(generator, state, upstream, directions, q, depth,
                              nodes, weights, dt)
    n = length(state)
    workspace = ArnoldiWorkspace(ComplexF64, n, depth)
    forward_basis = arnoldi_basis!(workspace, generator, state; m=depth)
    forward_times = vcat(dt, dt .* nodes)
    forward_values = _eval_krylov_batch(
        forward_basis, forward_times, forward_basis.beta,
    )
    prediction_norm = norm(view(forward_values, :, 1))

    # This call deliberately overwrites the forward basis.  The forward path
    # values above are all that the quadrature contraction still needs.
    adjoint_basis = arnoldi_basis!(workspace, adjoint(generator), upstream; m=depth)
    reduced_adjoint = Matrix(adjoint_basis.H)
    reduced_work = zeros(ComplexF64, adjoint_basis.m)
    adjoint_value = zeros(ComplexF64, n)
    direction_buffer = zeros(ComplexF64, n)
    gradient = zeros(Float64, length(directions))
    for node_index in eachindex(nodes)
        reduced_action!(
            adjoint_value, adjoint_basis, dt * (1 - nodes[node_index]),
            reduced_adjoint, adjoint_basis.beta, reduced_work,
        )
        contract_directions!(
            gradient, directions, adjoint_value,
            view(forward_values, :, node_index + 1), direction_buffer, q,
            dt * weights[node_index],
        )
    end
    (
        gradient=gradient,
        prediction_norm=prediction_norm,
        forward_dimension=forward_basis.m,
        adjoint_dimension=adjoint_basis.m,
    )
end

function projected_core(generator, state, upstream, directions, q, depth, dt,
                        core_rtol)
    n = length(state)
    forward_workspace = ArnoldiWorkspace(ComplexF64, n, depth)
    adjoint_workspace = ArnoldiWorkspace(ComplexF64, n, depth)
    forward_basis = arnoldi_basis!(
        forward_workspace, generator, state; m=depth,
    )
    adjoint_basis = arnoldi_basis!(
        adjoint_workspace, adjoint(generator), upstream; m=depth,
    )
    mf = forward_basis.m
    ma = adjoint_basis.m
    block = zeros(ComplexF64, ma + mf, ma + mf)
    block[1:ma, 1:ma] .= adjoint_basis.H
    block[(ma + 1):end, (ma + 1):end] .= forward_basis.H'
    block[1, ma + 1] = adjoint_basis.beta * forward_basis.beta
    core = exp(dt .* block)[1:ma, (ma + 1):end]
    decomposition = svd(core)
    threshold = core_rtol * norm(decomposition.S)
    tail_squared = zero(Float64)
    rank = length(decomposition.S)
    for candidate in length(decomposition.S):-1:1
        proposed_tail = tail_squared + abs2(decomposition.S[candidate])
        if sqrt(proposed_tail) > threshold
            rank = candidate
            break
        end
        tail_squared = proposed_tail
        rank = candidate - 1
    end

    left = zeros(ComplexF64, n)
    right = zeros(ComplexF64, n)
    direction_buffer = zeros(ComplexF64, n)
    gradient = zeros(Float64, length(directions))
    for factor in 1:rank
        mul!(
            left, view(adjoint_basis.V, :, 1:ma),
            view(decomposition.U, :, factor),
        )
        mul!(
            right, view(forward_basis.V, :, 1:mf),
            view(decomposition.V, :, factor),
        )
        contract_directions!(
            gradient, directions, left, right, direction_buffer, q,
            decomposition.S[factor],
        )
    end
    endpoint = zeros(ComplexF64, n)
    reduced_work = zeros(ComplexF64, mf)
    reduced_action!(
        endpoint, forward_basis, dt, Matrix(forward_basis.H),
        forward_basis.beta, reduced_work,
    )
    (
        gradient=gradient,
        prediction_norm=norm(endpoint),
        forward_dimension=mf,
        adjoint_dimension=ma,
        core_rank=rank,
    )
end

function selected_core_rank(singular_values, core_rtol)
    threshold = core_rtol * norm(singular_values)
    tail_squared = zero(Float64)
    rank = length(singular_values)
    for candidate in length(singular_values):-1:1
        proposed_tail = tail_squared + abs2(singular_values[candidate])
        if sqrt(proposed_tail) > threshold
            return candidate
        end
        tail_squared = proposed_tail
        rank = candidate - 1
    end
    rank
end


function projected_core_recompute(generator, state, upstream, directions, q,
                                  depth, dt, core_rtol)
    n = length(state)
    workspace = ArnoldiWorkspace(ComplexF64, n, depth)

    # The first forward pass leaves only its reduced matrix and scalar norm.
    forward_basis = arnoldi_basis!(workspace, generator, state; m=depth)
    mf = forward_basis.m
    forward_reduced = Matrix(forward_basis.H)
    forward_beta = forward_basis.beta
    endpoint = zeros(ComplexF64, n)
    reduced_work = zeros(ComplexF64, mf)
    reduced_action!(
        endpoint, forward_basis, dt, forward_reduced, forward_beta,
        reduced_work,
    )
    prediction_norm = norm(endpoint)
    endpoint = nothing
    GC.gc()

    # The adjoint pass overwrites the forward basis.  Its selected full-space
    # factors are retained before the workspace is reused a second time.
    adjoint_basis = arnoldi_basis!(
        workspace, adjoint(generator), upstream; m=depth,
    )
    ma = adjoint_basis.m
    adjoint_reduced = Matrix(adjoint_basis.H)
    block = zeros(ComplexF64, ma + mf, ma + mf)
    block[1:ma, 1:ma] .= adjoint_reduced
    block[(ma + 1):end, (ma + 1):end] .= forward_reduced'
    block[1, ma + 1] = adjoint_basis.beta * forward_beta
    core = exp(dt .* block)[1:ma, (ma + 1):end]
    decomposition = svd(core)
    rank = selected_core_rank(decomposition.S, core_rtol)
    left_factors = view(adjoint_basis.V, :, 1:ma) *
                   view(decomposition.U, :, 1:rank)

    # Rebuilding the forward basis replaces storage of both Arnoldi bases by
    # one basis and the selected adjoint-side factors.
    forward_basis = arnoldi_basis!(workspace, generator, state; m=depth)
    right = zeros(ComplexF64, n)
    direction_buffer = zeros(ComplexF64, n)
    gradient = zeros(Float64, length(directions))
    for factor in 1:rank
        mul!(
            right, view(forward_basis.V, :, 1:forward_basis.m),
            view(decomposition.V, :, factor),
        )
        contract_directions!(
            gradient, directions, view(left_factors, :, factor), right,
            direction_buffer, q, decomposition.S[factor],
        )
    end
    (
        gradient=gradient,
        prediction_norm=prediction_norm,
        forward_dimension=mf,
        adjoint_dimension=ma,
        core_rank=rank,
    )
end

function gibibytes(bytes)
    bytes / 2.0^30
end

function storage_model(method, n, depth, order)
    bytes_per_value = sizeof(ComplexF64)
    large_vectors = if method === :quadrature
        depth + order + 4
    elseif method === :core
        2depth + 6
    else
        # The selected rank is not known before the reduced core is formed.
        # The result line replaces this conservative order-based estimate
        # with the measured rank when reporting the final calculation.
        depth + order + 4
    end
    large_vectors, gibibytes(bytes_per_value * n * large_vectors)
end

function main()
    @printf(
        "Building q=%d diagonal local Lindblad generator (n=%d)\n", Q, 4^Q,
    )
    diagonal = lindblad_diagonal(Q)
    generator = Diagonal(diagonal)
    state = initial_state(Q)
    upstream = cotangent(Q)
    local_directions = directions(Q)
    nodes, weights = gauss_legendre_01(NQ)
    n = length(state)
    reference_gradient = exact_gradient(
        diagonal, state, upstream, local_directions, Q, DT,
    )
    modeled_vectors, modeled_gib = storage_model(METHOD, n, DEPTH, NQ)
    GC.gc()

    elapsed = @elapsed result = if METHOD === :quadrature
        streamed_quadrature(
            generator, state, upstream, local_directions, Q, DEPTH,
            nodes, weights, DT,
        )
    elseif METHOD === :core
        projected_core(
            generator, state, upstream, local_directions, Q, DEPTH, DT,
            CORE_RTOL,
        )
    else
        projected_core_recompute(
            generator, state, upstream, local_directions, Q, DEPTH, DT,
            CORE_RTOL,
        )
    end

    rank = hasproperty(result, :core_rank) ? result.core_rank : 0
    if METHOD === :core_recompute
        modeled_vectors = DEPTH + rank + 4
        modeled_gib = gibibytes(sizeof(ComplexF64) * n * modeled_vectors)
    end
    relative_error = norm(result.gradient - reference_gradient) /
                     max(norm(reference_gradient), eps())
    @printf(
        "RESULT method=%s q=%d n=%d m=%d nq=%d elapsed_s=%.6f modeled_vectors=%d modeled_large_storage_gib=%.6f mf=%d ma=%d core_rank=%d relative_error=%.16e prediction_norm=%.16e gradient_1=%.16e gradient_2=%.16e reference_1=%.16e reference_2=%.16e\n",
        String(METHOD), Q, n, DEPTH, NQ, elapsed, modeled_vectors,
        modeled_gib, result.forward_dimension, result.adjoint_dimension, rank,
        relative_error, result.prediction_norm, result.gradient[1],
        result.gradient[2], reference_gradient[1], reference_gradient[2],
    )
end

main()
