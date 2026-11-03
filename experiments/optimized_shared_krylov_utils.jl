"""Reusable kernels for the optimized shared-Krylov timing experiments.

The optimized path keeps Arnoldi workspaces and the sparse library projection
alive across calls.  Within a call, each reduced Hessenberg matrix is
factorized once and evaluated at the endpoint and all quadrature nodes.
"""

using LinearAlgebra, SparseArrays

struct OptimizedQuadratureWorkspace{T, RT, C}
    forward :: ArnoldiWorkspace{T}
    adjoint :: ArnoldiWorkspace{T}
    direction_cache :: C
    kernel_values :: Vector{T}
    projected_values :: Vector{T}
    gradient :: Vector{RT}
    forward_times :: Vector{RT}
    adjoint_times :: Vector{RT}
end

function OptimizedQuadratureWorkspace(
    dirs::AbstractVector{<:SparseMatrixCSC{T}},
    n::Int,
    m::Int,
    n_quad::Int,
) where {T}
    cache = SparseDirectionCache(dirs)
    RT = real_scalar_type(T)
    OptimizedQuadratureWorkspace{T, RT, typeof(cache)}(
        ArnoldiWorkspace(T, n, min(m, n)),
        ArnoldiWorkspace(T, n, min(m, n)),
        cache,
        zeros(T, length(cache.rows)),
        zeros(T, length(dirs)),
        zeros(RT, length(dirs)),
        zeros(RT, n_quad + 1),
        zeros(RT, n_quad + 1),
    )
end

function set_evaluation_times!(workspace, nodes, dt)
    length(workspace.forward_times) == length(nodes) + 1 ||
        throw(DimensionMismatch("quadrature order does not match workspace"))
    workspace.forward_times[1] = dt
    workspace.adjoint_times[1] = dt
    @inbounds for index in eachindex(nodes)
        workspace.forward_times[index + 1] = dt * nodes[index]
        workspace.adjoint_times[index + 1] = dt * (1 - nodes[index])
    end
    nothing
end

function fused_library_projection!(
    workspace,
    forward_values,
    adjoint_values,
    weights,
    dt,
)
    cache = workspace.direction_cache
    kernel = workspace.kernel_values
    fill!(kernel, zero(eltype(kernel)))
    @inbounds for quadrature_index in eachindex(weights)
        weight = weights[quadrature_index]
        value_column = quadrature_index + 1
        for coordinate in eachindex(kernel)
            row = cache.rows[coordinate]
            column = cache.cols[coordinate]
            kernel[coordinate] +=
                weight * conj(adjoint_values[row, value_column]) *
                forward_values[column, value_column]
        end
    end
    mul!(
        workspace.projected_values,
        transpose(cache.coefficients),
        kernel,
    )
    @inbounds for index in eachindex(workspace.gradient)
        workspace.gradient[index] =
            real(dt * workspace.projected_values[index])
    end
    workspace.gradient
end

"""Evaluate a squared-error loss and its parameter and state VJPs.

The sparse generator ``A`` is supplied preassembled.  Arnoldi storage and the
library projection buffers are persistent.  Returned arrays are overwritten
by the next call using the same workspace.
"""
function optimized_loss_and_vjp!(
    workspace,
    A,
    p0,
    target,
    dt,
    nodes,
    weights;
    m_krylov,
)
    set_evaluation_times!(workspace, nodes, dt)
    forward_basis = arnoldi_basis!(
        workspace.forward, A, p0; m=m_krylov,
    )
    forward_values = _eval_krylov_batch(
        forward_basis,
        workspace.forward_times,
        forward_basis.beta,
    )
    prediction = view(forward_values, :, 1)
    upstream = prediction - target
    loss = real_scalar_type(eltype(p0))(0.5) * sum(abs2, upstream)

    adjoint_basis = arnoldi_basis!(
        workspace.adjoint, A', upstream; m=m_krylov,
    )
    adjoint_values = _eval_krylov_batch(
        adjoint_basis,
        workspace.adjoint_times,
        adjoint_basis.beta,
    )
    parameter_vjp = fused_library_projection!(
        workspace,
        forward_values,
        adjoint_values,
        weights,
        dt,
    )
    (
        loss=loss,
        prediction=prediction,
        parameter_vjp=parameter_vjp,
        state_vjp=view(adjoint_values, :, 1),
        forward_dimension=forward_basis.m,
        adjoint_dimension=adjoint_basis.m,
    )
end

"""Evaluate a linear cotangent and its semigroup parameter and state VJPs."""
function optimized_linear_loss_and_vjp!(
    workspace,
    A,
    p0,
    upstream,
    dt,
    nodes,
    weights;
    m_krylov,
)
    set_evaluation_times!(workspace, nodes, dt)
    forward_basis = arnoldi_basis!(
        workspace.forward, A, p0; m=m_krylov,
    )
    forward_values = _eval_krylov_batch(
        forward_basis,
        workspace.forward_times,
        forward_basis.beta,
    )
    adjoint_basis = arnoldi_basis!(
        workspace.adjoint, A', upstream; m=m_krylov,
    )
    adjoint_values = _eval_krylov_batch(
        adjoint_basis,
        workspace.adjoint_times,
        adjoint_basis.beta,
    )
    parameter_vjp = fused_library_projection!(
        workspace,
        forward_values,
        adjoint_values,
        weights,
        dt,
    )
    prediction = view(forward_values, :, 1)
    (
        loss=real(dot(upstream, prediction)),
        prediction=prediction,
        parameter_vjp=parameter_vjp,
        state_vjp=view(adjoint_values, :, 1),
        forward_dimension=forward_basis.m,
        adjoint_dimension=adjoint_basis.m,
    )
end

function reduced_path_batch(ks, times)
    H = Matrix(ks.H)
    e1 = zeros(eltype(H), ks.m)
    e1[1] = one(eltype(H))
    values = _eval_reduced_exponential_batch(H, times, e1)
    if values === nothing
        values = Matrix{eltype(H)}(undef, ks.m, length(times))
        @inbounds for index in eachindex(times)
            values[:, index] = exp(times[index] .* H) * e1
        end
    end
    ks.beta .* values
end

"""Compress the quadrature outer-product factors through an `n_quad`-square SVD.

The Arnoldi paths remain in reduced coordinates during compression.  If the
weighted reduced path matrices are `X` and `Y`, thin QR factorizations reduce
the SVD of `X*Y'` to the SVD of `R_X*R_Y'`.  Only the retained factors are
lifted to the full state space for contraction with the sparse library.
"""
function compressed_quadrature_linear_loss_and_vjp!(
    workspace,
    A,
    p0,
    upstream,
    dt,
    nodes,
    weights;
    m_krylov,
    retained_rank=length(nodes),
)
    dt >= 0 || throw(ArgumentError("compressed quadrature requires dt >= 0"))
    set_evaluation_times!(workspace, nodes, dt)
    forward_basis = arnoldi_basis!(
        workspace.forward, A, p0; m=m_krylov,
    )
    adjoint_basis = arnoldi_basis!(
        workspace.adjoint, A', upstream; m=m_krylov,
    )
    forward_reduced = reduced_path_batch(
        forward_basis, workspace.forward_times,
    )
    adjoint_reduced = reduced_path_batch(
        adjoint_basis, workspace.adjoint_times,
    )

    nq = length(nodes)
    rank = min(retained_rank, nq, forward_basis.m, adjoint_basis.m)
    scales = sqrt.(dt .* weights)
    forward_factors = forward_reduced[:, 2:end] .* transpose(scales)
    adjoint_factors = adjoint_reduced[:, 2:end] .* transpose(scales)
    forward_qr = qr(forward_factors)
    adjoint_qr = qr(adjoint_factors)
    Qf = Matrix(forward_qr.Q)[:, 1:nq]
    Qa = Matrix(adjoint_qr.Q)[:, 1:nq]
    Rf = Matrix(forward_qr.R)[1:nq, 1:nq]
    Ra = Matrix(adjoint_qr.R)[1:nq, 1:nq]
    decomposition = svd(Ra * Rf')
    left_coordinates = Qa * view(decomposition.U, :, 1:rank)
    right_coordinates = Qf * view(decomposition.V, :, 1:rank)
    left_factors = view(adjoint_basis.V, :, 1:adjoint_basis.m) *
                   left_coordinates
    right_factors = view(forward_basis.V, :, 1:forward_basis.m) *
                    right_coordinates

    cache = workspace.direction_cache
    kernel = workspace.kernel_values
    fill!(kernel, zero(eltype(kernel)))
    @inbounds for factor_index in 1:rank
        sigma = decomposition.S[factor_index]
        for coordinate in eachindex(kernel)
            row = cache.rows[coordinate]
            column = cache.cols[coordinate]
            kernel[coordinate] += sigma *
                conj(left_factors[row, factor_index]) *
                right_factors[column, factor_index]
        end
    end
    mul!(
        workspace.projected_values,
        transpose(cache.coefficients),
        kernel,
    )
    workspace.gradient .= real.(workspace.projected_values)

    prediction = view(forward_basis.V, :, 1:forward_basis.m) *
                 view(forward_reduced, :, 1)
    state_vjp = view(adjoint_basis.V, :, 1:adjoint_basis.m) *
                view(adjoint_reduced, :, 1)
    (
        loss=real(dot(upstream, prediction)),
        prediction=prediction,
        parameter_vjp=workspace.gradient,
        state_vjp=state_vjp,
        singular_values=decomposition.S,
        retained_rank=rank,
        forward_dimension=forward_basis.m,
        adjoint_dimension=adjoint_basis.m,
    )
end
