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
