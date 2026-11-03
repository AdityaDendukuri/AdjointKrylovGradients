"""Commutator-aware quadrature on a mixed KS operator library.

The library contains translation-invariant circulant directions, which commute
with the constant-coefficient KS generator, and spatially modulated directions,
which do not.  We compare a uniform Gauss--Legendre rule with a grouped rule
that evaluates commuting directions from the endpoint and applies quadrature
only to the noncommuting group.  Both direct sparse contractions and the fused
union-sparsity kernel used by the optimized implementation are timed.
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, SparseDirectionCache,
    arnoldi_basis!, _eval_krylov_batch, compressed_core_frechet_vjp
using LinearAlgebra, SparseArrays, Statistics, Printf
using FastGaussQuadrature: gausslegendre
using CSV, DataFrames

const N = parse(Int, get(ENV, "COMM_KS_N", "2048"))
const M = parse(Int, get(ENV, "COMM_KS_LIBRARY", "256"))
const DEPTH = parse(Int, get(ENV, "COMM_KS_DEPTH", "40"))
const ORDERS = parse.(Int, split(get(ENV, "COMM_KS_ORDERS", "2,3,4,5,6,8"), ","))
const FRACTIONS = parse.(Float64, split(get(ENV, "COMM_KS_FRACTIONS", "0,0.25,0.5,0.75,1"), ","))
const REPEATS = parse(Int, get(ENV, "COMM_KS_REPEATS", "7"))
const DT = parse(Float64, get(ENV, "COMM_KS_DT", "0.0125"))
const DOMAIN_LENGTH = 22.0 * N / 64
const OUTPUT = get(ENV, "COMM_KS_OUTPUT",
    joinpath(@__DIR__, "..", "benchmark_results", "ks_commutator_grouped_quadrature.csv"))

function periodic_derivatives(n, domain_length)
    dx = domain_length / n
    second = spdiagm(
        -1 => fill(1.0 / dx^2, n - 1),
         0 => fill(-2.0 / dx^2, n),
         1 => fill(1.0 / dx^2, n - 1),
    )
    second[1, n] = second[n, 1] = 1.0 / dx^2
    second, sparse(second * second)
end

"""A symmetric periodic stencil at offset k, scaled to unit Frobenius norm."""
function circulant_direction(n, k)
    rows = Vector{Int}(undef, 3n)
    cols = Vector{Int}(undef, 3n)
    vals = Vector{Float64}(undef, 3n)
    for column in 1:n
        base = 3(column - 1)
        rows[base + 1] = mod1(column + k, n)
        rows[base + 2] = mod1(column - k, n)
        rows[base + 3] = column
        cols[base + 1] = cols[base + 2] = cols[base + 3] = column
        vals[base + 1] = vals[base + 2] = 0.5
        vals[base + 3] = -1.0
    end
    direction = sparse(rows, cols, vals, n, n)
    direction ./ norm(direction)
end

function mixed_library(n, count, commuting_count, xs)
    commuting = SparseMatrixCSC{Float64,Int}[]
    noncommuting = SparseMatrixCSC{Float64,Int}[]
    for index in 1:commuting_count
        push!(commuting, circulant_direction(n, index))
    end
    for index in 1:(count - commuting_count)
        base = circulant_direction(n, mod1(index, max(1, fld(n, 8))))
        mode = fld(index - 1, 2) + 1
        field = isodd(index) ? cos.(2pi .* mode .* xs ./ DOMAIN_LENGTH) :
                              sin.(2pi .* mode .* xs ./ DOMAIN_LENGTH)
        direction = sparse(spdiagm(0 => field) * base)
        push!(noncommuting, direction ./ max(norm(direction), eps()))
    end
    commuting, noncommuting
end

gauss01(order) = let (x, w) = gausslegendre(order); ((x .+ 1) ./ 2, w ./ 2) end

mutable struct PathWorkspace
    forward::ArnoldiWorkspace{Float64}
    adjoint::ArnoldiWorkspace{Float64}
    buffer::Vector{Float64}
end

PathWorkspace(n, depth) = PathWorkspace(
    ArnoldiWorkspace(Float64, n, depth),
    ArnoldiWorkspace(Float64, n, depth),
    zeros(n),
)

function paths!(workspace, A, state, cotangent, nodes)
    forward = arnoldi_basis!(workspace.forward, A, state; m=DEPTH)
    adjoint = arnoldi_basis!(workspace.adjoint, A', cotangent; m=DEPTH)
    forward_values = _eval_krylov_batch(
        forward, vcat(DT, DT .* nodes), forward.beta,
    )
    adjoint_values = _eval_krylov_batch(
        adjoint, vcat(DT, DT .* (1 .- nodes)), adjoint.beta,
    )
    forward_values, adjoint_values
end

function direct_uniform!(gradient, workspace, A, directions, state, cotangent,
                         nodes, weights)
    forward, adjoint = paths!(workspace, A, state, cotangent, nodes)
    for j in eachindex(directions)
        value = 0.0
        for q in eachindex(nodes)
            mul!(workspace.buffer, directions[j], view(forward, :, q + 1))
            value += weights[q] * dot(view(adjoint, :, q + 1), workspace.buffer)
        end
        gradient[j] = DT * value
    end
    gradient
end

function direct_grouped!(gradient, workspace, A, commuting, noncommuting,
                         state, cotangent, nodes, weights)
    forward, adjoint = paths!(workspace, A, state, cotangent, nodes)
    endpoint = view(forward, :, 1)
    for j in eachindex(commuting)
        mul!(workspace.buffer, commuting[j], endpoint)
        gradient[j] = DT * dot(cotangent, workspace.buffer)
    end
    offset = length(commuting)
    for j in eachindex(noncommuting)
        value = 0.0
        for q in eachindex(nodes)
            mul!(workspace.buffer, noncommuting[j], view(forward, :, q + 1))
            value += weights[q] * dot(view(adjoint, :, q + 1), workspace.buffer)
        end
        gradient[offset + j] = DT * value
    end
    gradient
end

function project_kernel!(output, kernel, cache, forward, adjoint, weights)
    fill!(kernel, 0.0)
    for q in eachindex(weights)
        fq = view(forward, :, q + 1)
        aq = view(adjoint, :, q + 1)
        weight = weights[q]
        @inbounds for coordinate in eachindex(kernel)
            kernel[coordinate] += weight * aq[cache.rows[coordinate]] *
                                  fq[cache.cols[coordinate]]
        end
    end
    mul!(output, transpose(cache.coefficients), kernel)
    output .*= DT
end

function endpoint_project!(output, kernel, cache, endpoint, cotangent)
    @inbounds for coordinate in eachindex(kernel)
        kernel[coordinate] = cotangent[cache.rows[coordinate]] *
                             endpoint[cache.cols[coordinate]]
    end
    mul!(output, transpose(cache.coefficients), kernel)
    output .*= DT
end

function fused_uniform!(gradient, workspace, cache, kernel, A, state, cotangent,
                        nodes, weights)
    forward, adjoint = paths!(workspace, A, state, cotangent, nodes)
    project_kernel!(gradient, kernel, cache, forward, adjoint, weights)
end

function fused_grouped!(gradient, workspace, commuting_cache, noncommuting_cache,
                        commuting_kernel, noncommuting_kernel, commuting_output,
                        noncommuting_output, A, state, cotangent, nodes, weights)
    forward, adjoint = paths!(workspace, A, state, cotangent, nodes)
    if commuting_cache !== nothing
        endpoint_project!(commuting_output, commuting_kernel, commuting_cache,
                          view(forward, :, 1), cotangent)
        gradient[1:length(commuting_output)] .= commuting_output
    end
    if noncommuting_cache !== nothing
        project_kernel!(noncommuting_output, noncommuting_kernel,
                        noncommuting_cache, forward, adjoint, weights)
        offset = length(commuting_output)
        gradient[offset + 1:end] .= noncommuting_output
    end
    gradient
end

median_ms(f) = begin
    f()
    median((@elapsed f()) * 1000 for _ in 1:REPEATS)
end

second, fourth = periodic_derivatives(N, DOMAIN_LENGTH)
A = sparse(-second - fourth)
xs = collect(range(0, DOMAIN_LENGTH; length=N + 1))[1:end-1]
phase = 2pi .* xs ./ DOMAIN_LENGTH
# A resolved band of Fourier modes gives the noncommuting directions a
# nontrivial time dependence while keeping the reference Krylov calculation
# exact to roundoff at the selected depth.
state = zeros(N)
cotangent = zeros(N)
for mode in 1:28
    state .+= cos.(mode .* phase .+ 0.13mode) ./ sqrt(mode)
    cotangent .+= cos.(mode .* phase .- 0.19mode) ./ sqrt(mode + 1)
end
state ./= norm(state)
cotangent ./= norm(cotangent)

rows = NamedTuple[]
for fraction in FRACTIONS
    commuting_count = clamp(round(Int, fraction * M), 0, M)
    commuting, noncommuting = mixed_library(N, M, commuting_count, xs)
    directions = vcat(commuting, noncommuting)
    full_cache = SparseDirectionCache(directions)
    commuting_cache = isempty(commuting) ? nothing : SparseDirectionCache(commuting)
    noncommuting_cache = isempty(noncommuting) ? nothing : SparseDirectionCache(noncommuting)

    # An independent projected-core value removes quadrature error from the reference.
    reference = compressed_core_frechet_vjp(
        A, directions, state, DT, cotangent;
        m_krylov=DEPTH, core_rtol=0.0, core_atol=0.0,
        ws_f=ArnoldiWorkspace(Float64, N, DEPTH),
        ws_a=ArnoldiWorkspace(Float64, N, DEPTH),
        direction_cache=full_cache,
    )

    for order in ORDERS
        nodes, weights = gauss01(order)

        direct_workspace = PathWorkspace(N, DEPTH)
        direct_uniform_gradient = zeros(M)
        direct_grouped_gradient = zeros(M)
        direct_uniform = () -> direct_uniform!(direct_uniform_gradient,
            direct_workspace, A, directions, state, cotangent, nodes, weights)
        direct_grouped = () -> direct_grouped!(direct_grouped_gradient,
            direct_workspace, A, commuting, noncommuting, state, cotangent,
            nodes, weights)

        fused_workspace = PathWorkspace(N, DEPTH)
        fused_uniform_gradient = zeros(M)
        fused_grouped_gradient = zeros(M)
        full_kernel = zeros(length(full_cache.rows))
        commuting_kernel = commuting_cache === nothing ? Float64[] :
            zeros(length(commuting_cache.rows))
        noncommuting_kernel = noncommuting_cache === nothing ? Float64[] :
            zeros(length(noncommuting_cache.rows))
        commuting_output = zeros(length(commuting))
        noncommuting_output = zeros(length(noncommuting))
        fused_uniform = () -> fused_uniform!(fused_uniform_gradient,
            fused_workspace, full_cache, full_kernel, A, state, cotangent,
            nodes, weights)
        fused_grouped = () -> fused_grouped!(fused_grouped_gradient,
            fused_workspace, commuting_cache, noncommuting_cache,
            commuting_kernel, noncommuting_kernel, commuting_output,
            noncommuting_output, A, state, cotangent, nodes, weights)

        du = copy(direct_uniform())
        dg = copy(direct_grouped())
        fu = copy(fused_uniform())
        fg = copy(fused_grouped())
        du_ms, dg_ms = median_ms(direct_uniform), median_ms(direct_grouped)
        fu_ms, fg_ms = median_ms(fused_uniform), median_ms(fused_grouped)

        scale = max(norm(reference), eps())
        direct_count_uniform = M * order
        direct_count_grouped = commuting_count + (M - commuting_count) * order
        push!(rows, (
            n=N, library_size=M, commuting_count=commuting_count,
            commuting_fraction=commuting_count / M, krylov_depth=DEPTH,
            quadrature_order=order,
            direct_uniform_ms=du_ms, direct_grouped_ms=dg_ms,
            direct_speedup=du_ms / dg_ms,
            fused_uniform_ms=fu_ms, fused_grouped_ms=fg_ms,
            fused_speedup=fu_ms / fg_ms,
            direct_uniform_relative_error=norm(du-reference)/scale,
            direct_grouped_relative_error=norm(dg-reference)/scale,
            fused_uniform_relative_error=norm(fu-reference)/scale,
            fused_grouped_relative_error=norm(fg-reference)/scale,
            grouped_direct_vs_fused=norm(dg-fg)/scale,
            direct_contractions_uniform=direct_count_uniform,
            direct_contractions_grouped=direct_count_grouped,
            contraction_reduction=1-direct_count_grouped/direct_count_uniform,
            full_union_coordinates=length(full_cache.rows),
            commuting_union_coordinates=commuting_cache === nothing ? 0 : length(commuting_cache.rows),
            noncommuting_union_coordinates=noncommuting_cache === nothing ? 0 : length(noncommuting_cache.rows),
            repeats=REPEATS,
        ))
        @printf("comm=%3d/%3d q=%d direct %7.2f -> %7.2f ms (%4.2fx), fused %7.2f -> %7.2f ms (%4.2fx), err %.2e\n",
            commuting_count, M, order, du_ms, dg_ms, du_ms/dg_ms,
            fu_ms, fg_ms, fu_ms/fg_ms, norm(fg-reference)/scale)
    end
end

mkpath(dirname(OUTPUT))
CSV.write(OUTPUT, DataFrame(rows))
println("Saved ", OUTPUT)
