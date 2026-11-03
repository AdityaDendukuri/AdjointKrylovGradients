"""Quadrature/core regime audit on the paper's Fourier-modulated KS library.

Both timings include the forward prediction, parameter VJP, and input-state
VJP.  The projected core is retained without SVD truncation, so it supplies a
matched-basis reference for the quadrature-order sweep.
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, SparseDirectionCache,
    arnoldi_basis!, _eval_krylov_batch, _compressed_core_from_bases,
    real_scalar_type
using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames
using FastGaussQuadrature: gausslobatto

include("optimized_shared_krylov_utils.jl")

const N = parse(Int, get(ENV, "KS_REGIME_N", "2048"))
const LIBRARY_SIZES = parse.(Int, split(get(ENV, "KS_REGIME_LIBRARIES", "16,64,256"), ","))
const M = parse(Int, get(ENV, "KS_REGIME_M", "40"))
const ORDERS = parse.(Int, split(get(ENV, "KS_REGIME_ORDERS", "2,3,4,5,6,8"), ","))
const REPEATS = parse(Int, get(ENV, "KS_REGIME_REPEATS", "7"))
const BASE_DX = 22.0 / 64
const DOMAIN_LENGTH = BASE_DX * N
const DT = 0.0125
const OUTPUT = get(ENV, "KS_REGIME_OUTPUT",
    joinpath(@__DIR__, "..", "benchmark_results", "ks_quadrature_regime.csv"))

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

function coefficient_field(xs, index)
    mode = fld(index - 1, 2) + 1
    isodd(index) ? cos.(2pi .* mode .* xs ./ DOMAIN_LENGTH) :
                   sin.(2pi .* mode .* xs ./ DOMAIN_LENGTH)
end

function direction_library(xs, second, fourth, count)
    directions = SparseMatrixCSC{Float64,Int}[]
    for index in 1:count
        field = coefficient_field(xs, index)
        derivative = isodd(fld(index - 1, 2)) ? second : fourth
        scale = derivative === second ? 1.0 : BASE_DX^2
        push!(directions, sparse(spdiagm(0 => field) * (-scale .* derivative)))
    end
    directions
end

median_ms(f) = (f(); median((@elapsed f()) * 1000 for _ in 1:REPEATS))

function full_lobatto!(workspace, A, state, cotangent, nodes, weights)
    interior = length(nodes) - 2
    workspace.forward_times[1] = DT
    workspace.adjoint_times[1] = DT
    for k in 1:interior
        workspace.forward_times[k + 1] = DT * nodes[k + 1]
        workspace.adjoint_times[k + 1] = DT * (1 - nodes[k + 1])
    end
    fb = arnoldi_basis!(workspace.forward, A, state; m=M)
    ab = arnoldi_basis!(workspace.adjoint, A', cotangent; m=M)
    fv = _eval_krylov_batch(fb, workspace.forward_times, fb.beta)
    av = _eval_krylov_batch(ab, workspace.adjoint_times, ab.beta)
    cache, kernel = workspace.direction_cache, workspace.kernel_values
    fill!(kernel, zero(eltype(kernel)))
    for coordinate in eachindex(kernel)
        row, column = cache.rows[coordinate], cache.cols[coordinate]
        kernel[coordinate] += weights[1] * conj(av[row, 1]) * state[column]
        for k in 1:interior
            kernel[coordinate] += weights[k + 1] * conj(av[row, k + 1]) * fv[column, k + 1]
        end
        kernel[coordinate] += weights[end] * conj(cotangent[row]) * fv[column, 1]
    end
    mul!(workspace.projected_values, transpose(cache.coefficients), kernel)
    workspace.gradient .= real.(DT .* workspace.projected_values)
    (prediction=view(fv, :, 1), parameter_vjp=workspace.gradient,
     state_vjp=view(av, :, 1))
end

function full_core!(fw, aw, cache, A, dirs, state, cotangent)
    fb = arnoldi_basis!(fw, A, state; m=M)
    ab = arnoldi_basis!(aw, A', cotangent; m=M)
    prediction = _eval_krylov_batch(fb, [DT], fb.beta)
    state_vjp = _eval_krylov_batch(ab, [DT], ab.beta)
    gradient = _compressed_core_from_bases(fb, ab, dirs, DT;
        core_rtol=0.0, core_atol=0.0, direction_cache=cache)
    (prediction=view(prediction, :, 1), parameter_vjp=gradient,
     state_vjp=view(state_vjp, :, 1))
end

function compressed_full_core!(fw, aw, cache, A, dirs, state, cotangent, rtol)
    fb = arnoldi_basis!(fw, A, state; m=M)
    ab = arnoldi_basis!(aw, A', cotangent; m=M)
    prediction = _eval_krylov_batch(fb, [DT], fb.beta)
    state_vjp = _eval_krylov_batch(ab, [DT], ab.beta)
    diagnostics = _compressed_core_from_bases(fb, ab, dirs, DT;
        core_rtol=rtol, core_atol=0.0, direction_cache=cache,
        return_diagnostics=true)
    (prediction=view(prediction, :, 1), parameter_vjp=diagnostics.gradient,
     state_vjp=view(state_vjp, :, 1), core_rank=diagnostics.core_rank,
     core_relative_tail=diagnostics.core_relative_tail)
end

second, fourth = periodic_derivatives(N, DOMAIN_LENGTH)
A = sparse(-second - fourth)
xs = collect(range(0, DOMAIN_LENGTH; length=N + 1))[1:end-1]
phase = 2pi .* xs ./ DOMAIN_LENGTH
state = 0.72 .* cos.(8 .* phase) .+ 0.22 .* sin.(13 .* phase) .- 0.10 .* cos.(21 .* phase)
cotangent = 0.55 .* sin.(7 .* phase .+ 0.2) .+ 0.31 .* cos.(17 .* phase .- 0.4)

rows = NamedTuple[]
for library_size in LIBRARY_SIZES
    dirs = direction_library(xs, second, fourth, library_size)
    cache = SparseDirectionCache(dirs)
    fw = ArnoldiWorkspace(Float64, N, M)
    aw = ArnoldiWorkspace(Float64, N, M)
    core = () -> full_core!(fw, aw, cache, A, dirs, state, cotangent)
    cref = core()
    core_ms = median_ms(core)
    for rtol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16)
        compressed = () -> compressed_full_core!(fw, aw, cache, A, dirs,
            state, cotangent, rtol)
        cvalue = compressed()
        compressed_ms = median_ms(compressed)
        compressed_error = norm(cvalue.parameter_vjp - cref.parameter_vjp) /
                           max(norm(cref.parameter_vjp), eps())
        @printf("M=%3d core rtol=%5.0e rank=%2d time=%8.3f ms ratio(full/compressed)=%5.2f grad.err=%.2e\n",
            library_size, rtol, cvalue.core_rank, compressed_ms,
            core_ms / compressed_ms, compressed_error)
    end
    for order in ORDERS
        x, weights = gausslobatto(order)
        nodes = (x .+ 1) ./ 2
        weights ./= 2
        workspace = OptimizedQuadratureWorkspace(dirs, N, M, order - 2)
        quadrature = () -> full_lobatto!(workspace, A, state, cotangent, nodes, weights)
        qvalue = quadrature()
        quadrature_ms = median_ms(quadrature)
        gradient_error = norm(qvalue.parameter_vjp - cref.parameter_vjp) /
                         max(norm(cref.parameter_vjp), eps())
        state_error = norm(qvalue.state_vjp - cref.state_vjp) /
                      max(norm(cref.state_vjp), eps())
        prediction_error = norm(qvalue.prediction - cref.prediction) /
                           max(norm(cref.prediction), eps())
        push!(rows, (n=N, library_size=library_size, krylov_depth=M,
            quadrature_rule="Gauss-Lobatto", quadrature_order=order,
            repeats=REPEATS, quadrature_full_layer_ms=quadrature_ms,
            core_full_layer_ms=core_ms, core_over_quadrature=core_ms / quadrature_ms,
            parameter_vjp_relative_difference=gradient_error,
            state_vjp_relative_difference=state_error,
            prediction_relative_difference=prediction_error))
        @printf("M=%3d Lobatto=%d quad=%8.3f ms core=%8.3f ms ratio=%5.2f grad.err=%.2e\n",
            library_size, order, quadrature_ms, core_ms, core_ms / quadrature_ms,
            gradient_error)
    end
end

mkpath(dirname(OUTPUT))
CSV.write(OUTPUT, DataFrame(rows))
println("Saved ", OUTPUT)
