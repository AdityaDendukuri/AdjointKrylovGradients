"""Portable optimized Julia CPU scaling sweep on a local sparse generator.

The mathematical problem matches ``portable_iso_scaling.py``.  This driver
records the optimized Julia quadrature path so its CPU results can be joined
with Matfree and the JAX implementations from the same machine.
"""

using CSV, DataFrames
using LinearAlgebra, SparseArrays, Printf, Statistics
using ExponentialUtilities: expv_timestep

include("../src/shared_krylov.jl")
include("optimized_shared_krylov_utils.jl")

parse_list(name, default, type) =
    parse.(type, split(get(ENV, name, default), ","))

const N_VALUES = parse_list(
    "JULIA_SCALING_N_VALUES", "80,160,320", Int,
)
const DIRECTION_VALUES = parse_list(
    "JULIA_SCALING_DIRECTION_VALUES", "10,100", Int,
)
const M_VALUES = parse_list(
    "JULIA_SCALING_M_VALUES", "20,30,40,50,60,80", Int,
)
const NQ_VALUES = parse_list(
    "JULIA_SCALING_NQ_VALUES", "8,16,24", Int,
)
const DT = parse(Float64, get(ENV, "JULIA_SCALING_DT", "0.3"))
const REPEATS = parse(Int, get(ENV, "JULIA_SCALING_REPEATS", "31"))
const WARMUPS = parse(Int, get(ENV, "JULIA_SCALING_WARMUPS", "3"))
const REFERENCE_NQ = parse(
    Int, get(ENV, "JULIA_SCALING_REFERENCE_NQ", "64"),
)
const REFERENCE_CHECK_NQ = parse(
    Int, get(ENV, "JULIA_SCALING_REFERENCE_CHECK_NQ", "48"),
)
const RUN_LABEL = get(ENV, "JULIA_SCALING_RUN_LABEL", "local_julia_cpu")
const OUTPUT = get(
    ENV,
    "JULIA_SCALING_OUTPUT",
    joinpath(
        @__DIR__, "..", "benchmark_results", "portable_iso",
        "portable_iso_$(RUN_LABEL).csv",
    ),
)

relative_error(value, reference) =
    norm(Float64.(value) - Float64.(reference)) /
    max(norm(Float64.(reference)), 1e-14)

function gauss_legendre_unit(order::Int)
    off_diagonal = [
        index / sqrt(4 * index^2 - 1) for index in 1:(order - 1)
    ]
    decomposition = eigen(SymTridiagonal(zeros(order), off_diagonal))
    (decomposition.values .+ 1) ./ 2,
        vec(abs2.(decomposition.vectors[1, :]))
end

function make_local_generator_library(n::Int, number_of_directions::Int)
    lower_rate = 35.0
    upper_rate = 65.0
    lower = fill(lower_rate, n - 1)
    upper = fill(upper_rate, n - 1)
    main = fill(-(lower_rate + upper_rate), n)
    main[1] = -lower_rate
    main[end] = -upper_rate
    base = spdiagm(-1 => lower, 0 => main, 1 => upper)

    directions = SparseMatrixCSC{Float64,Int}[]
    sizehint!(directions, number_of_directions)
    for zero_index in 0:(number_of_directions - 1)
        edge = (zero_index ÷ 2) % (n - 1)
        generation = zero_index ÷ (2 * (n - 1))
        amplitude = 0.5 +
                    0.05 * ((7 * zero_index + generation) % 11)
        if iseven(zero_index)
            direction = sparse(
                [edge + 1, edge + 2],
                [edge + 1, edge + 1],
                [-amplitude, amplitude],
                n,
                n,
            )
        else
            direction = sparse(
                [edge + 2, edge + 1],
                [edge + 2, edge + 2],
                [-amplitude, amplitude],
                n,
                n,
            )
        end
        push!(directions, direction)
    end
    base, directions
end

function semigroup_reference(A, directions, state, upstream, dt, order)
    nodes, weights = gauss_legendre_unit(order)
    n = length(state)
    forward_path = Matrix{Float64}(undef, n, order)
    adjoint_path = Matrix{Float64}(undef, n, order)
    reference_m = min(40, n)
    for index in eachindex(nodes)
        forward_path[:, index] = expv_timestep(
            dt * nodes[index],
            A,
            state;
            m=reference_m,
            tol=1e-13,
            adaptive=true,
            ishermitian=false,
        )
        adjoint_path[:, index] = expv_timestep(
            dt * (1 - nodes[index]),
            A',
            upstream;
            m=reference_m,
            tol=1e-13,
            adaptive=true,
            ishermitian=false,
        )
    end
    cache = SparseDirectionCache(directions)
    kernel = zeros(Float64, length(cache.rows))
    @inbounds for quadrature_index in eachindex(weights)
        for coordinate in eachindex(kernel)
            kernel[coordinate] +=
                weights[quadrature_index] *
                adjoint_path[cache.rows[coordinate], quadrature_index] *
                forward_path[cache.cols[coordinate], quadrature_index]
        end
    end
    parameter_vjp = dt .* (transpose(cache.coefficients) * kernel)
    prediction = expv_timestep(
        dt,
        A,
        state;
        m=reference_m,
        tol=1e-13,
        adaptive=true,
        ishermitian=false,
    )
    state_vjp = expv_timestep(
        dt,
        A',
        upstream;
        m=reference_m,
        tol=1e-13,
        adaptive=true,
        ishermitian=false,
    )
    (
        loss=real(dot(upstream, prediction)),
        parameter_vjp=parameter_vjp,
        state_vjp=state_vjp,
    )
end

function timing_summary(function_to_time)
    for _ in 1:WARMUPS
        function_to_time()
    end
    samples = Vector{Float64}(undef, REPEATS)
    result = function_to_time()
    for index in eachindex(samples)
        start = time_ns()
        result = function_to_time()
        samples[index] = (time_ns() - start) / 1e6
    end
    result, (
        runtime_median_ms=median(samples),
        runtime_q25_ms=quantile(samples, 0.25),
        runtime_q75_ms=quantile(samples, 0.75),
        runtime_min_ms=minimum(samples),
    )
end

BLAS.set_num_threads(1)
rows = NamedTuple[]
for n in N_VALUES
    coordinates = collect(range(-2.0, 2.0; length=n))
    state = exp.(-0.5 .* coordinates .^ 2)
    state ./= norm(state)
    upstream = sin.(1.7 .* coordinates) .+
               0.2 .* cos.(3.1 .* coordinates)
    upstream ./= norm(upstream)

    for number_of_directions in DIRECTION_VALUES
        base, directions = make_local_generator_library(
            n, number_of_directions,
        )
        reference = semigroup_reference(
            base,
            directions,
            state,
            upstream,
            DT,
            REFERENCE_NQ,
        )
        reference_check = semigroup_reference(
            base,
            directions,
            state,
            upstream,
            DT,
            REFERENCE_CHECK_NQ,
        )
        reference_self_error = max(
            relative_error(
                reference_check.parameter_vjp,
                reference.parameter_vjp,
            ),
            relative_error(
                reference_check.state_vjp,
                reference.state_vjp,
            ),
        )

        for requested_m in M_VALUES
            requested_m <= n || continue
            for order in NQ_VALUES
                nodes, weights = gauss_legendre_unit(order)
                workspace = OptimizedQuadratureWorkspace(
                    directions, n, requested_m, order,
                )
                run = () -> optimized_linear_loss_and_vjp!(
                    workspace,
                    base,
                    state,
                    upstream,
                    DT,
                    nodes,
                    weights;
                    m_krylov=requested_m,
                )
                result, timings = timing_summary(run)
                allocated_bytes = @allocated run()
                parameter_error = relative_error(
                    result.parameter_vjp,
                    reference.parameter_vjp,
                )
                state_error = relative_error(
                    result.state_vjp,
                    reference.state_vjp,
                )
                row = (
                    run_label=RUN_LABEL,
                    backend="julia_cpu",
                    device_kind=Sys.CPU_NAME,
                    method="Julia_optimized_semigroup_quadrature",
                    system="local_birth_death_generator",
                    precision="float64",
                    n=n,
                    M=number_of_directions,
                    operator_nnz=nnz(base),
                    direction_nnz=sum(nnz, directions),
                    dt=DT,
                    m=requested_m,
                    core_rank=missing,
                    n_quad=order,
                    loss_abs_error=abs(result.loss - reference.loss),
                    parameter_vjp_relerr=parameter_error,
                    state_vjp_relerr=state_error,
                    max_vjp_relerr=max(parameter_error, state_error),
                    reference_self_relerr=reference_self_error,
                    timings...,
                    compile_seconds=NaN,
                    repeats=REPEATS,
                    warmups=WARMUPS,
                    reference_steps=REFERENCE_NQ,
                    allocated_bytes=allocated_bytes,
                    julia_version=string(VERSION),
                    julia_threads=Threads.nthreads(),
                )
                push!(rows, row)
                @printf(
                    "n=%5d M=%5d m=%3d nq=%2d error=%9.2e time=%8.3f ms\n",
                    n,
                    number_of_directions,
                    requested_m,
                    order,
                    row.max_vjp_relerr,
                    row.runtime_median_ms,
                )
            end
        end
    end
end

mkpath(dirname(OUTPUT))
CSV.write(OUTPUT, DataFrame(rows))
println("Saved $(OUTPUT)")
