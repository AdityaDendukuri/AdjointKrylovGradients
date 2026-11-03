"""Pilot benchmark for sparse-direct rational exponential gradients.

The experiment compares the CRAM16/LU reverse-mode implementation with the
optimized shared polynomial-Krylov implementation on the same sparse generator,
library, state, and cotangent.  Accuracy is measured against a dense block
exponential, not against either competing method.

Run from the repository root:

    julia --project=experiments experiments/sparse_direct_rational_benchmark.jl
"""

using LinearAlgebra, SparseArrays, Statistics, Random, Printf, CSV, DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "shared_krylov.jl"))
include(joinpath(ROOT, "src", "sparse_direct_rational.jl"))
include(joinpath(@__DIR__, "optimized_shared_krylov_utils.jl"))
BLAS.set_num_threads(1)

const OUTPUT = get(
    ENV, "SPARSE_DIRECT_OUTPUT",
    joinpath(ROOT, "benchmark_results", "sparse_direct_rational_pilot.csv"),
)
const REPEATS = parse(Int, get(ENV, "SPARSE_DIRECT_REPEATS", "7"))
const KRYLOV_DEPTHS = parse.(
    Int, split(get(ENV, "SPARSE_DIRECT_DEPTHS", "12,20,30,40"), ','),
)
const QUADRATURE_ORDERS = parse.(
    Int, split(get(ENV, "SPARSE_DIRECT_ORDERS", "8,16,24"), ','),
)

relative_error(x, reference) =
    norm(x - reference) / max(norm(reference), eps(Float64))

function timing(f)
    f()
    samples = [(@elapsed f()) * 1e3 for _ in 1:REPEATS]
    (
        median_ms=median(samples),
        q25_ms=quantile(samples, 0.25),
        q75_ms=quantile(samples, 0.75),
    )
end

function birth_death_problem(n, library_size)
    rng = MersenneTwister(20261010 + n + library_size)
    lower = 15 .+ 5 .* rand(rng, n - 1)
    upper = 12 .+ 5 .* rand(rng, n - 1)
    A = spdiagm(
        -1 => lower,
         0 => -vcat(lower, 0.0) .- vcat(0.0, upper),
         1 => upper,
    )
    candidates = SparseMatrixCSC{Float64,Int}[]
    for index in randperm(rng, 2(n - 1))[1:library_size]
        column = index <= n - 1 ? index : index - (n - 1) + 1
        row = index <= n - 1 ? column + 1 : column - 1
        push!(
            candidates,
            sparse([row, column], [column, column], [1.0, -1.0], n, n),
        )
    end
    v = rand(rng, n); v ./= norm(v)
    upstream = randn(rng, n); upstream ./= norm(upstream)
    A, candidates, v, upstream
end

function dense_reference(A, directions, v, upstream, dt)
    n = size(A, 1)
    zero_block = zeros(n, n)
    block = [Matrix(A') upstream * v'; zero_block Matrix(A')]
    frechet_adjoint = exp(dt .* block)[1:n, n+1:2n]
    (
        prediction=exp(dt .* Matrix(A)) * v,
        state_vjp=exp(dt .* Matrix(A')) * upstream,
        parameter_vjp=[dot(frechet_adjoint, direction) for direction in directions],
    )
end

function main()
    n = parse(Int, get(ENV, "SPARSE_DIRECT_N", "320"))
    dt = parse(Float64, get(ENV, "SPARSE_DIRECT_DT", "0.3"))
    maximum_library = min(parse(Int, get(ENV, "SPARSE_DIRECT_MAX_LIBRARY", "512")), 2(n - 1))
    A, all_directions, v, upstream =
        birth_death_problem(n, maximum_library)
    reference = dense_reference(A, all_directions, v, upstream, dt)
    rows = NamedTuple[]

    library_sizes = filter(<=(maximum_library), [16, 64, 256, 512])
    for library_size in library_sizes
        directions = all_directions[1:library_size]
        reference_gradient = reference.parameter_vjp[1:library_size]
        cache_time = @elapsed direction_cache = SparseDirectionCache(directions)

        direct = () -> sparse_direct_exponential_vjp(
            A, direction_cache, v, dt, upstream,
        )
        direct_result = direct()
        direct_times = timing(direct)
        push!(rows, (
            n=n, library_size=library_size, method="sparse_direct_cram16",
            krylov_depth=0, quadrature_order=0,
            parameter_error=relative_error(
                direct_result.parameter_vjp, reference_gradient,
            ),
            forward_error=relative_error(
                direct_result.prediction, reference.prediction,
            ),
            state_error=relative_error(
                direct_result.state_vjp, reference.state_vjp,
            ),
            median_ms=direct_times.median_ms,
            q25_ms=direct_times.q25_ms,
            q75_ms=direct_times.q75_ms,
            setup_ms=cache_time * 1e3,
            factorizations=direct_result.factorizations,
            repeats=REPEATS,
        ))

        for depth in KRYLOV_DEPTHS, order in QUADRATURE_ORDERS
            setup_time = @elapsed workspace = OptimizedQuadratureWorkspace(
                directions, n, depth, order,
            )
            nodes, weights = gauss_legendre_01(order)
            krylov = () -> optimized_linear_loss_and_vjp!(
                workspace, A, v, upstream, dt, nodes, weights;
                m_krylov=depth,
            )
            result = krylov()
            times = timing(krylov)
            push!(rows, (
                n=n, library_size=library_size, method="shared_krylov",
                krylov_depth=depth, quadrature_order=order,
                parameter_error=relative_error(
                    result.parameter_vjp, reference_gradient,
                ),
                forward_error=relative_error(
                    result.prediction, reference.prediction,
                ),
                state_error=relative_error(
                    result.state_vjp, reference.state_vjp,
                ),
                median_ms=times.median_ms,
                q25_ms=times.q25_ms,
                q75_ms=times.q75_ms,
                setup_ms=setup_time * 1e3,
                factorizations=0,
                repeats=REPEATS,
            ))
        end
        @printf("M=%d complete\n", library_size)
        flush(stdout)
    end

    data = DataFrame(rows)
    mkpath(dirname(OUTPUT))
    CSV.write(OUTPUT, data)

    println("\nFastest configurations with all three errors <= 1e-8:")
    for library_size in library_sizes
        eligible = filter(
            row -> row.library_size == library_size &&
                maximum((row.parameter_error, row.forward_error, row.state_error)) <= 1e-8,
            data,
        )
        for method in ("sparse_direct_cram16", "shared_krylov")
            candidates = filter(row -> row.method == method, eligible)
            isempty(candidates) && continue
            row = candidates[argmin(candidates.median_ms), :]
            @printf(
                "M=%3d %-22s %8.3f ms  m=%2d nq=%2d err=%.2e\n",
                library_size, method, row.median_ms, row.krylov_depth,
                row.quadrature_order,
                maximum((row.parameter_error, row.forward_error, row.state_error)),
            )
        end
    end
    println("saved ", OUTPUT)
end

main()
