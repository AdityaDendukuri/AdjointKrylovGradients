"""Optimized Julia CPU iso-accuracy sweep on the matched FPE problem.

This driver differs from ``iso_accuracy_shared_krylov.jl`` only in execution
organization.  It preassembles the evaluated generator, retains Arnoldi and
library-projection workspaces, evaluates each reduced Hessenberg matrix once
at the endpoint and quadrature nodes, and fuses all sparse directions through
their union pattern.  It also includes the full-dimensional ``m=n`` case.
"""

using CSV, DataFrames
using LinearAlgebra, SparseArrays, Printf, Statistics

include("../src/fpe.jl")
include("../src/shared_krylov.jl")
include("optimized_shared_krylov_utils.jl")

parse_list(name, default, type) =
    parse.(type, split(get(ENV, name, default), ","))

const M_VALUES = parse_list(
    "OPT_ISO_M_VALUES",
    "30,35,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,79,80",
    Int,
)
const NQ_VALUES = parse_list(
    "OPT_ISO_NQ_VALUES", "8,12,16,20,24,32,40,48", Int,
)
const DT_VALUES = parse_list(
    "OPT_ISO_DT_VALUES", "0.3,1.0", Float64,
)
const REPEATS = parse(Int, get(ENV, "OPT_ISO_REPEATS", "51"))
const OUTPUT = get(
    ENV,
    "OPT_ISO_OUTPUT",
    joinpath(
        @__DIR__, "..", "paper", "figures",
        "iso_accuracy_shared_krylov_optimized_raw.csv",
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

function exact_reference(A0, dirs, theta_true, theta_eval, initial, dt)
    matrix_true = Matrix(
        A0 + sum(theta_true[j] * dirs[j] for j in eachindex(dirs)),
    )
    matrix_eval = Matrix(
        A0 + sum(theta_eval[j] * dirs[j] for j in eachindex(dirs)),
    )
    target = exp(matrix_true * dt) * initial
    prediction = exp(matrix_eval * dt) * initial
    upstream = prediction - target
    n = length(initial)
    zero_block = zeros(Float64, n, n)
    parameter_vjp = [
        dot(
            upstream,
            exp([
                matrix_eval * dt Matrix(direction) * dt
                zero_block matrix_eval * dt
            ])[1:n, (n + 1):(2n)] * initial,
        ) for direction in dirs
    ]
    state_vjp = exp(matrix_eval' * dt) * upstream
    (; target, prediction, parameter_vjp, state_vjp)
end

function median_runtime_ms(function_to_time)
    function_to_time()
    samples = Vector{Float64}(undef, REPEATS)
    for index in eachindex(samples)
        start = time_ns()
        function_to_time()
        samples[index] = (time_ns() - start) / 1e6
    end
    median(samples)
end

BLAS.set_num_threads(1)
n = 80
number_of_directions = 10
coordinates = collect(range(-2.0, 2.0; length=n))
drift_functions = [
    let power=index
        coordinate -> (coordinate / 2.0)^power
    end for index in 1:number_of_directions
]
base, directions = fpe_library_1d(
    coordinates, drift_functions; D=0.5,
)
theta_true = zeros(number_of_directions)
theta_true[1] = -4.0
theta_true[3] = 4.0
theta_eval = theta_true +
             0.08 .* collect(range(-1.0, 1.0; length=number_of_directions))
initial = exp.(-0.5 .* coordinates .^ 2)
initial ./= sum(initial)
evaluated_generator =
    base + sum(theta_eval[j] * directions[j] for j in eachindex(directions))

quadrature_rules = Dict(
    order => gauss_legendre_unit(order) for order in NQ_VALUES
)
rows = NamedTuple[]
for dt in DT_VALUES
    reference = exact_reference(
        base, directions, theta_true, theta_eval, initial, dt,
    )
    for requested_m in M_VALUES
        for order in NQ_VALUES
            nodes, weights = quadrature_rules[order]
            workspace = OptimizedQuadratureWorkspace(
                directions, n, requested_m, order,
            )
            run = () -> optimized_loss_and_vjp!(
                workspace,
                evaluated_generator,
                initial,
                reference.target,
                dt,
                nodes,
                weights;
                m_krylov=requested_m,
            )
            result = run()
            runtime_ms = median_runtime_ms(run)
            allocated_bytes = @allocated run()
            forward_error = relative_error(
                result.prediction, reference.prediction,
            )
            parameter_error = relative_error(
                result.parameter_vjp, reference.parameter_vjp,
            )
            state_error = relative_error(
                result.state_vjp, reference.state_vjp,
            )
            row = (
                backend="Julia_CPU",
                method="shared_krylov_quadrature_optimized",
                variant="GL$(order)",
                system="FPE",
                precision="float64",
                n=n,
                M=number_of_directions,
                dt=dt,
                requested_m=requested_m,
                forward_m=result.forward_dimension,
                adjoint_m=result.adjoint_dimension,
                n_quad=order,
                forward_relerr=forward_error,
                parameter_vjp_relerr=parameter_error,
                state_vjp_relerr=state_error,
                max_vjp_relerr=max(parameter_error, state_error),
                overall_relerr=max(
                    forward_error, parameter_error, state_error,
                ),
                loss_and_full_vjp_ms=runtime_ms,
                allocated_bytes=allocated_bytes,
                repeats=REPEATS,
                reorthogonalization="single_MGS_pass",
                gradient_semantics="approximate_exact_semigroup_VJP",
                timing_scope=(
                    "preassembled_generator_persistent_workspaces_" *
                    "fused_library_projection"
                ),
                platform="$(Sys.KERNEL)-$(Sys.ARCH)-cpu",
                cpu_name=Sys.CPU_NAME,
                julia_threads=Threads.nthreads(),
            )
            push!(rows, row)
            @printf(
                "optimized dt=%3.1f m=%2d nq=%2d theta=%9.2e state=%9.2e time=%7.3f ms alloc=%d\n",
                dt,
                requested_m,
                order,
                parameter_error,
                state_error,
                runtime_ms,
                allocated_bytes,
            )
        end
    end
end

mkpath(dirname(OUTPUT))
CSV.write(OUTPUT, DataFrame(rows))
println("Saved $(OUTPUT)")
