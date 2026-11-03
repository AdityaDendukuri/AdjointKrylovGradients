"""
Iso-accuracy sweep for the Julia shared-Krylov reverse rule.

The benchmark uses the same 80-state Fokker--Planck generator, loss, and
dense Float64 reference as `kraemer_comparison.jl`.  It adds a fine Krylov
dimension sweep and arbitrary Gauss--Legendre orders without changing the
submitted implementation.  Each timing contains the forward prediction,
loss, parameter VJP, and input-state VJP.  Setup and warmup are excluded.

Outputs:
  paper/figures/iso_accuracy_shared_krylov_raw.csv

Environment variables:
  ISO_M_VALUES       comma-separated Krylov dimensions
  ISO_NQ_VALUES      comma-separated Gauss--Legendre orders
  ISO_CORE_RTOLS     comma-separated projected-core tolerances
  ISO_DT_VALUES      comma-separated time steps
  ISO_REPEATS        timing repetitions after warmup
  ISO_JULIA_OUTPUT   output CSV path
"""

using CSV, DataFrames
using LinearAlgebra, SparseArrays, Printf, Statistics

include("../src/fpe.jl")
include("../src/shared_krylov.jl")

parse_list(name, default, type) =
    parse.(type, split(get(ENV, name, default), ","))

const ISO_M_VALUES = parse_list(
    "ISO_M_VALUES",
    "5,10,15,20,25,30,35,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,79",
    Int,
)
const ISO_NQ_VALUES = parse_list(
    "ISO_NQ_VALUES", "4,5,6,8,10,12,16,20,24", Int,
)
const ISO_CORE_RTOLS = parse_list(
    "ISO_CORE_RTOLS", "1e-3,1e-5,1e-7,0", Float64,
)
const ISO_DT_VALUES = parse_list(
    "ISO_DT_VALUES", "0.3,1.0", Float64,
)
const ISO_REPEATS = parse(Int, get(ENV, "ISO_REPEATS", "21"))
const ISO_JULIA_OUTPUT = get(
    ENV,
    "ISO_JULIA_OUTPUT",
    joinpath(
        @__DIR__, "..", "paper", "figures",
        "iso_accuracy_shared_krylov_raw.csv",
    ),
)

relative_error(value, reference) =
    norm(Float64.(value) - Float64.(reference)) /
    max(norm(Float64.(reference)), 1e-14)

"""Return Gauss--Legendre nodes and weights on [0, 1]."""
function gauss_legendre_unit(order::Int)
    order > 0 || throw(ArgumentError("quadrature order must be positive"))
    if order == 1
        return [0.5], [1.0]
    end
    off_diagonal = [
        index / sqrt(4 * index^2 - 1) for index in 1:(order - 1)
    ]
    decomposition = eigen(SymTridiagonal(zeros(order), off_diagonal))
    nodes = (decomposition.values .+ 1) ./ 2
    weights = vec(abs2.(decomposition.vectors[1, :]))
    @assert isapprox(sum(weights), 1.0; rtol=1e-14, atol=1e-14)
    nodes, weights
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
    adjoint_block = matrix_eval' * dt
    direction_seed = dt .* (upstream * initial')
    zero_block = zeros(Float64, n, n)
    frechet_adjoint = exp([
        adjoint_block direction_seed
        zero_block adjoint_block
    ])[1:n, (n + 1):(2n)]
    parameter_vjp = [
        real(dot(frechet_adjoint, Matrix(direction))) for direction in dirs
    ]
    state_vjp = exp(matrix_eval' * dt) * upstream
    (; target, prediction, parameter_vjp, state_vjp)
end

function fused_quadrature_loss_and_vjp(
    A0,
    dirs,
    theta,
    initial,
    target,
    dt,
    nodes,
    weights;
    m_krylov,
)
    matrix = A0 + sum(theta[j] * dirs[j] for j in eachindex(dirs))
    state_type = eltype(initial)
    real_type = real_scalar_type(state_type)
    node_values = real_type.(nodes)
    weight_values = real_type.(weights)

    forward_workspace = ArnoldiWorkspace(
        state_type, length(initial), m_krylov,
    )
    adjoint_workspace = ArnoldiWorkspace(
        state_type, length(initial), m_krylov,
    )
    forward_basis = arnoldi_basis!(
        forward_workspace, matrix, initial; m=m_krylov,
    )
    prediction = _eval_krylov_batch(
        forward_basis, real_type[dt], forward_basis.beta,
    )[:, 1]
    upstream = prediction - target
    loss = real_type(0.5) * sum(abs2, upstream)

    adjoint_basis = arnoldi_basis!(
        adjoint_workspace, matrix', upstream; m=m_krylov,
    )
    state_vjp = _eval_krylov_batch(
        adjoint_basis, real_type[dt], adjoint_basis.beta,
    )[:, 1]
    forward_path = _eval_krylov_batch(
        forward_basis,
        node_values .* dt,
        forward_basis.beta,
    )
    adjoint_path = _eval_krylov_batch(
        adjoint_basis,
        (one(real_type) .- node_values) .* dt,
        adjoint_basis.beta,
    )

    buffer = zeros(state_type, length(initial))
    parameter_vjp = zeros(real_type, length(dirs))
    for j in eachindex(dirs)
        value = zero(state_type)
        @inbounds for q in eachindex(node_values)
            mul!(buffer, dirs[j], view(forward_path, :, q))
            value += weight_values[q] *
                     dot(view(adjoint_path, :, q), buffer)
        end
        parameter_vjp[j] = real(dt * value)
    end

    (
        loss=loss,
        prediction=prediction,
        parameter_vjp=parameter_vjp,
        state_vjp=state_vjp,
        forward_dimension=forward_basis.m,
        adjoint_dimension=adjoint_basis.m,
        core_rank=missing,
        core_full_rank=missing,
        core_relative_tail=NaN,
    )
end

function fused_core_loss_and_vjp(
    A0,
    dirs,
    theta,
    initial,
    target,
    dt;
    m_krylov,
    core_rtol,
    direction_cache,
)
    matrix = A0 + sum(theta[j] * dirs[j] for j in eachindex(dirs))
    state_type = eltype(initial)
    real_type = real_scalar_type(state_type)
    forward_workspace = ArnoldiWorkspace(
        state_type, length(initial), m_krylov,
    )
    adjoint_workspace = ArnoldiWorkspace(
        state_type, length(initial), m_krylov,
    )

    forward_basis = arnoldi_basis!(
        forward_workspace, matrix, initial; m=m_krylov,
    )
    prediction = _eval_krylov_batch(
        forward_basis, real_type[dt], forward_basis.beta,
    )[:, 1]
    upstream = prediction - target
    loss = real_type(0.5) * sum(abs2, upstream)
    adjoint_basis = arnoldi_basis!(
        adjoint_workspace, matrix', upstream; m=m_krylov,
    )
    state_vjp = _eval_krylov_batch(
        adjoint_basis, real_type[dt], adjoint_basis.beta,
    )[:, 1]
    diagnostics = _compressed_core_from_bases(
        forward_basis,
        adjoint_basis,
        dirs,
        dt;
        core_rtol=core_rtol,
        direction_cache=direction_cache,
        return_diagnostics=true,
    )

    (
        loss=loss,
        prediction=prediction,
        parameter_vjp=diagnostics.gradient,
        state_vjp=state_vjp,
        forward_dimension=forward_basis.m,
        adjoint_dimension=adjoint_basis.m,
        core_rank=diagnostics.core_rank,
        core_full_rank=diagnostics.core_full_rank,
        core_relative_tail=diagnostics.core_relative_tail,
    )
end

function median_runtime_ms(function_to_time)
    function_to_time()
    samples = Float64[]
    sizehint!(samples, ISO_REPEATS)
    for _ in 1:ISO_REPEATS
        start = time_ns()
        function_to_time()
        push!(samples, (time_ns() - start) / 1e6)
    end
    median(samples)
end

function make_row(
    result,
    reference,
    dt,
    requested_m,
    method,
    variant,
    n_quad,
    core_rtol,
    runtime_ms,
)
    parameter_error = relative_error(
        result.parameter_vjp, reference.parameter_vjp,
    )
    state_error = relative_error(result.state_vjp, reference.state_vjp)
    forward_error = relative_error(result.prediction, reference.prediction)
    (
        backend="Julia_CPU",
        method=method,
        variant=variant,
        system="FPE",
        precision="float64",
        n=length(result.prediction),
        M=length(result.parameter_vjp),
        dt=dt,
        requested_m=requested_m,
        forward_m=result.forward_dimension,
        adjoint_m=result.adjoint_dimension,
        n_quad=n_quad,
        core_rtol=core_rtol,
        core_rank=result.core_rank,
        core_full_rank=result.core_full_rank,
        core_relative_tail=result.core_relative_tail,
        forward_relerr=forward_error,
        parameter_vjp_relerr=parameter_error,
        state_vjp_relerr=state_error,
        max_vjp_relerr=max(parameter_error, state_error),
        overall_relerr=max(forward_error, parameter_error, state_error),
        loss_and_full_vjp_ms=runtime_ms,
        repeats=ISO_REPEATS,
        reorthogonalization="single_MGS_pass",
        gradient_semantics="approximate_exact_semigroup_VJP",
        platform="$(Sys.KERNEL)-$(Sys.ARCH)-cpu",
        cpu_name=Sys.CPU_NAME,
    )
end

BLAS.set_num_threads(1)

n = 80
number_of_directions = 10
xs = collect(range(-2.0, 2.0; length=n))
drift_functions = [
    let power=j
        x -> (x / 2.0)^power
    end for j in 1:number_of_directions
]
base, directions = fpe_library_1d(xs, drift_functions; D=0.5)
theta_true = zeros(number_of_directions)
theta_true[1] = -4.0
theta_true[3] = 4.0
theta_eval = theta_true +
             0.08 .* collect(range(-1.0, 1.0; length=number_of_directions))
initial = exp.(-0.5 .* xs .^ 2)
initial ./= sum(initial)
direction_cache = SparseDirectionCache(directions)
quadrature_rules = Dict(
    order => gauss_legendre_unit(order) for order in ISO_NQ_VALUES
)

rows = NamedTuple[]
for dt in ISO_DT_VALUES
    reference = exact_reference(
        base, directions, theta_true, theta_eval, initial, dt,
    )
    for requested_m in ISO_M_VALUES
        for order in ISO_NQ_VALUES
            nodes, weights = quadrature_rules[order]
            run = () -> fused_quadrature_loss_and_vjp(
                base,
                directions,
                theta_eval,
                initial,
                reference.target,
                dt,
                nodes,
                weights;
                m_krylov=requested_m,
            )
            result = run()
            runtime_ms = median_runtime_ms(run)
            row = make_row(
                result,
                reference,
                dt,
                requested_m,
                "shared_krylov_quadrature",
                "GL$(order)",
                order,
                NaN,
                runtime_ms,
            )
            push!(rows, row)
            @printf(
                "Julia GL%-2d dt=%3.1f m=%2d theta=%9.2e state=%9.2e time=%7.3f ms\n",
                order,
                dt,
                requested_m,
                row.parameter_vjp_relerr,
                row.state_vjp_relerr,
                runtime_ms,
            )
        end

        for core_rtol in ISO_CORE_RTOLS
            run = () -> fused_core_loss_and_vjp(
                base,
                directions,
                theta_eval,
                initial,
                reference.target,
                dt;
                m_krylov=requested_m,
                core_rtol=core_rtol,
                direction_cache=direction_cache,
            )
            result = run()
            runtime_ms = median_runtime_ms(run)
            variant = iszero(core_rtol) ?
                "projected_core_full" :
                "projected_core_rtol_$(core_rtol)"
            row = make_row(
                result,
                reference,
                dt,
                requested_m,
                "projected_core",
                variant,
                missing,
                core_rtol,
                runtime_ms,
            )
            push!(rows, row)
            @printf(
                "Julia core dt=%3.1f m=%2d rank=%2d rtol=%7.0e theta=%9.2e state=%9.2e time=%7.3f ms\n",
                dt,
                requested_m,
                result.core_rank,
                core_rtol,
                row.parameter_vjp_relerr,
                row.state_vjp_relerr,
                runtime_ms,
            )
        end
    end
end

mkpath(dirname(ISO_JULIA_OUTPUT))
CSV.write(ISO_JULIA_OUTPUT, DataFrame(rows))
println("Saved $(ISO_JULIA_OUTPUT)")
