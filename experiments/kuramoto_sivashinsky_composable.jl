"""
Composable constitutive-parameter learning for the Kuramoto--Sivashinsky PDE.

We identify the coefficients in

    u_t = -c u u_x - a u_xx - b u_xxxx

on a periodic grid.  The exponential layer has the two fixed sparse
directions -Dxx and -Dxxxx, with coefficients a and b.  A conservative
nonlinear update for -c u u_x is applied before the exponential layer, so
the gradient for c must pass through the layer's input-state pullback.

Training uses the first-order integrating-factor step

    u_{k+1} = exp((-a Dxx - b Dxxxx) dt)
              (u_k - c dt D_x(u_k^2) / 2).

Reference trajectories are generated independently with small-step RK4.
The zero-state ablation has the same forward map but stops the pullback
between the nonlinear update and the exponential layer.

Environment variables:
  KS_N                spatial grid size (default: 64)
  KS_LENGTH           domain length (default: 22)
  KS_DT               snapshot interval (default: 0.0125)
  KS_STEPS            training transitions per initial state (default: 10)
  KS_EPOCHS           training epochs (default: 500)
  KS_SEEDS            comma-separated seeds (default: 11,22,33,44,55)
  KS_LR               Adam learning rate (default: 0.02)
  KS_M                Krylov dimension (default: 40)
  KS_NQ               quadrature order, 5 or 8 (default: 8)
  KS_REF_SUBSTEPS      RK4 substeps per snapshot interval (default: 20)
  KS_OUTPUT_DIR        result directory (default: paper/figures)
  KS_SMOKE             use a small fast configuration when set to 1

Outputs:
  kuramoto_sivashinsky_composable_raw.csv
  kuramoto_sivashinsky_composable_summary.csv
  kuramoto_sivashinsky_composable_trajectory.csv
"""

using LinearAlgebra, SparseArrays, Random, Statistics, Printf
using Optimisers, Zygote
using CSV, DataFrames

include("../src/shared_krylov.jl")

const KS_SMOKE = get(ENV, "KS_SMOKE", "0") == "1"
const KS_N = parse(Int, get(ENV, "KS_N", KS_SMOKE ? "32" : "64"))
const KS_LENGTH = parse(Float64, get(ENV, "KS_LENGTH", "22.0"))
const KS_DT = parse(Float64, get(ENV, "KS_DT", "0.0125"))
const KS_STEPS = parse(Int, get(ENV, "KS_STEPS", KS_SMOKE ? "3" : "10"))
const KS_EPOCHS = parse(Int, get(ENV, "KS_EPOCHS", KS_SMOKE ? "5" : "500"))
const KS_SEEDS = parse.(
    Int,
    split(get(ENV, "KS_SEEDS", KS_SMOKE ? "11" : "11,22,33,44,55"), ","),
)
const KS_LR = parse(Float64, get(ENV, "KS_LR", "0.02"))
const KS_M = parse(Int, get(ENV, "KS_M", KS_SMOKE ? "20" : "40"))
const KS_NQ = parse(Int, get(ENV, "KS_NQ", "8"))
const KS_REF_SUBSTEPS = parse(
    Int,
    get(ENV, "KS_REF_SUBSTEPS", KS_SMOKE ? "5" : "20"),
)
const KS_OUTPUT_DIR = get(
    ENV,
    "KS_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)

KS_NQ in (5, 8) || error("KS_NQ must be 5 or 8")
KS_N >= 16 || error("KS_N must be at least 16")
KS_LENGTH > 0 || error("KS_LENGTH must be positive")
KS_DT > 0 || error("KS_DT must be positive")

const TRUE_PARAMETERS = (a=1.0, b=1.0, c=1.0)
const INITIAL_PARAMETERS = (a=0.72, b=1.28, c=0.68)

softplus_inverse(x) = log(expm1(x))
positive_parameters(raw) = softplus.(raw)

function periodic_derivatives(n, domain_length)
    dx = domain_length / n
    first = spdiagm(
        -1 => fill(-1.0 / (2dx), n - 1),
         1 => fill(1.0 / (2dx), n - 1),
    )
    first[1, n] = -1.0 / (2dx)
    first[n, 1] = 1.0 / (2dx)

    second = spdiagm(
        -1 => fill(1.0 / dx^2, n - 1),
         0 => fill(-2.0 / dx^2, n),
         1 => fill(1.0 / dx^2, n - 1),
    )
    second[1, n] = 1.0 / dx^2
    second[n, 1] = 1.0 / dx^2
    fourth = sparse(second * second)
    first, second, fourth
end

function nonlinear_tendency(u, c, first)
    -0.5 .* c .* (first * abs2.(u))
end

function integrating_factor_step(
    layer,
    first,
    u,
    parameters,
    dt;
    stop_input_gradient=false,
)
    a, b, c = parameters
    nonlinear_input = u .+ dt .* nonlinear_tendency(u, c, first)
    linear_input = stop_input_gradient ?
        Zygote.dropgrad(nonlinear_input) : nonlinear_input
    layer(linear_input, [a, b], dt)
end

function rhs(u, first, minus_second, minus_fourth, parameters)
    a, b, c = parameters
    a .* (minus_second * u) .+
        b .* (minus_fourth * u) .+
        nonlinear_tendency(u, c, first)
end

function rk4_interval(
    u,
    first,
    minus_second,
    minus_fourth,
    parameters,
    dt,
    n_substeps,
)
    step = dt / n_substeps
    state = copy(u)
    for _ in 1:n_substeps
        k1 = rhs(state, first, minus_second, minus_fourth, parameters)
        k2 = rhs(
            state .+ (step / 2) .* k1,
            first,
            minus_second,
            minus_fourth,
            parameters,
        )
        k3 = rhs(
            state .+ (step / 2) .* k2,
            first,
            minus_second,
            minus_fourth,
            parameters,
        )
        k4 = rhs(
            state .+ step .* k3,
            first,
            minus_second,
            minus_fourth,
            parameters,
        )
        state = state .+
            (step / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    state
end

function initial_conditions(xs, domain_length)
    phase = 2π .* xs ./ domain_length
    [
        0.72 .* cos.(2 .* phase) .+
            0.22 .* sin.(3 .* phase) .-
            0.10 .* cos.(5 .* phase),
        0.58 .* sin.(2 .* phase .+ 0.3) .-
            0.26 .* cos.(4 .* phase) .+
            0.12 .* sin.(6 .* phase),
        0.64 .* cos.(3 .* phase .- 0.4) .+
            0.18 .* sin.(5 .* phase .+ 0.2),
    ]
end

function heldout_initial(xs, domain_length)
    phase = 2π .* xs ./ domain_length
    0.61 .* sin.(2 .* phase .- 0.5) .+
        0.24 .* cos.(3 .* phase .+ 0.4) .-
        0.11 .* sin.(7 .* phase)
end

function make_reference_trajectories(
    initials,
    first,
    minus_second,
    minus_fourth,
    parameters,
    dt,
    n_steps,
    n_substeps,
)
    trajectories = Vector{Vector{Vector{Float64}}}()
    for initial in initials
        trajectory = [copy(initial)]
        for _ in 1:n_steps
            push!(
                trajectory,
                rk4_interval(
                    trajectory[end],
                    first,
                    minus_second,
                    minus_fourth,
                    parameters,
                    dt,
                    n_substeps,
                ),
            )
        end
        push!(trajectories, trajectory)
    end
    trajectories
end

function snapshot_loss(
    layer,
    first,
    raw,
    trajectories;
    stop_input_gradient=false,
)
    parameters = positive_parameters(raw)
    total = sum(
        0.5 * mean(abs2, trajectory[index + 1] -
            integrating_factor_step(
                layer,
                first,
                trajectory[index],
                parameters,
                KS_DT;
                stop_input_gradient=stop_input_gradient,
            ))
        for trajectory in trajectories
        for index in 1:(length(trajectory) - 1)
    )
    total / sum(length(trajectory) - 1 for trajectory in trajectories)
end

function relative_parameter_error(estimate)
    truth = collect(TRUE_PARAMETERS)
    norm(estimate - truth) / norm(truth)
end

function rollout(layer, first, initial, parameters, n_steps)
    states = [copy(initial)]
    for _ in 1:n_steps
        push!(
            states,
            integrating_factor_step(
                layer,
                first,
                states[end],
                parameters,
                KS_DT,
            ),
        )
    end
    states
end

function trajectory_relative_error(prediction, reference)
    sqrt(
        sum(
            norm(prediction[index] - reference[index])^2
            for index in eachindex(reference)
        ) / sum(norm(state)^2 for state in reference),
    )
end

function train_case(layer, first, trajectories, seed; stop_input_gradient)
    Random.seed!(seed)
    initial = collect(INITIAL_PARAMETERS)
    raw = softplus_inverse.(initial) .+ 0.08 .* randn(3)
    optimizer = Optimisers.setup(Optimisers.Adam(KS_LR), raw)

    initial_gradient = Zygote.gradient(raw) do values
        snapshot_loss(
            layer,
            first,
            values,
            trajectories;
            stop_input_gradient=stop_input_gradient,
        )
    end[1]

    seconds = @elapsed for _ in 1:KS_EPOCHS
        _, gradients = Zygote.withgradient(raw) do values
            snapshot_loss(
                layer,
                first,
                values,
                trajectories;
                stop_input_gradient=stop_input_gradient,
            )
        end
        optimizer, raw = Optimisers.update!(optimizer, raw, gradients[1])
    end

    estimate = positive_parameters(raw)
    (
        estimate=estimate,
        train_loss=snapshot_loss(layer, first, raw, trajectories),
        initial_gradient=initial_gradient,
        train_seconds=seconds,
    )
end

function summarize(raw)
    rows = NamedTuple[]
    for method in unique(raw.method)
        subset = filter(:method => ==(method), raw)
        push!(rows, (
            method=method,
            seeds=nrow(subset),
            n=KS_N,
            domain_length=KS_LENGTH,
            dt=KS_DT,
            steps=KS_STEPS,
            epochs=KS_EPOCHS,
            krylov_m=KS_M,
            n_quad=KS_NQ,
            parameter_relerr_mean=mean(subset.parameter_relerr),
            parameter_relerr_std=std(
                subset.parameter_relerr;
                corrected=false,
            ),
            heldout_rollout_relerr_mean=mean(
                subset.heldout_rollout_relerr,
            ),
            heldout_rollout_relerr_std=std(
                subset.heldout_rollout_relerr;
                corrected=false,
            ),
            train_loss_mean=mean(subset.train_loss),
            train_seconds_mean=mean(subset.train_seconds),
            a_relerr_mean=mean(subset.a_relerr),
            b_relerr_mean=mean(subset.b_relerr),
            c_relerr_mean=mean(subset.c_relerr),
            initial_c_gradient_mean=mean(abs.(subset.initial_c_gradient)),
        ))
    end
    DataFrame(rows)
end

mkpath(KS_OUTPUT_DIR)
dx = KS_LENGTH / KS_N
xs = collect(range(0, KS_LENGTH; length=KS_N + 1))[1:end-1]
first, second, fourth = periodic_derivatives(KS_N, KS_LENGTH)
minus_second = sparse(-second)
minus_fourth = sparse(-fourth)
zero_operator = spzeros(Float64, KS_N, KS_N)
layer = SemigroupLayer(
    zero_operator,
    SparseMatrixCSC{Float64,Int}[minus_second, minus_fourth];
    m_krylov=min(KS_M, KS_N - 1),
    n_quad=KS_NQ,
)
truth = collect(TRUE_PARAMETERS)
training_trajectories = make_reference_trajectories(
    initial_conditions(xs, KS_LENGTH),
    first,
    minus_second,
    minus_fourth,
    truth,
    KS_DT,
    KS_STEPS,
    KS_REF_SUBSTEPS,
)
test_steps = 2 * KS_STEPS
test_reference = make_reference_trajectories(
    [heldout_initial(xs, KS_LENGTH)],
    first,
    minus_second,
    minus_fourth,
    truth,
    KS_DT,
    test_steps,
    KS_REF_SUBSTEPS,
)[1]

println("Composable Kuramoto--Sivashinsky identification")
@printf(
    "  n=%d L=%.1f dx=%.4f dt=%.4f transitions=%d epochs=%d m=%d nq=%d seeds=%s\n",
    KS_N,
    KS_LENGTH,
    dx,
    KS_DT,
    KS_STEPS,
    KS_EPOCHS,
    KS_M,
    KS_NQ,
    repr(KS_SEEDS),
)
println("  linear library: {-Dxx, -Dxxxx}; nonlinear basis: {-u ux}")

raw_rows = NamedTuple[]
best = nothing
for seed in KS_SEEDS
    for (method, stop_gradient) in (
        ("full_state_pullback", false),
        ("zero_state_ablation", true),
    )
        global best
        trained = train_case(
            layer,
            first,
            training_trajectories,
            seed;
            stop_input_gradient=stop_gradient,
        )
        estimate = trained.estimate
        prediction = rollout(
            layer,
            first,
            heldout_initial(xs, KS_LENGTH),
            estimate,
            test_steps,
        )
        rollout_error = trajectory_relative_error(prediction, test_reference)
        row = (
            method=method,
            seed=seed,
            estimate_a=estimate[1],
            estimate_b=estimate[2],
            estimate_c=estimate[3],
            parameter_relerr=relative_parameter_error(estimate),
            heldout_rollout_relerr=rollout_error,
            train_loss=trained.train_loss,
            train_seconds=trained.train_seconds,
            a_relerr=abs(estimate[1] - truth[1]) / truth[1],
            b_relerr=abs(estimate[2] - truth[2]) / truth[2],
            c_relerr=abs(estimate[3] - truth[3]) / truth[3],
            initial_a_gradient=trained.initial_gradient[1],
            initial_b_gradient=trained.initial_gradient[2],
            initial_c_gradient=trained.initial_gradient[3],
        )
        push!(raw_rows, row)
        @printf(
            "  %-20s seed=%d params=(%.5f, %.5f, %.5f) param_err=%.3e rollout=%.3e time=%.2fs\n",
            method,
            seed,
            estimate[1],
            estimate[2],
            estimate[3],
            row.parameter_relerr,
            row.heldout_rollout_relerr,
            row.train_seconds,
        )
        if method == "full_state_pullback" &&
                (best === nothing ||
                 row.parameter_relerr < best.row.parameter_relerr)
            best = (row=row, prediction=prediction)
        end
        CSV.write(
            joinpath(
                KS_OUTPUT_DIR,
                "kuramoto_sivashinsky_composable_raw.csv",
            ),
            DataFrame(raw_rows),
        )
    end
end

raw = DataFrame(raw_rows)
summary = summarize(raw)
CSV.write(
    joinpath(
        KS_OUTPUT_DIR,
        "kuramoto_sivashinsky_composable_summary.csv",
    ),
    summary,
)

trajectory_rows = NamedTuple[]
for index in eachindex(test_reference)
    for state_index in eachindex(test_reference[index])
        push!(trajectory_rows, (
            time=(index - 1) * KS_DT,
            x=xs[state_index],
            reference=test_reference[index][state_index],
            prediction=best.prediction[index][state_index],
        ))
    end
end
CSV.write(
    joinpath(
        KS_OUTPUT_DIR,
        "kuramoto_sivashinsky_composable_trajectory.csv",
    ),
    DataFrame(trajectory_rows),
)

println("\nSummary")
show(stdout, MIME("text/plain"), summary)
println()
