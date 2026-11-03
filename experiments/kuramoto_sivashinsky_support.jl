"""
Support recovery with inactive Kuramoto--Sivashinsky library terms.

The true equation is

    u_t = -u u_x - u_xx - u_xxxx.

The supplied semilinear library contains nine terms:

  1. -D_x u                 inactive
  2. -D_xx u                active
  3.  D_xxx u               inactive
  4. -D_xxxx u              active
  5. -D_x(u^2) / 2          active
  6. -D_x(u^3) / 3          inactive
  7. -D_x(u^4) / 4          inactive
  8. -D_x(u^5) / 5          inactive
  9. -D_x(u^6) / 6          inactive

Terms 1--4 form the sparse exponential library. Terms 5--9 form the
nonlinear flux library applied before the exponential layer. Training uses
an L1 discovery objective, a fixed threshold, and an unpenalized masked
L-BFGS refit.

Environment variables:
  KS_SUPPORT_SEEDS          comma-separated seeds (default: 11,22,33,44,55)
  KS_SUPPORT_L1S            comma-separated L1 values (default: 1e-7)
  KS_SUPPORT_THRESHOLD      support threshold (default: 0.08)
  KS_SUPPORT_N              spatial grid size (default: 64)
  KS_SUPPORT_MAX_POWER      largest nonlinear flux power (default: 6)
  KS_SUPPORT_M_KRYLOV       Krylov depth (default: 40)
  KS_SUPPORT_N_QUAD         quadrature nodes (default: 8)
  KS_SUPPORT_STAGE1_EPOCHS  Adam discovery epochs (default: 500)
  KS_SUPPORT_LBFGS_ITERS    L-BFGS iterations per stage (default: 250)
  KS_SUPPORT_STEPS          training transitions per field (default: 10)
  KS_SUPPORT_NOISE_REL      training-snapshot relative noise (default: 0)
  KS_SUPPORT_OUTPUT_TAG     optional suffix for output filenames
  KS_SUPPORT_OUTPUT_DIR     output directory (default: paper/figures)
  KS_SUPPORT_SMOKE          use a small fast configuration when set to 1

Outputs:
  kuramoto_sivashinsky_support_raw.csv
  kuramoto_sivashinsky_support_summary.csv
"""

using LinearAlgebra, SparseArrays, Random, Statistics, Printf
using Optimisers, Zygote
using CSV, DataFrames
using Optim

include("../src/shared_krylov.jl")

parse_list(::Type{T}, name, default) where {T} =
    parse.(T, split(get(ENV, name, default), ","))

const KSS_SMOKE = get(ENV, "KS_SUPPORT_SMOKE", "0") == "1"
const KSS_N = parse(
    Int,
    get(ENV, "KS_SUPPORT_N", KSS_SMOKE ? "32" : "64"),
)
const KSS_LENGTH = 22.0
const KSS_DT = 0.0125
const KSS_STEPS = parse(
    Int,
    get(ENV, "KS_SUPPORT_STEPS", KSS_SMOKE ? "3" : "10"),
)
const KSS_STAGE1_EPOCHS = parse(
    Int,
    get(
        ENV,
        "KS_SUPPORT_STAGE1_EPOCHS",
        KSS_SMOKE ? "5" : "500",
    ),
)
const KSS_LBFGS_ITERS = parse(
    Int,
    get(ENV, "KS_SUPPORT_LBFGS_ITERS", KSS_SMOKE ? "5" : "250"),
)
const KSS_SEEDS = parse_list(
    Int,
    "KS_SUPPORT_SEEDS",
    KSS_SMOKE ? "11" : "11,22,33,44,55",
)
const KSS_L1S = parse_list(
    Float64,
    "KS_SUPPORT_L1S",
    KSS_SMOKE ? "1e-7" : "1e-7",
)
const KSS_THRESHOLD = parse(
    Float64,
    get(ENV, "KS_SUPPORT_THRESHOLD", "0.08"),
)
const KSS_NOISE_REL = parse(
    Float64,
    get(ENV, "KS_SUPPORT_NOISE_REL", "0.0"),
)
const KSS_OUTPUT_TAG = get(ENV, "KS_SUPPORT_OUTPUT_TAG", "")
const KSS_OUTPUT_DIR = get(
    ENV,
    "KS_SUPPORT_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)
const KSS_M = parse(
    Int,
    get(ENV, "KS_SUPPORT_M_KRYLOV", KSS_SMOKE ? "20" : "40"),
)
const KSS_NQ = parse(Int, get(ENV, "KS_SUPPORT_N_QUAD", "8"))
const KSS_REF_SUBSTEPS = parse(
    Int,
    get(ENV, "KS_SUPPORT_REF_SUBSTEPS", KSS_SMOKE ? "5" : "20"),
)
const KSS_MAX_POWER = parse(
    Int,
    get(ENV, "KS_SUPPORT_MAX_POWER", "6"),
)
const KSS_TRUE = vcat([0.0, 1.0, 0.0, 1.0, 1.0], zeros(KSS_MAX_POWER - 2))
const KSS_TRUE_SUPPORT = Set([2, 4, 5])
const KSS_LABELS = vcat([
    "-Dx",
    "-Dxx",
    "Dxxx",
    "-Dxxxx",
], ["-Dx(u^$(power))/$(power)" for power in 2:KSS_MAX_POWER])

KSS_STEPS > 0 || error("KS_SUPPORT_STEPS must be positive")
KSS_NOISE_REL >= 0 || error("KS_SUPPORT_NOISE_REL must be nonnegative")
KSS_N >= 16 || error("KS_SUPPORT_N must be at least 16")
KSS_MAX_POWER >= 2 || error("KS_SUPPORT_MAX_POWER must be at least 2")

softplus_inverse(x) = log(expm1(x))

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
    third = sparse(first * second)
    fourth = sparse(second * second)
    first, second, third, fourth
end

function flux_direction(first, u, power)
    -(first * (u .^ power)) ./ power
end

function nonlinear_tendency(first, u, coefficients)
    sum(
        coefficients[index] .* flux_direction(first, u, index + 1)
        for index in eachindex(coefficients)
    )
end

function model_step(
    layer,
    first,
    u,
    coefficients,
    dt,
)
    linear = coefficients[1:4]
    nonlinear = coefficients[5:end]
    nonlinear_input = u .+ dt .* nonlinear_tendency(
        first,
        u,
        nonlinear,
    )
    layer(nonlinear_input, linear, dt)
end

function rhs(
    u,
    first,
    linear_directions,
    coefficients,
)
    linear = sum(
        coefficients[index] .* (linear_directions[index] * u)
        for index in 1:4
    )
    linear .+ nonlinear_tendency(first, u, coefficients[5:end])
end

function rk4_interval(
    u,
    first,
    linear_directions,
    coefficients,
    dt,
    n_substeps,
)
    step = dt / n_substeps
    state = copy(u)
    for _ in 1:n_substeps
        k1 = rhs(state, first, linear_directions, coefficients)
        k2 = rhs(
            state .+ (step / 2) .* k1,
            first,
            linear_directions,
            coefficients,
        )
        k3 = rhs(
            state .+ (step / 2) .* k2,
            first,
            linear_directions,
            coefficients,
        )
        k4 = rhs(
            state .+ step .* k3,
            first,
            linear_directions,
            coefficients,
        )
        state = state .+
            (step / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    state
end

function initial_conditions(xs)
    phase = 2π .* xs ./ KSS_LENGTH
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

function heldout_initial(xs)
    phase = 2π .* xs ./ KSS_LENGTH
    0.61 .* sin.(2 .* phase .- 0.5) .+
        0.24 .* cos.(3 .* phase .+ 0.4) .-
        0.11 .* sin.(7 .* phase)
end

function reference_trajectories(
    initials,
    first,
    linear_directions,
    n_steps,
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
                    linear_directions,
                    KSS_TRUE,
                    KSS_DT,
                    KSS_REF_SUBSTEPS,
                ),
            )
        end
        push!(trajectories, trajectory)
    end
    trajectories
end

function add_snapshot_noise(trajectories, relative_noise, seed)
    relative_noise == 0 && return deepcopy(trajectories)
    rng = MersenneTwister(seed)
    [
        [
            state .+
                relative_noise .* sqrt(mean(abs2, state)) .*
                randn(rng, length(state))
            for state in trajectory
        ]
        for trajectory in trajectories
    ]
end

function data_loss(layer, first, coefficients, trajectories)
    sum(
        0.5 * mean(abs2, trajectory[index + 1] -
            model_step(
                layer,
                first,
                trajectory[index],
                coefficients,
                KSS_DT,
            ))
        for trajectory in trajectories
        for index in 1:(length(trajectory) - 1)
    ) / sum(length(trajectory) - 1 for trajectory in trajectories)
end

function discovery_objective(
    layer,
    first,
    raw,
    trajectories,
    l1,
)
    coefficients = softplus.(raw)
    data_loss(layer, first, coefficients, trajectories) +
        l1 * sum(coefficients)
end

function train_adam(
    layer,
    first,
    raw_start,
    trajectories,
    l1,
)
    raw = copy(raw_start)
    optimizer = Optimisers.setup(Optimisers.Adam(2.0e-2), raw)
    seconds = @elapsed for _ in 1:KSS_STAGE1_EPOCHS
        _, gradients = Zygote.withgradient(raw) do values
            discovery_objective(
                layer,
                first,
                values,
                trajectories,
                l1,
            )
        end
        optimizer, raw = Optimisers.update!(optimizer, raw, gradients[1])
    end
    raw, seconds
end

function lbfgs_optimize(objective, raw_start)
    function gradient!(storage, raw)
        _, gradients = Zygote.withgradient(objective, raw)
        storage .= gradients[1]
        nothing
    end
    result = optimize(
        objective,
        gradient!,
        copy(raw_start),
        LBFGS(),
        Optim.Options(
            iterations=KSS_LBFGS_ITERS,
            show_trace=false,
            store_trace=false,
        ),
    )
    (
        raw=Optim.minimizer(result),
        converged=Optim.converged(result),
        iterations=Optim.iterations(result),
        minimum=Optim.minimum(result),
    )
end

function polish_discovery(
    layer,
    first,
    raw_start,
    trajectories,
    l1,
)
    objective(raw) = discovery_objective(
        layer,
        first,
        raw,
        trajectories,
        l1,
    )
    lbfgs_optimize(objective, raw_start)
end

function masked_refit(
    layer,
    first,
    raw_start,
    trajectories,
)
    raw = copy(raw_start)
    mask = Float64.(softplus.(raw) .>= KSS_THRESHOLD)
    all(iszero, mask) && (mask[argmax(raw)] = 1.0)
    total_iterations = 0
    converged = true

    for _ in 1:5
        objective(candidate) = data_loss(
            layer,
            first,
            softplus.(candidate) .* mask,
            trajectories,
        )
        result = lbfgs_optimize(objective, raw)
        raw = result.raw
        total_iterations += result.iterations
        converged &= result.converged

        coefficients = softplus.(raw) .* mask
        new_mask = mask .* Float64.(coefficients .>= KSS_THRESHOLD)
        new_mask == mask && break
        mask = new_mask
    end

    coefficients = softplus.(raw) .* mask
    coefficients[coefficients .< KSS_THRESHOLD] .= 0.0
    (
        raw=raw,
        coefficients=coefficients,
        mask=mask,
        converged=converged,
        iterations=total_iterations,
    )
end

function support_metrics(coefficients)
    selected = findall(>=(KSS_THRESHOLD), coefficients)
    selected_set = Set(selected)
    true_positives = length(intersect(selected_set, KSS_TRUE_SUPPORT))
    false_positives = length(setdiff(selected_set, KSS_TRUE_SUPPORT))
    false_negatives = length(setdiff(KSS_TRUE_SUPPORT, selected_set))
    (
        selected_indices=repr(selected),
        selected_labels=join(KSS_LABELS[selected], ","),
        selected_size=length(selected),
        exact_support=selected_set == KSS_TRUE_SUPPORT,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        support_precision=true_positives /
            max(true_positives + false_positives, 1),
        support_recall=true_positives /
            max(true_positives + false_negatives, 1),
    )
end

function rollout(layer, first, initial, coefficients, n_steps)
    states = [copy(initial)]
    for _ in 1:n_steps
        push!(
            states,
            model_step(
                layer,
                first,
                states[end],
                coefficients,
                KSS_DT,
            ),
        )
    end
    states
end

function trajectory_error(prediction, reference)
    sqrt(
        sum(
            norm(prediction[index] - reference[index])^2
            for index in eachindex(reference)
        ) / sum(norm(state)^2 for state in reference),
    )
end

function evaluate(
    layer,
    first,
    coefficients,
    trajectories,
    test_initial,
    test_reference,
)
    support = support_metrics(coefficients)
    true_indices = [2, 4, 5]
    false_indices = collect(setdiff(eachindex(coefficients), true_indices))
    prediction = rollout(
        layer,
        first,
        test_initial,
        coefficients,
        length(test_reference) - 1,
    )
    merge(
        support,
        (
            data_loss=data_loss(
                layer,
                first,
                coefficients,
                trajectories,
            ),
            coefficient_relerr=norm(coefficients - KSS_TRUE) /
                norm(KSS_TRUE),
            active_coefficient_relerr=norm(
                coefficients[true_indices] -
                KSS_TRUE[true_indices],
            ) / norm(KSS_TRUE[true_indices]),
            true_coefficient_min=minimum(coefficients[true_indices]),
            false_coefficient_max=maximum(coefficients[false_indices]),
            separation_margin=minimum(coefficients[true_indices]) -
                maximum(coefficients[false_indices]),
            heldout_rollout_relerr=trajectory_error(
                prediction,
                test_reference,
            ),
        ),
    )
end

function result_row(
    stage,
    seed,
    noise_seed,
    l1,
    coefficients,
    evaluation,
    optimizer_result;
    adam_seconds=missing,
)
    merge(
        (
            stage=stage,
            seed=seed,
            noise_seed=noise_seed,
            l1=l1,
            threshold=KSS_THRESHOLD,
            relative_noise=KSS_NOISE_REL,
            library_size=length(coefficients),
            training_steps=KSS_STEPS,
            optimizer_converged=optimizer_result.converged,
            optimizer_iterations=optimizer_result.iterations,
            adam_seconds=adam_seconds,
            spatial_size=KSS_N,
            max_flux_power=KSS_MAX_POWER,
        ),
        (; (
            Symbol("theta_$(index)") => coefficients[index]
            for index in eachindex(coefficients)
        )...),
        evaluation,
    )
end

function summarize(raw)
    combine(
        groupby(raw, [:l1, :stage]),
        nrow => :n_seeds,
        :exact_support => mean => :exact_support_rate,
        :support_precision => mean => :support_precision_mean,
        :support_recall => mean => :support_recall_mean,
        :false_positives => mean => :false_positives_mean,
        :false_negatives => mean => :false_negatives_mean,
        :coefficient_relerr => mean => :coefficient_relerr_mean,
        :active_coefficient_relerr =>
            mean => :active_coefficient_relerr_mean,
        :heldout_rollout_relerr =>
            mean => :heldout_rollout_relerr_mean,
        :true_coefficient_min => mean => :true_coefficient_min_mean,
        :false_coefficient_max => mean => :false_coefficient_max_mean,
        :separation_margin => mean => :separation_margin_mean,
        :optimizer_converged => mean => :optimizer_converged_rate,
    )
end

mkpath(KSS_OUTPUT_DIR)
xs = collect(range(0, KSS_LENGTH; length=KSS_N + 1))[1:end-1]
first, second, third, fourth = periodic_derivatives(KSS_N, KSS_LENGTH)
linear_directions = SparseMatrixCSC{Float64,Int}[
    sparse(-first),
    sparse(-second),
    third,
    sparse(-fourth),
]
zero_operator = spzeros(Float64, KSS_N, KSS_N)
layer = SemigroupLayer(
    zero_operator,
    linear_directions;
    m_krylov=min(KSS_M, KSS_N - 1),
    n_quad=KSS_NQ,
)
clean_training_trajectories = reference_trajectories(
    initial_conditions(xs),
    first,
    linear_directions,
    KSS_STEPS,
)
test_initial = heldout_initial(xs)
test_reference = reference_trajectories(
    [test_initial],
    first,
    linear_directions,
    2 * KSS_STEPS,
)[1]

println("Kuramoto--Sivashinsky support recovery")
@printf(
    "  n=%d dt=%.4f transitions=%d noise=%.1e library=%d Adam=%d L-BFGS=%d threshold=%.3f seeds=%s l1=%s\n",
    KSS_N,
    KSS_DT,
    KSS_STEPS,
    KSS_NOISE_REL,
    length(KSS_TRUE),
    KSS_STAGE1_EPOCHS,
    KSS_LBFGS_ITERS,
    KSS_THRESHOLD,
    repr(KSS_SEEDS),
    repr(KSS_L1S),
)
println("  true support: ", join(KSS_LABELS[sort(collect(KSS_TRUE_SUPPORT))], ","))

rows = NamedTuple[]
output_suffix = isempty(KSS_OUTPUT_TAG) ? "" : "_" * KSS_OUTPUT_TAG
for l1 in KSS_L1S
    for (seed_index, seed) in enumerate(KSS_SEEDS)
        noise_seed = 2026 + seed_index
        training_trajectories = add_snapshot_noise(
            clean_training_trajectories,
            KSS_NOISE_REL,
            noise_seed,
        )
        Random.seed!(seed)
        raw_start = fill(softplus_inverse(0.10), length(KSS_TRUE)) .+
            0.08 .* randn(length(KSS_TRUE))
        raw_adam, adam_seconds = train_adam(
            layer,
            first,
            raw_start,
            training_trajectories,
            l1,
        )
        discovery = polish_discovery(
            layer,
            first,
            raw_adam,
            training_trajectories,
            l1,
        )
        discovery_coefficients = softplus.(discovery.raw)
        discovery_evaluation = evaluate(
            layer,
            first,
            discovery_coefficients,
            training_trajectories,
            test_initial,
            test_reference,
        )
        push!(
            rows,
            result_row(
                "l1_discovery",
                seed,
                noise_seed,
                l1,
                discovery_coefficients,
                discovery_evaluation,
                discovery;
                adam_seconds=adam_seconds,
            ),
        )

        refit = masked_refit(
            layer,
            first,
            discovery.raw,
            training_trajectories,
        )
        refit_evaluation = evaluate(
            layer,
            first,
            refit.coefficients,
            training_trajectories,
            test_initial,
            test_reference,
        )
        push!(
            rows,
            result_row(
                "masked_refit",
                seed,
                noise_seed,
                l1,
                refit.coefficients,
                refit_evaluation,
                refit,
            ),
        )

        @printf(
            "  l1=%.1e seed=%d select=%s exact=%s margin=%.3e refit=%s exact=%s coeff=%.3e rollout=%.3e\n",
            l1,
            seed,
            discovery_evaluation.selected_indices,
            string(discovery_evaluation.exact_support),
            discovery_evaluation.separation_margin,
            refit_evaluation.selected_indices,
            string(refit_evaluation.exact_support),
            refit_evaluation.active_coefficient_relerr,
            refit_evaluation.heldout_rollout_relerr,
        )

        raw = DataFrame(rows)
        CSV.write(
            joinpath(
                KSS_OUTPUT_DIR,
                "kuramoto_sivashinsky_support$(output_suffix)_raw.csv",
            ),
            raw,
        )
        CSV.write(
            joinpath(
                KSS_OUTPUT_DIR,
                "kuramoto_sivashinsky_support$(output_suffix)_summary.csv",
            ),
            summarize(raw),
        )
    end
end

println("\nSummary")
show(stdout, MIME("text/plain"), summarize(DataFrame(rows)))
println()
