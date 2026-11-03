"""
Ungrouped sparse discovery in the 40-direction Brusselator CME library.

Unlike ``brusselator_sparse_discovery.jl``, this driver supplies no
one-direction-per-channel grouping.  A single nonnegative 40-by-3 linear
context map is fitted over the entire library,

    theta_j(A, B) = W[j, :] dot [1, A, B],

with an L1 penalty during support discovery.  A fixed threshold, chosen
independently of the true support size, defines the support before an
unpenalized masked refit.  The true support is used only for evaluation.

The state grid is deliberately smaller than the headline experiment so that a
five-seed no-grouping audit can run quickly.  The candidate mechanism grammar
and all 40 stoichiometry/propensity directions are unchanged.

Environment variables
---------------------

    BRUSS_UNGROUPED_SEEDS=11,22,33,44,55
    BRUSS_UNGROUPED_STAGE1=500
    BRUSS_UNGROUPED_STAGE2=300
    BRUSS_UNGROUPED_L1=1e-4
    BRUSS_UNGROUPED_THRESHOLD=0.2
    BRUSS_UNGROUPED_GRID_MAX=12
    BRUSS_UNGROUPED_OBSERVATION_GRID=submitted_gaps
    BRUSS_UNGROUPED_STEP_BATCH=3
    BRUSS_UNGROUPED_LBFGS_ITERS=0
    BRUSS_UNGROUPED_OUTPUT_DIR=paper/figures
    BRUSS_UNGROUPED_SMOKE=0
"""

using LinearAlgebra, SparseArrays, Random, Statistics, Printf
using Optim, Optimisers, Zygote
using CSV, DataFrames

include("../src/shared_krylov.jl")

const UNGROUPED_SEEDS = parse.(
    Int,
    split(
        get(ENV, "BRUSS_UNGROUPED_SEEDS", "11,22,33,44,55"),
        ",",
    ),
)
const UNGROUPED_STAGE1 = parse(
    Int, get(ENV, "BRUSS_UNGROUPED_STAGE1", "500"),
)
const UNGROUPED_STAGE2 = parse(
    Int, get(ENV, "BRUSS_UNGROUPED_STAGE2", "300"),
)
const UNGROUPED_L1 = parse(
    Float64, get(ENV, "BRUSS_UNGROUPED_L1", "1e-4"),
)
const UNGROUPED_THRESHOLD = parse(
    Float64, get(ENV, "BRUSS_UNGROUPED_THRESHOLD", "0.2"),
)
const UNGROUPED_GRID_MAX = parse(
    Int, get(ENV, "BRUSS_UNGROUPED_GRID_MAX", "12"),
)
const UNGROUPED_OBSERVATION_GRID = get(
    ENV, "BRUSS_UNGROUPED_OBSERVATION_GRID", "submitted_gaps",
)
const TIMES_UNGROUPED = if UNGROUPED_OBSERVATION_GRID == "dense_dt_0.01"
    collect(0.0:0.01:0.75)
elseif UNGROUPED_OBSERVATION_GRID == "medium_dt_0.05"
    collect(0.0:0.05:0.75)
elseif UNGROUPED_OBSERVATION_GRID == "coarse_dt_0.15"
    collect(0.0:0.15:0.75)
elseif UNGROUPED_OBSERVATION_GRID == "submitted_gaps"
    [0.0, 0.05, 0.25, 0.75]
else
    error("Unknown observation grid: $UNGROUPED_OBSERVATION_GRID")
end
const DTS_UNGROUPED = diff(TIMES_UNGROUPED)
const UNGROUPED_STEP_BATCH = min(
    length(DTS_UNGROUPED),
    parse(Int, get(ENV, "BRUSS_UNGROUPED_STEP_BATCH", "3")),
)
const UNGROUPED_LBFGS_ITERS = parse(
    Int, get(ENV, "BRUSS_UNGROUPED_LBFGS_ITERS", "0"),
)
const UNGROUPED_OUTPUT_DIR = get(
    ENV,
    "BRUSS_UNGROUPED_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)
const UNGROUPED_SMOKE =
    get(ENV, "BRUSS_UNGROUPED_SMOKE", "0") == "1"

const CHANNELS_UNGROUPED = [(1, 0), (-1, 0), (-1, 1), (1, -1)]
const MONOMIALS_UNGROUPED = [
    (a, degree - a)
    for degree in 0:3
    for a in 0:degree
]
const M_UNGROUPED =
    length(CHANNELS_UNGROUPED) * length(MONOMIALS_UNGROUPED)
const TRUE_SUPPORT_UNGROUPED = Set([1, 13, 23, 39])
const TRAIN_CONTEXTS_UNGROUPED = [
    (0.8, 2.4),
    (0.8, 3.6),
    (1.4, 2.4),
    (1.4, 3.6),
]
const TEST_CONTEXTS_UNGROUPED = [
    (0.9, 2.5),
    (1.1, 2.9),
    (1.0, 3.4),
    (1.3, 3.1),
]
const LONG_TIME_UNGROUPED = 1.5
const INITIAL_STATES_UNGROUPED = [
    (x_initial, y_initial)
    for x_initial in (1, 3, 5)
    for y_initial in (0, 2)
]

function comb_ungrouped(x, order)
    order == 0 && return 1.0
    x < order && return 0.0
    value = 1.0
    for offset in 0:(order - 1)
        value *= x - offset
    end
    value / factorial(order)
end

function build_ungrouped_library(grid_max)
    n = (grid_max + 1)^2
    linear_index(x, y) = x * (grid_max + 1) + y + 1
    directions = SparseMatrixCSC{Float64,Int}[]
    labels = String[]
    for (dx, dy) in CHANNELS_UNGROUPED
        for (a, b) in MONOMIALS_UNGROUPED
            rows = Int[]
            cols = Int[]
            values = Float64[]
            for x in 0:grid_max, y in 0:grid_max
                (x < a || y < b) && continue
                x_next = x + dx
                y_next = y + dy
                (
                    x_next < 0 || x_next > grid_max ||
                    y_next < 0 || y_next > grid_max
                ) && continue
                propensity =
                    comb_ungrouped(x, a) * comb_ungrouped(y, b)
                iszero(propensity) && continue
                source = linear_index(x, y)
                destination = linear_index(x_next, y_next)
                push!(rows, destination)
                push!(cols, source)
                push!(values, propensity)
                push!(rows, source)
                push!(cols, source)
                push!(values, -propensity)
            end
            push!(directions, sparse(rows, cols, values, n, n))
            push!(
                labels,
                "($dx,$dy)*C(x,$a)*C(y,$b)",
            )
        end
    end
    directions, labels
end

function true_rates_ungrouped(context)
    A_context, B_context = context
    rates = zeros(M_UNGROUPED)
    rates[1] = A_context
    rates[13] = 1.0
    rates[23] = B_context
    rates[39] = 1.0
    rates
end

context_features_ungrouped(context) =
    Float64[1.0, context[1], context[2]]

function initial_distribution_ungrouped(
    x_initial,
    y_initial,
    grid_max,
)
    p = zeros((grid_max + 1)^2)
    x_initial = clamp(x_initial, 0, grid_max)
    index = x_initial * (grid_max + 1) + y_initial + 1
    p[index] = 1.0
    p
end

function exact_paths_ungrouped(
    context,
    directions,
    grid_max,
)
    generator = sum(
        true_rates_ungrouped(context)[j] * directions[j]
        for j in 1:M_UNGROUPED
    )
    propagators = Dict{Float64,Matrix{Float64}}()
    for dt in DTS_UNGROUPED
        key = round(dt; digits=12)
        get!(propagators, key) do
            exp(Matrix(generator) * key)
        end
    end
    [
        let
            states = [
                initial_distribution_ungrouped(
                    x_initial, y_initial, grid_max,
                ),
            ]
            for dt in DTS_UNGROUPED
                key = round(dt; digits=12)
                push!(states, propagators[key] * states[end])
            end
            states
        end
        for (x_initial, y_initial) in INITIAL_STATES_UNGROUPED
    ]
end

function fisher_scales_ungrouped(directions, training_data)
    states = [
        state
        for context_data in values(training_data)
        for path in context_data
        for state in path
    ]
    scales = [
        mean(sum(abs, direction * state) / 2 for state in states)
        for direction in directions
    ]
    max.(scales, 1.0e-8)
end

function prediction_loss_ungrouped(
    layer,
    weights,
    mask,
    data,
    contexts,
    steps=eachindex(DTS_UNGROUPED),
)
    total = 0.0
    count = 0
    masked_weights = weights .* mask
    for context in contexts
        coefficients =
            masked_weights * context_features_ungrouped(context)
        for path in data[context]
            for step in steps
                dt = DTS_UNGROUPED[step]
                prediction = layer(path[step], coefficients, dt)
                total += 0.5 * sum(abs2, path[step + 1] - prediction)
                count += 1
            end
        end
    end
    total / count
end

function support_scores_ungrouped(weights, contexts, scales)
    scaled_coefficients = [
        weights * context_features_ungrouped(context)
        for context in contexts
    ]
    physical_coefficients = [
        coefficients ./ scales for coefficients in scaled_coefficients
    ]
    [
        mean(abs(coefficients[j]) for coefficients in physical_coefficients)
        for j in 1:M_UNGROUPED
    ]
end

function train_ungrouped_seed(
    seed,
    layer,
    scales,
    training_data,
    test_data,
    directions,
)
    Random.seed!(seed)
    weights = 0.01 .* rand(M_UNGROUPED, 3)
    full_mask = ones(M_UNGROUPED, 1)
    optimizer = Optimisers.setup(Optimisers.Adam(2.0e-2), weights)
    stage1_epochs = UNGROUPED_SMOKE ? 2 : UNGROUPED_STAGE1
    stage2_epochs = UNGROUPED_SMOKE ? 1 : UNGROUPED_STAGE2

    for _ in 1:stage1_epochs
        steps = sort(randperm(length(DTS_UNGROUPED))[1:UNGROUPED_STEP_BATCH])
        _, gradient = Zygote.withgradient(weights) do candidate
            prediction_loss_ungrouped(
                layer,
                candidate,
                full_mask,
                training_data,
                TRAIN_CONTEXTS_UNGROUPED,
                steps,
            ) + UNGROUPED_L1 * sum(abs, candidate)
        end
        Optimisers.update!(optimizer, weights, gradient[1])
        weights .= max.(weights, 0.0)
    end

    stage1_scores = support_scores_ungrouped(
        weights, TRAIN_CONTEXTS_UNGROUPED, scales,
    )
    stage1_order = sortperm(stage1_scores; rev=true)
    stage1_top_scores = repr([
        (index, stage1_scores[index])
        for index in stage1_order[1:min(12, length(stage1_order))]
    ])
    stage1_min_true_score = minimum(
        stage1_scores[index] for index in TRUE_SUPPORT_UNGROUPED
    )
    stage1_max_false_score = maximum(
        stage1_scores[index]
        for index in setdiff(
            Set(1:M_UNGROUPED), TRUE_SUPPORT_UNGROUPED,
        )
    )
    selected = findall(>=(UNGROUPED_THRESHOLD), stage1_scores)
    isempty(selected) && (selected = [argmax(stage1_scores)])
    mask = zeros(M_UNGROUPED, 1)
    mask[selected] .= 1.0

    refit_optimizer = Optimisers.setup(Optimisers.Adam(8.0e-3), weights)
    for _ in 1:stage2_epochs
        steps = sort(randperm(length(DTS_UNGROUPED))[1:UNGROUPED_STEP_BATCH])
        _, gradient = Zygote.withgradient(weights) do candidate
            prediction_loss_ungrouped(
                layer,
                candidate,
                mask,
                training_data,
                TRAIN_CONTEXTS_UNGROUPED,
                steps,
            )
        end
        Optimisers.update!(refit_optimizer, weights, gradient[1])
        weights .= max.(weights, 0.0)
        weights .*= mask
    end

    final_scores = support_scores_ungrouped(
        weights, TRAIN_CONTEXTS_UNGROUPED, scales,
    )
    selected = findall(>=(UNGROUPED_THRESHOLD), final_scores)
    if UNGROUPED_LBFGS_ITERS > 0 && !isempty(selected)
        active = copy(selected)
        polish_steps = unique(round.(
            Int,
            range(
                1,
                length(DTS_UNGROUPED);
                length=min(UNGROUPED_STEP_BATCH, length(DTS_UNGROUPED)),
            ),
        ))
        selector = sparse(
            active,
            collect(1:length(active)),
            ones(length(active)),
            M_UNGROUPED,
            length(active),
        )
        unpack_active(candidate) =
            selector * reshape(candidate, length(active), 3)
        objective(candidate) = prediction_loss_ungrouped(
            layer,
            unpack_active(candidate),
            full_mask,
            training_data,
            TRAIN_CONTEXTS_UNGROUPED,
            polish_steps,
        )
        function gradient!(storage, candidate)
            _, gradient = Zygote.withgradient(objective, candidate)
            storage .= gradient[1]
            nothing
        end
        result = optimize(
            objective,
            gradient!,
            vec(copy(weights[active, :])),
            LBFGS(),
            Optim.Options(
                iterations=UNGROUPED_LBFGS_ITERS,
                show_trace=false,
                store_trace=false,
            ),
        )
        weights .= unpack_active(Optim.minimizer(result))
        weights .= max.(weights, 0.0)
        mask .= 0.0
        mask[active] .= 1.0
        final_scores = support_scores_ungrouped(
            weights, TRAIN_CONTEXTS_UNGROUPED, scales,
        )
        selected = findall(>=(UNGROUPED_THRESHOLD), final_scores)
    end
    selected_set = Set(selected)
    true_positives =
        length(intersect(selected_set, TRUE_SUPPORT_UNGROUPED))
    false_positives =
        length(setdiff(selected_set, TRUE_SUPPORT_UNGROUPED))
    false_negatives =
        length(setdiff(TRUE_SUPPORT_UNGROUPED, selected_set))
    precision =
        true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, eps())

    coefficient_errors = Float64[]
    prediction_errors = Float64[]
    long_errors = Float64[]
    for context in TEST_CONTEXTS_UNGROUPED
        scaled_hat =
            (weights .* mask) * context_features_ungrouped(context)
        rates_hat = scaled_hat ./ scales
        rates_true = true_rates_ungrouped(context)
        push!(
            coefficient_errors,
            norm(rates_hat - rates_true) / norm(rates_true),
        )
        generator_hat = sum(
            rates_hat[j] * directions[j] for j in 1:M_UNGROUPED
        )
        generator_true = sum(
            rates_true[j] * directions[j] for j in 1:M_UNGROUPED
        )
        for path in test_data[context]
            p_initial = path[1]
            prediction_hat =
                exp(Matrix(generator_hat) * sum(DTS_UNGROUPED)) * p_initial
            prediction_true =
                exp(Matrix(generator_true) * sum(DTS_UNGROUPED)) * p_initial
            push!(
                prediction_errors,
                norm(prediction_hat - prediction_true) /
                norm(prediction_true),
            )
            long_hat =
                exp(Matrix(generator_hat) * LONG_TIME_UNGROUPED) * p_initial
            long_true =
                exp(Matrix(generator_true) * LONG_TIME_UNGROUPED) * p_initial
            push!(
                long_errors,
                norm(long_hat - long_true) / norm(long_true),
            )
        end
    end

    return (
        seed=seed,
        stage1_top_scores=stage1_top_scores,
        stage1_min_true_score=stage1_min_true_score,
        stage1_max_false_score=stage1_max_false_score,
        selected=repr(selected),
        selected_size=length(selected),
        exact_support=selected_set == TRUE_SUPPORT_UNGROUPED,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        support_precision=precision,
        support_recall=recall,
        support_f1=f1,
        coefficient_relerr=mean(coefficient_errors),
        heldout_prediction_relerr=mean(prediction_errors),
        long_horizon_relerr=mean(long_errors),
        train_loss=prediction_loss_ungrouped(
            layer,
            weights,
            mask,
            training_data,
            TRAIN_CONTEXTS_UNGROUPED,
        ),
        l1=UNGROUPED_L1,
        threshold=UNGROUPED_THRESHOLD,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        grid_max=UNGROUPED_GRID_MAX,
        observation_grid=UNGROUPED_OBSERVATION_GRID,
        snapshots=length(TIMES_UNGROUPED),
        step_batch=UNGROUPED_STEP_BATCH,
        lbfgs_iters=UNGROUPED_LBFGS_ITERS,
        library_size=M_UNGROUPED,
        support_oracle=false,
        channel_grouping=false,
    )
end

println("Building ungrouped 40-direction Brusselator library")
directions, labels =
    build_ungrouped_library(UNGROUPED_GRID_MAX)
length(directions) == M_UNGROUPED ||
    error("Expected $M_UNGROUPED directions")

training_data = Dict(
    context => exact_paths_ungrouped(
        context, directions, UNGROUPED_GRID_MAX,
    )
    for context in TRAIN_CONTEXTS_UNGROUPED
)
test_data = Dict(
    context => exact_paths_ungrouped(
        context, directions, UNGROUPED_GRID_MAX,
    )
    for context in TEST_CONTEXTS_UNGROUPED
)
scales = fisher_scales_ungrouped(directions, training_data)
scaled_directions = [
    directions[j] / scales[j] for j in 1:M_UNGROUPED
]
layer = SemigroupLayer(
    spzeros(size(directions[1])...),
    scaled_directions;
    m_krylov=min(35, size(directions[1], 1)),
    n_quad=8,
)

seeds = UNGROUPED_SMOKE ? [first(UNGROUPED_SEEDS)] : UNGROUPED_SEEDS
rows = NamedTuple[]
println(
    "Ungrouped Brusselator discovery with M=$M_UNGROUPED, " *
    "grid=$UNGROUPED_OBSERVATION_GRID, snapshots=$(length(TIMES_UNGROUPED)), " *
    "seeds=$seeds, lambda=$UNGROUPED_L1, " *
    "threshold=$UNGROUPED_THRESHOLD",
)
for seed in seeds
    seconds = @elapsed result = train_ungrouped_seed(
        seed,
        layer,
        scales,
        training_data,
        test_data,
        directions,
    )
    row = merge(result, (train_seconds=seconds,))
    push!(rows, row)
    @printf(
        "seed=%d support=%s TP=%d FP=%d F1=%.3f coef=%.2e pred=%.2e long=%.2e (%.1fs)\n",
        seed,
        result.selected,
        result.true_positives,
        result.false_positives,
        result.support_f1,
        result.coefficient_relerr,
        result.heldout_prediction_relerr,
        result.long_horizon_relerr,
        seconds,
    )
    flush(stdout)
end

raw = DataFrame(rows)
summary = DataFrame([(
    n_seeds=nrow(raw),
    exact_support_rate=mean(raw.exact_support),
    support_f1_mean=mean(raw.support_f1),
    support_f1_std=nrow(raw) > 1 ? std(raw.support_f1) : NaN,
    selected_size_mean=mean(raw.selected_size),
    coefficient_relerr_mean=mean(raw.coefficient_relerr),
    coefficient_relerr_std=
        nrow(raw) > 1 ? std(raw.coefficient_relerr) : NaN,
    heldout_prediction_relerr_mean=
        mean(raw.heldout_prediction_relerr),
    long_horizon_relerr_mean=mean(raw.long_horizon_relerr),
    train_seconds_mean=mean(raw.train_seconds),
    library_size=M_UNGROUPED,
    grouped_support=false,
    support_size_oracle=false,
    l1=UNGROUPED_L1,
    threshold=UNGROUPED_THRESHOLD,
    grid_max=UNGROUPED_GRID_MAX,
    observation_grid=UNGROUPED_OBSERVATION_GRID,
    snapshots=length(TIMES_UNGROUPED),
    step_batch=UNGROUPED_STEP_BATCH,
    lbfgs_iters=UNGROUPED_LBFGS_ITERS,
)])

mkpath(UNGROUPED_OUTPUT_DIR)
CSV.write(
    joinpath(
        UNGROUPED_OUTPUT_DIR,
        "brusselator_ungrouped_raw.csv",
    ),
    raw,
)
CSV.write(
    joinpath(
        UNGROUPED_OUTPUT_DIR,
        "brusselator_ungrouped_summary.csv",
    ),
    summary,
)
println("Saved ungrouped Brusselator results to $UNGROUPED_OUTPUT_DIR")
