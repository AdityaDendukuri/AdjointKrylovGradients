"""
Generic candidate-library robustness stress test.

The system is a finite-state continuous-time Markov chain with a fixed
birth--death background generator and four active sparse jump mechanisms.
The experiment varies:
  * total candidate-library size,
  * correlation between inactive distractors and active directions,
  * mild out-of-library misspecification.

Environment variables:
  LIB_STRESS_SEEDS          comma-separated seeds (default: 11,22,33,44,55)
  LIB_STRESS_STAGE1_EPOCHS  sparse-selection epochs (default: 400)
  LIB_STRESS_STAGE2_EPOCHS  masked-refit epochs (default: 250)
  LIB_STRESS_L1             stage-1 L1 coefficient (default: 2e-4)
  LIB_STRESS_THRESHOLD      absolute support threshold (default: 0.08)
  LIB_STRESS_OUTPUT_DIR     output directory (default: paper/figures)
  LIB_STRESS_SMOKE          if 1, run one small scenario
  LIB_STRESS_EXHAUSTIVE     if 1, use all n(n-1)=870 directed jump edges

Outputs:
  library_robustness_raw.csv
  library_robustness_summary.csv
  exhaustive_edge_library_raw.csv       (exhaustive mode)
  exhaustive_edge_library_summary.csv   (exhaustive mode)
"""

using LinearAlgebra, SparseArrays, Random, Statistics, Printf
using Flux, Zygote
using CSV, DataFrames

include("../src/shared_krylov.jl")

const STRESS_SEEDS = parse.(
    Int,
    split(get(ENV, "LIB_STRESS_SEEDS", "11,22,33,44,55"), ","),
)
const STAGE1_EPOCHS = parse(
    Int, get(ENV, "LIB_STRESS_STAGE1_EPOCHS", "400"),
)
const STAGE2_EPOCHS = parse(
    Int, get(ENV, "LIB_STRESS_STAGE2_EPOCHS", "250"),
)
const STRESS_L1 = parse(Float64, get(ENV, "LIB_STRESS_L1", "2e-4"))
const SUPPORT_THRESHOLD = parse(
    Float64, get(ENV, "LIB_STRESS_THRESHOLD", "0.08"),
)
const STRESS_OUTPUT_DIR = get(
    ENV,
    "LIB_STRESS_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)
const STRESS_SMOKE = get(ENV, "LIB_STRESS_SMOKE", "0") == "1"
const STRESS_EXHAUSTIVE =
    get(ENV, "LIB_STRESS_EXHAUSTIVE", "0") == "1"

const N_STRESS = 30
const TRUE_RATES = [1.0, 0.8, 1.2, 0.6]
const TRAIN_TIMES = [0.2, 0.6, 1.2]
const LONG_TIME = 2.0
const KRYLOV_M = 25
const N_QUAD = 8

function jump_direction(src, dst, n=N_STRESS)
    src == dst && error("A jump must change state")
    sparse([dst, src], [src, src], [1.0, -1.0], n, n)
end

function background_generator(n=N_STRESS; rate=0.12)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for src in 1:n
        if src > 1
            push!(rows, src - 1); push!(cols, src); push!(vals, rate)
            push!(rows, src);     push!(cols, src); push!(vals, -rate)
        end
        if src < n
            push!(rows, src + 1); push!(cols, src); push!(vals, rate)
            push!(rows, src);     push!(cols, src); push!(vals, -rate)
        end
    end
    sparse(rows, cols, vals, n, n)
end

function normalized_mixture(D1, D2, ρ)
    0.0 <= ρ <= 1.0 || error("ρ must be in [0,1]")
    D = ρ .* D1 + sqrt(max(1 - ρ^2, 0.0)) .* D2
    D ./ (norm(D) / norm(D1))
end

function random_jump(rng, forbidden_sources, used_pairs)
    available_sources = setdiff(1:N_STRESS, forbidden_sources)
    while true
        src = rand(rng, available_sources)
        dst = rand(rng, setdiff(1:N_STRESS, [src]))
        pair = (src, dst)
        pair in used_pairs && continue
        push!(used_pairs, pair)
        return jump_direction(src, dst), pair
    end
end

function build_stress_library(M, correlation, misspecification, seed)
    M >= 8 || error("M must be at least 8")
    rng = MersenneTwister(seed)
    true_pairs = [(5, 9), (11, 16), (20, 15), (26, 22)]
    true_dirs = [jump_direction(src, dst) for (src, dst) in true_pairs]
    true_sources = first.(true_pairs)
    used_pairs = Set(true_pairs)

    # Reserved orthogonal jumps make the requested correlations exact because
    # their nonzeros occupy columns disjoint from each corresponding true jump.
    orthogonal_dirs = SparseMatrixCSC{Float64,Int}[]
    for _ in 1:4
        D, _ = random_jump(rng, true_sources, used_pairs)
        push!(orthogonal_dirs, D)
    end

    dirs = SparseMatrixCSC{Float64,Int}[]
    for j in 1:4
        if j == 4 && misspecification > 0
            ρ_miss = sqrt(max(1 - misspecification^2, 0.0))
            push!(dirs, normalized_mixture(
                true_dirs[j], orthogonal_dirs[j], ρ_miss,
            ))
        else
            push!(dirs, true_dirs[j])
        end
    end

    # Four inactive correlated distractors, one near each active mechanism.
    for j in 1:4
        push!(dirs, normalized_mixture(
            true_dirs[j], orthogonal_dirs[j], correlation,
        ))
    end
    while length(dirs) < M
        D, _ = random_jump(rng, true_sources, used_pairs)
        push!(dirs, D)
    end

    actual_correlations = [
        abs(dot(true_dirs[j], dirs[j + 4])) /
        (norm(true_dirs[j]) * norm(dirs[j + 4]))
        for j in 1:4
    ]
    misspec_correlation = abs(dot(true_dirs[4], dirs[4])) /
                          (norm(true_dirs[4]) * norm(dirs[4]))

    return (
        dirs=dirs,
        true_dirs=true_dirs,
        true_indices=collect(1:4),
        actual_correlation=mean(actual_correlations),
        misspec_correlation=misspec_correlation,
    )
end

function build_exhaustive_edge_library(seed)
    rng = MersenneTwister(seed)
    true_pairs = [(5, 9), (11, 16), (20, 15), (26, 22)]
    all_pairs = [
        (src, dst)
        for src in 1:N_STRESS
        for dst in 1:N_STRESS
        if src != dst
    ]
    shuffle!(rng, all_pairs)
    dirs = [jump_direction(src, dst) for (src, dst) in all_pairs]
    true_indices = [
        only(findall(==(pair), all_pairs)) for pair in true_pairs
    ]
    true_dirs = [
        jump_direction(src, dst) for (src, dst) in true_pairs
    ]
    return (
        dirs=dirs,
        true_dirs=true_dirs,
        true_indices=true_indices,
        actual_correlation=0.0,
        misspec_correlation=1.0,
    )
end

function distribution(weights)
    p = zeros(N_STRESS)
    for (state, weight) in weights
        p[state] = weight
    end
    p ./= sum(p)
    p
end

const TRAIN_INITIALS = [
    distribution([(5, 0.40), (11, 0.30), (20, 0.20), (26, 0.10)]),
    distribution([(5, 0.10), (11, 0.35), (20, 0.15), (26, 0.40)]),
    distribution([(5, 0.20), (11, 0.10), (20, 0.45), (26, 0.25)]),
]
const TEST_INITIAL = distribution([
    (4, 0.10), (5, 0.15), (10, 0.10), (11, 0.15),
    (19, 0.10), (20, 0.15), (25, 0.10), (26, 0.15),
])

softplus_inverse(x) = log(expm1(x))

function make_targets(
    A_true,
    initials=TRAIN_INITIALS,
    times=TRAIN_TIMES,
)
    [
        [exp(Matrix(A_true) * t) * p0 for t in times]
        for p0 in initials
    ]
end

function prediction_loss(
    sl,
    raw,
    mask,
    targets,
    initials=TRAIN_INITIALS,
    times=TRAIN_TIMES,
)
    θ = softplus.(raw) .* mask
    sum(
        0.5 * sum(abs2, targets[i][k] - sl(initials[i], θ, t))
        for i in eachindex(initials)
        for (k, t) in enumerate(times)
    ) / (length(initials) * length(times))
end

function train_stress_case(
    M,
    correlation,
    misspecification,
    seed;
    exhaustive=false,
)
    Random.seed!(seed)
    A0 = background_generator()
    library = exhaustive ?
        build_exhaustive_edge_library(seed) :
        build_stress_library(M, correlation, misspecification, seed)
    M = length(library.dirs)
    A_true = A0 + sum(
        TRUE_RATES[j] * library.true_dirs[j] for j in 1:4
    )
    targets = make_targets(A_true)
    sl = SemigroupLayer(
        A0, library.dirs;
        m_krylov=KRYLOV_M, n_quad=N_QUAD,
    )

    initial_rate = exhaustive ? 0.01 : 0.06
    raw = fill(softplus_inverse(initial_rate), M) .+
          0.05 .* randn(M)
    mask_all = ones(M)
    opt = Flux.setup(Flux.Adam(2e-2), raw)
    for _ in 1:STAGE1_EPOCHS
        _, grads = Zygote.withgradient(raw) do r
            θ = softplus.(r)
            prediction_loss(sl, r, mask_all, targets) + STRESS_L1 * sum(θ)
        end
        Flux.update!(opt, raw, grads[1])
    end

    θ_stage1 = softplus.(raw)
    selected = findall(>=(SUPPORT_THRESHOLD), θ_stage1)
    isempty(selected) && (selected = [argmax(θ_stage1)])
    mask = zeros(M)
    mask[selected] .= 1.0

    opt_refit = Flux.setup(Flux.Adam(8e-3), raw)
    for _ in 1:STAGE2_EPOCHS
        _, grads = Zygote.withgradient(raw) do r
            prediction_loss(sl, r, mask, targets)
        end
        Flux.update!(opt_refit, raw, grads[1])
    end
    θ_hat = softplus.(raw) .* mask
    selected = findall(>(SUPPORT_THRESHOLD), θ_hat)

    true_support = Set(library.true_indices)
    selected_set = Set(selected)
    tp = length(intersect(true_support, selected_set))
    fp = length(setdiff(selected_set, true_support))
    fn = length(setdiff(true_support, selected_set))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    support_f1 = 2 * precision * recall / max(precision + recall, eps())
    best_available_exact = selected_set == true_support
    support_interpretable = misspecification == 0
    exact_support = support_interpretable ? best_available_exact : missing
    reported_precision = support_interpretable ? precision : missing
    reported_recall = support_interpretable ? recall : missing
    reported_f1 = support_interpretable ? support_f1 : missing

    coefficient_error = misspecification == 0 ?
        norm(θ_hat[library.true_indices] - TRUE_RATES) /
        norm(TRUE_RATES) : NaN

    A_hat = A0 + sum(θ_hat[j] * library.dirs[j] for j in 1:M)
    heldout_errors = [
        norm(exp(Matrix(A_hat) * t) * TEST_INITIAL -
             exp(Matrix(A_true) * t) * TEST_INITIAL) /
        norm(exp(Matrix(A_true) * t) * TEST_INITIAL)
        for t in TRAIN_TIMES
    ]
    long_error = norm(
        exp(Matrix(A_hat) * LONG_TIME) * TEST_INITIAL -
        exp(Matrix(A_true) * LONG_TIME) * TEST_INITIAL,
    ) / norm(exp(Matrix(A_true) * LONG_TIME) * TEST_INITIAL)

    return (
        true_support=repr(sort(library.true_indices)),
        selected=repr(selected),
        selected_size=length(selected),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        support_interpretable=support_interpretable,
        best_available_exact=best_available_exact,
        exact_support=exact_support,
        support_precision=reported_precision,
        support_recall=reported_recall,
        support_f1=reported_f1,
        coefficient_relerr=coefficient_error,
        heldout_prediction_relerr=mean(heldout_errors),
        long_horizon_relerr=long_error,
        train_loss=prediction_loss(sl, raw, mask, targets),
        actual_correlation=library.actual_correlation,
        misspec_correlation=library.misspec_correlation,
    )
end

function scenario_list()
    STRESS_EXHAUSTIVE && return [(
        sweep="all_edges",
        M=N_STRESS * (N_STRESS - 1),
        correlation=0.0,
        misspecification=0.0,
        exhaustive=true,
    )]
    STRESS_SMOKE && return [
        (
            sweep="smoke", M=10, correlation=0.0,
            misspecification=0.0, exhaustive=false,
        ),
    ]
    scenarios = NamedTuple[]
    for M in (10, 20, 40, 80)
        push!(scenarios, (
            sweep="library_size", M=M,
            correlation=0.0, misspecification=0.0,
            exhaustive=false,
        ))
    end
    for correlation in (0.9, 0.99, 0.999)
        push!(scenarios, (
            sweep="correlation", M=40,
            correlation=correlation, misspecification=0.0,
            exhaustive=false,
        ))
    end
    for misspecification in (0.01, 0.05, 0.10, 0.20)
        push!(scenarios, (
            sweep="misspecification", M=40,
            correlation=0.0, misspecification=misspecification,
            exhaustive=false,
        ))
    end
    scenarios
end

function checkpoint(raw_rows)
    mkpath(STRESS_OUTPUT_DIR)
    CSV.write(
        joinpath(
            STRESS_OUTPUT_DIR,
            STRESS_EXHAUSTIVE ?
                "exhaustive_edge_library_raw.csv" :
                "library_robustness_raw.csv",
        ),
        DataFrame(raw_rows),
    )
end

function main()
    raw_rows = NamedTuple[]
    scenarios = scenario_list()
    println("Candidate-library robustness stress test")
    println("  scenarios=$(length(scenarios)) seeds=$(STRESS_SEEDS)")

    for scenario in scenarios, seed in STRESS_SEEDS
        seconds = @elapsed result = train_stress_case(
            scenario.M,
            scenario.correlation,
            scenario.misspecification,
            seed,
            exhaustive=scenario.exhaustive,
        )
        row = merge(
            scenario,
            (seed=seed, train_seconds=seconds),
            result,
        )
        push!(raw_rows, row)
        checkpoint(raw_rows)
        @printf(
            "%-17s M=%2d ρ=%.3f ε=%.2f seed=%d |S|=%d F1=%.2f pred=%.2e long=%.2e (%.1fs)\n",
            scenario.sweep,
            scenario.M,
            scenario.correlation,
            scenario.misspecification,
            seed,
            result.selected_size,
            coalesce(result.support_f1, NaN),
            result.heldout_prediction_relerr,
            result.long_horizon_relerr,
            seconds,
        )
        flush(stdout)
    end

    raw_df = DataFrame(raw_rows)
    groups = groupby(
        raw_df,
        [:sweep, :M, :correlation, :misspecification],
    )
    finite_mean(x) = let values = filter(isfinite, x)
        isempty(values) ? NaN : mean(values)
    end
    nonmissing_mean(x) = let values = collect(skipmissing(x))
        isempty(values) ? missing : mean(values)
    end
    nonmissing_std(x) = let values = collect(skipmissing(x))
        length(values) < 2 ? missing : std(values)
    end
    summary = combine(
        groups,
        nrow => :n_seeds,
        :support_interpretable => all => :support_interpretable,
        :best_available_exact => mean => :best_available_exact_rate,
        :exact_support => nonmissing_mean => :exact_support_rate,
        :support_f1 => nonmissing_mean => :support_f1_mean,
        :support_f1 => nonmissing_std => :support_f1_std,
        :selected_size => mean => :selected_size_mean,
        :coefficient_relerr => finite_mean =>
            :coefficient_relerr_mean,
        :heldout_prediction_relerr => mean =>
            :heldout_prediction_relerr_mean,
        :heldout_prediction_relerr => std =>
            :heldout_prediction_relerr_std,
        :long_horizon_relerr => mean => :long_horizon_relerr_mean,
        :long_horizon_relerr => std => :long_horizon_relerr_std,
        :actual_correlation => mean => :actual_correlation_mean,
        :misspec_correlation => mean => :misspec_correlation_mean,
        :train_seconds => mean => :train_seconds_mean,
    )
    CSV.write(
        joinpath(
            STRESS_OUTPUT_DIR,
            STRESS_EXHAUSTIVE ?
                "exhaustive_edge_library_summary.csv" :
                "library_robustness_summary.csv",
        ),
        summary,
    )
    println("\nSaved robustness results to $STRESS_OUTPUT_DIR")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
