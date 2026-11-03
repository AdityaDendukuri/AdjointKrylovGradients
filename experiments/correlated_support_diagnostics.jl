"""
Focused diagnostic for support recovery with correlated library directions.

The experiment separates effects from excitation, regularization, thresholding,
and optimization. It uses the same CTMC and training procedure as
library_robustness.jl.

Environment variables:
  CORR_DIAG_SEEDS          comma-separated seeds (default: 11,22,33,44,55)
  CORR_DIAG_RHOS           comma-separated correlations (default: 0.9,0.99,0.999)
  CORR_DIAG_L1S            comma-separated L1 values for the current design
                           (default: 5e-5,2e-4,8e-4)
  CORR_DIAG_THRESHOLDS     comma-separated diagnostic cutoffs
  CORR_DIAG_STAGE1_EPOCHS  Adam selection epochs (default: 400)
  CORR_DIAG_STAGE2_EPOCHS  iterative masked-refit epochs (default: 250)
  CORR_DIAG_LBFGS_ITERS    post-Adam L-BFGS iterations (default: 300)
  CORR_DIAG_OUTPUT_DIR     output directory (default: paper/figures)

Outputs:
  correlated_support_diagnostic_raw.csv
  correlated_support_diagnostic_summary.csv
  correlated_support_thresholds.csv
  correlated_support_threshold_summary.csv
"""

using Optim

include("library_robustness.jl")

parse_list(::Type{T}, name, default) where {T} =
    parse.(T, split(get(ENV, name, default), ","))

const DIAG_SEEDS = parse_list(
    Int,
    "CORR_DIAG_SEEDS",
    "11,22,33,44,55",
)
const DIAG_RHOS = parse_list(
    Float64,
    "CORR_DIAG_RHOS",
    "0.9,0.99,0.999",
)
const DIAG_L1S = parse_list(
    Float64,
    "CORR_DIAG_L1S",
    "5e-5,2e-4,8e-4",
)
const DIAG_THRESHOLDS = parse_list(
    Float64,
    "CORR_DIAG_THRESHOLDS",
    "0.02,0.04,0.08,0.12,0.16,0.24,0.32,0.48,0.64",
)
const DIAG_STAGE1_EPOCHS = parse(
    Int,
    get(ENV, "CORR_DIAG_STAGE1_EPOCHS", "400"),
)
const DIAG_STAGE2_EPOCHS = parse(
    Int,
    get(ENV, "CORR_DIAG_STAGE2_EPOCHS", "250"),
)
const DIAG_LBFGS_ITERS = parse(
    Int,
    get(ENV, "CORR_DIAG_LBFGS_ITERS", "300"),
)
const DIAG_OUTPUT_DIR = get(
    ENV,
    "CORR_DIAG_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)
const DIAG_BASELINE_L1 = 2.0e-4
const DIAG_DEFAULT_THRESHOLD = 0.08

function fixed_broad_initials()
    rng = MersenneTwister(2027)
    initials = Vector{Vector{Float64}}()
    for _ in 1:4
        p = rand(rng, N_STRESS)
        push!(initials, p ./ sum(p))
    end
    initials
end

const BROAD_INITIALS = fixed_broad_initials()

function prediction_loss_theta(sl, θ, targets, initials)
    sum(
        0.5 * sum(abs2, targets[i][k] - sl(initials[i], θ, t))
        for i in eachindex(initials)
        for (k, t) in enumerate(TRAIN_TIMES)
    ) / (length(initials) * length(TRAIN_TIMES))
end

function stage1_objective(sl, raw, targets, initials, l1)
    θ = softplus.(raw)
    prediction_loss_theta(sl, θ, targets, initials) + l1 * sum(θ)
end

function train_adam(sl, raw_start, targets, initials, l1)
    raw = copy(raw_start)
    optimizer = Flux.setup(Flux.Adam(2.0e-2), raw)
    seconds = @elapsed for _ in 1:DIAG_STAGE1_EPOCHS
        _, grads = Zygote.withgradient(raw) do r
            stage1_objective(sl, r, targets, initials, l1)
        end
        Flux.update!(optimizer, raw, grads[1])
    end
    raw, seconds
end

function polish_lbfgs(sl, raw_start, targets, initials, l1)
    objective(raw) = stage1_objective(
        sl,
        raw,
        targets,
        initials,
        l1,
    )
    function gradient!(storage, raw)
        _, grads = Zygote.withgradient(raw) do r
            stage1_objective(sl, r, targets, initials, l1)
        end
        storage .= grads[1]
        nothing
    end

    result = optimize(
        objective,
        gradient!,
        copy(raw_start),
        LBFGS(),
        Optim.Options(
            iterations=DIAG_LBFGS_ITERS,
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

function iterative_refit(
    sl,
    raw_start,
    targets,
    initials,
    threshold,
)
    raw = copy(raw_start)
    mask = Float64.(softplus.(raw) .>= threshold)
    all(iszero, mask) && (mask[argmax(raw)] = 1.0)
    rounds = 0
    seconds = @elapsed for round in 1:5
        rounds = round
        optimizer = Flux.setup(Flux.Adam(8.0e-3), raw)
        for _ in 1:DIAG_STAGE2_EPOCHS
            _, grads = Zygote.withgradient(raw) do r
                θ = softplus.(r) .* mask
                prediction_loss_theta(sl, θ, targets, initials)
            end
            Flux.update!(optimizer, raw, grads[1])
        end

        θ = softplus.(raw) .* mask
        new_mask = mask .* Float64.(θ .>= threshold)
        if new_mask == mask
            break
        end
        mask = new_mask
    end
    θ = softplus.(raw) .* mask
    θ[θ .< threshold] .= 0.0
    (
        theta=θ,
        raw=raw,
        rounds=rounds,
        seconds=seconds,
    )
end

function polish_masked_lbfgs(
    sl,
    raw_start,
    targets,
    initials,
    threshold,
)
    raw = copy(raw_start)
    mask = Float64.(softplus.(raw) .>= threshold)
    all(iszero, mask) && (mask[argmax(raw)] = 1.0)
    rounds = 0
    total_iterations = 0
    converged = true

    seconds = @elapsed for round in 1:5
        rounds = round
        objective(r) = prediction_loss_theta(
            sl,
            softplus.(r) .* mask,
            targets,
            initials,
        )
        function gradient!(storage, r)
            _, grads = Zygote.withgradient(r) do candidate
                prediction_loss_theta(
                    sl,
                    softplus.(candidate) .* mask,
                    targets,
                    initials,
                )
            end
            storage .= grads[1]
            nothing
        end
        result = optimize(
            objective,
            gradient!,
            raw,
            LBFGS(),
            Optim.Options(
                iterations=DIAG_LBFGS_ITERS,
                show_trace=false,
                store_trace=false,
            ),
        )
        raw = Optim.minimizer(result)
        total_iterations += Optim.iterations(result)
        converged &= Optim.converged(result)

        θ = softplus.(raw) .* mask
        new_mask = mask .* Float64.(θ .>= threshold)
        if new_mask == mask
            break
        end
        mask = new_mask
    end

    θ = softplus.(raw) .* mask
    θ[θ .< threshold] .= 0.0
    (
        theta=θ,
        raw=raw,
        rounds=rounds,
        seconds=seconds,
        converged=converged,
        iterations=total_iterations,
    )
end

function support_metrics(θ, threshold)
    selected = findall(>=(threshold), θ)
    selected_set = Set(selected)
    truth = Set(1:4)
    tp = length(intersect(truth, selected_set))
    fp = length(setdiff(selected_set, truth))
    fn = length(setdiff(truth, selected_set))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, eps())
    (
        selected=repr(selected),
        selected_size=length(selected),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        exact_support=selected_set == truth,
        support_precision=precision,
        support_recall=recall,
        support_f1=f1,
    )
end

function prediction_metrics(A0, library, θ, A_true)
    A_hat = A0 + sum(θ[j] * library.dirs[j] for j in eachindex(θ))
    heldout = mean(
        norm(
            exp(Matrix(A_hat) * t) * TEST_INITIAL -
            exp(Matrix(A_true) * t) * TEST_INITIAL,
        ) / norm(exp(Matrix(A_true) * t) * TEST_INITIAL)
        for t in TRAIN_TIMES
    )
    long_horizon = norm(
        exp(Matrix(A_hat) * LONG_TIME) * TEST_INITIAL -
        exp(Matrix(A_true) * LONG_TIME) * TEST_INITIAL,
    ) / norm(exp(Matrix(A_true) * LONG_TIME) * TEST_INITIAL)
    generator_error = norm(Matrix(A_hat - A_true)) / norm(Matrix(A_true))
    (
        heldout_prediction_relerr=heldout,
        long_horizon_relerr=long_horizon,
        generator_relerr=generator_error,
    )
end

function local_sensitivity_metrics(A_true, dirs, initials)
    n = size(A_true, 1)
    paired_dirs = dirs[1:8]
    sensitivity = zeros(
        n * length(initials) * length(TRAIN_TIMES),
        length(paired_dirs),
    )
    for (j, direction) in enumerate(paired_dirs)
        for (time_index, t) in enumerate(TRAIN_TIMES)
            block = Matrix([
                A_true direction
                spzeros(n, n) A_true
            ])
            frechet = exp(block * t)[1:n, n+1:2n]
            for (initial_index, p0) in enumerate(initials)
                offset = (
                    (time_index - 1) * length(initials) +
                    initial_index - 1
                ) * n
                rows = offset .+ (1:n)
                sensitivity[rows, j] = frechet * p0
            end
        end
    end

    norms = [norm(sensitivity[:, j]) for j in 1:8]
    normalized = sensitivity ./ reshape(norms, 1, :)
    singular_values = svdvals(normalized)
    pair_cosine = mean(
        abs(dot(normalized[:, j], normalized[:, j + 4]))
        for j in 1:4
    )
    (
        sensitivity_condition=singular_values[1] / singular_values[end],
        sensitivity_min_singular=singular_values[end],
        sensitivity_pair_cosine=pair_cosine,
    )
end

function evaluate_theta(
    θ,
    raw,
    sl,
    targets,
    initials,
    l1,
    A0,
    library,
    A_true,
    sensitivity,
    metadata;
    optimizer_converged=missing,
    optimizer_iterations=missing,
    train_seconds=missing,
    refit_rounds=missing,
)
    objective = prediction_loss_theta(sl, θ, targets, initials) +
                l1 * sum(θ)
    θ_true = zeros(length(θ))
    θ_true[1:4] .= TRUE_RATES
    true_objective = prediction_loss_theta(
        sl,
        θ_true,
        targets,
        initials,
    ) + l1 * sum(θ_true)
    false_max = maximum(θ[5:end])
    correlated_max = maximum(θ[5:8])
    other_max = maximum(θ[9:end])
    true_min = minimum(θ[1:4])
    support = support_metrics(θ, DIAG_DEFAULT_THRESHOLD)
    prediction = prediction_metrics(A0, library, θ, A_true)
    gradient_norm = if raw === nothing
        missing
    else
        norm(Zygote.gradient(
            r -> stage1_objective(sl, r, targets, initials, l1),
            raw,
        )[1])
    end

    merge(
        metadata,
        (
            optimizer_converged=optimizer_converged,
            optimizer_iterations=optimizer_iterations,
            train_seconds=train_seconds,
            refit_rounds=refit_rounds,
            objective=objective,
            true_support_objective=true_objective,
            objective_gap=objective - true_objective,
            gradient_norm=gradient_norm,
            data_loss=prediction_loss_theta(sl, θ, targets, initials),
            true_coefficient_min=true_min,
            false_coefficient_max=false_max,
            correlated_coefficient_max=correlated_max,
            other_coefficient_max=other_max,
            separation_margin=true_min - false_max,
            separating_threshold_exists=false_max < true_min,
            theta_true_terms=repr(θ[1:4]),
            theta_correlated_terms=repr(θ[5:8]),
            theta_other_max=other_max,
            coefficient_relerr=norm(θ[1:4] - TRUE_RATES) /
                               norm(TRUE_RATES),
        ),
        support,
        prediction,
        sensitivity,
    )
end

function threshold_rows(θ, metadata)
    [
        merge(
            metadata,
            (threshold=threshold,),
            support_metrics(θ, threshold),
        )
        for threshold in DIAG_THRESHOLDS
    ]
end

function checkpoint(raw_rows, threshold_data)
    mkpath(DIAG_OUTPUT_DIR)
    CSV.write(
        joinpath(
            DIAG_OUTPUT_DIR,
            "correlated_support_diagnostic_raw.csv",
        ),
        DataFrame(raw_rows),
    )
    CSV.write(
        joinpath(
            DIAG_OUTPUT_DIR,
            "correlated_support_thresholds.csv",
        ),
        DataFrame(threshold_data),
    )
end

function summarize(raw_rows, threshold_data)
    raw = DataFrame(raw_rows)
    raw_groups = groupby(
        raw,
        [:design, :correlation, :l1, :stage],
    )
    raw_summary = combine(
        raw_groups,
        nrow => :n_seeds,
        :exact_support => mean => :exact_support_rate,
        :support_recall => mean => :support_recall_mean,
        :false_positives => mean => :false_positives_mean,
        :separating_threshold_exists => mean =>
            :separating_threshold_rate,
        :separation_margin => mean => :separation_margin_mean,
        :objective_gap => mean => :objective_gap_mean,
        :gradient_norm => (x -> mean(skipmissing(x))) =>
            :gradient_norm_mean,
        :heldout_prediction_relerr => mean =>
            :heldout_prediction_relerr_mean,
        :long_horizon_relerr => mean => :long_horizon_relerr_mean,
        :generator_relerr => mean => :generator_relerr_mean,
        :sensitivity_condition => mean =>
            :sensitivity_condition_mean,
        :sensitivity_min_singular => mean =>
            :sensitivity_min_singular_mean,
        :sensitivity_pair_cosine => mean =>
            :sensitivity_pair_cosine_mean,
    )
    CSV.write(
        joinpath(
            DIAG_OUTPUT_DIR,
            "correlated_support_diagnostic_summary.csv",
        ),
        raw_summary,
    )

    thresholds = DataFrame(threshold_data)
    threshold_groups = groupby(
        thresholds,
        [:design, :correlation, :l1, :stage, :threshold],
    )
    threshold_summary = combine(
        threshold_groups,
        nrow => :n_seeds,
        :exact_support => mean => :exact_support_rate,
        :support_recall => mean => :support_recall_mean,
        :false_positives => mean => :false_positives_mean,
        :selected_size => mean => :selected_size_mean,
    )
    CSV.write(
        joinpath(
            DIAG_OUTPUT_DIR,
            "correlated_support_threshold_summary.csv",
        ),
        threshold_summary,
    )
end

function main_diagnostic()
    raw_rows = NamedTuple[]
    threshold_data = NamedTuple[]
    broad_design = vcat(TRAIN_INITIALS, BROAD_INITIALS)

    println("Correlated-support diagnostic")
    println("  seeds=$(DIAG_SEEDS)")
    println("  correlations=$(DIAG_RHOS)")
    println("  current-design L1 values=$(DIAG_L1S)")

    for correlation in DIAG_RHOS
        for seed in DIAG_SEEDS
            A0 = background_generator()
            library = build_stress_library(
                40,
                correlation,
                0.0,
                seed,
            )
            A_true = A0 + sum(
                TRUE_RATES[j] * library.true_dirs[j] for j in 1:4
            )
            sl = SemigroupLayer(
                A0,
                library.dirs;
                m_krylov=KRYLOV_M,
                n_quad=N_QUAD,
            )

            Random.seed!(seed)
            raw_start = fill(softplus_inverse(0.06), 40) .+
                        0.05 .* randn(40)

            for (design, initials) in (
                ("current", TRAIN_INITIALS),
                ("broad", broad_design),
            )
                targets = make_targets(A_true, initials)
                sensitivity = local_sensitivity_metrics(
                    A_true,
                    library.dirs,
                    initials,
                )
                l1_values = design == "current" ?
                    DIAG_L1S : [DIAG_BASELINE_L1]

                for l1 in l1_values
                    raw_adam, train_seconds = train_adam(
                        sl,
                        raw_start,
                        targets,
                        initials,
                        l1,
                    )
                    θ_adam = softplus.(raw_adam)
                    metadata = (
                        design=design,
                        correlation=correlation,
                        seed=seed,
                        l1=l1,
                        stage="adam_stage1",
                    )
                    adam_row = evaluate_theta(
                        θ_adam,
                        raw_adam,
                        sl,
                        targets,
                        initials,
                        l1,
                        A0,
                        library,
                        A_true,
                        sensitivity,
                        metadata;
                        train_seconds=train_seconds,
                    )
                    push!(raw_rows, adam_row)
                    append!(
                        threshold_data,
                        threshold_rows(θ_adam, metadata),
                    )
                    @printf(
                        "%-7s ρ=%.3f seed=%d λ=%.1e Adam |S|=%d FP=%d sep=%s gap=%+.2e\n",
                        design,
                        correlation,
                        seed,
                        l1,
                        adam_row.selected_size,
                        adam_row.false_positives,
                        string(adam_row.separating_threshold_exists),
                        adam_row.objective_gap,
                    )
                    flush(stdout)

                    if isapprox(l1, DIAG_BASELINE_L1)
                        refit = iterative_refit(
                            sl,
                            raw_adam,
                            targets,
                            initials,
                            DIAG_DEFAULT_THRESHOLD,
                        )
                        refit_metadata = merge(
                            metadata,
                            (stage="adam_iterative_refit",),
                        )
                        refit_row = evaluate_theta(
                            refit.theta,
                            nothing,
                            sl,
                            targets,
                            initials,
                            l1,
                            A0,
                            library,
                            A_true,
                            sensitivity,
                            refit_metadata;
                            train_seconds=refit.seconds,
                            refit_rounds=refit.rounds,
                        )
                        push!(raw_rows, refit_row)
                        append!(
                            threshold_data,
                            threshold_rows(
                                refit.theta,
                                refit_metadata,
                            ),
                        )
                        @printf(
                            "         refit |S|=%d FP=%d rounds=%d\n",
                            refit_row.selected_size,
                            refit_row.false_positives,
                            refit.rounds,
                        )
                        flush(stdout)
                    end

                    lbfgs_seconds = @elapsed lbfgs = polish_lbfgs(
                        sl,
                        raw_adam,
                        targets,
                        initials,
                        l1,
                    )
                    θ_lbfgs = softplus.(lbfgs.raw)
                    lbfgs_metadata = merge(
                        metadata,
                        (stage="adam_lbfgs_stage1",),
                    )
                    lbfgs_row = evaluate_theta(
                        θ_lbfgs,
                        lbfgs.raw,
                        sl,
                        targets,
                        initials,
                        l1,
                        A0,
                        library,
                        A_true,
                        sensitivity,
                        lbfgs_metadata;
                        optimizer_converged=lbfgs.converged,
                        optimizer_iterations=lbfgs.iterations,
                        train_seconds=lbfgs_seconds,
                    )
                    push!(raw_rows, lbfgs_row)
                    append!(
                        threshold_data,
                        threshold_rows(
                            θ_lbfgs,
                            lbfgs_metadata,
                        ),
                    )
                    @printf(
                        "         L-BFGS |S|=%d FP=%d sep=%s gap=%+.2e\n",
                        lbfgs_row.selected_size,
                        lbfgs_row.false_positives,
                        string(
                            lbfgs_row.separating_threshold_exists,
                        ),
                        lbfgs_row.objective_gap,
                    )
                    flush(stdout)

                    if isapprox(l1, DIAG_BASELINE_L1)
                        polished_refit = polish_masked_lbfgs(
                            sl,
                            lbfgs.raw,
                            targets,
                            initials,
                            DIAG_DEFAULT_THRESHOLD,
                        )
                        polished_refit_metadata = merge(
                            metadata,
                            (stage="adam_lbfgs_refit",),
                        )
                        polished_refit_row = evaluate_theta(
                            polished_refit.theta,
                            nothing,
                            sl,
                            targets,
                            initials,
                            l1,
                            A0,
                            library,
                            A_true,
                            sensitivity,
                            polished_refit_metadata;
                            optimizer_converged=
                                polished_refit.converged,
                            optimizer_iterations=
                                polished_refit.iterations,
                            train_seconds=polished_refit.seconds,
                            refit_rounds=polished_refit.rounds,
                        )
                        push!(raw_rows, polished_refit_row)
                        append!(
                            threshold_data,
                            threshold_rows(
                                polished_refit.theta,
                                polished_refit_metadata,
                            ),
                        )
                        @printf(
                            "         polished refit |S|=%d FP=%d rounds=%d pred=%.2e\n",
                            polished_refit_row.selected_size,
                            polished_refit_row.false_positives,
                            polished_refit.rounds,
                            polished_refit_row.heldout_prediction_relerr,
                        )
                        flush(stdout)
                    end
                    checkpoint(raw_rows, threshold_data)
                end
            end
        end
    end

    summarize(raw_rows, threshold_data)
    println("\nSaved correlated-support diagnostics to $DIAG_OUTPUT_DIR")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main_diagnostic()
end
