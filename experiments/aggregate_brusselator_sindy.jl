"""
Aggregate independent Brusselator Reactive-SINDy result directories.

Each argument must contain:
  paper/figures/brusselator_sindy_hyperparameters.csv
  paper/figures/brusselator_sindy_context_rates.csv

Example:
  julia --project=experiments experiments/aggregate_brusselator_sindy.jl \
      /tmp/bruss_seed2026 /tmp/bruss_seed2027 /tmp/bruss_seed2028
"""

using CSV, DataFrames, Statistics

isempty(ARGS) && error("Pass at least one result directory")

hyper_frames = DataFrame[]
rate_frames = DataFrame[]
for result_dir in ARGS
    figure_dir = joinpath(result_dir, "paper", "figures")
    push!(
        hyper_frames,
        CSV.read(
            joinpath(figure_dir, "brusselator_sindy_hyperparameters.csv"),
            DataFrame,
        ),
    )
    push!(
        rate_frames,
        CSV.read(
            joinpath(figure_dir, "brusselator_sindy_context_rates.csv"),
            DataFrame,
        ),
    )
end

hyper = vcat(hyper_frames...)
rates = vcat(rate_frames...)

rates.A_relerr = abs.(rates.A_pred .- rates.A_true) ./ abs.(rates.A_true)
rates.B_relerr = abs.(rates.B_pred .- rates.B_true) ./ abs.(rates.B_true)
rates.k2_relerr = abs.(rates.k2_pred .- rates.k2_true) ./ abs.(rates.k2_true)
rates.k4_relerr = abs.(rates.k4_pred .- rates.k4_true) ./ abs.(rates.k4_true)

seed_rows = NamedTuple[]
for seed in sort(unique(hyper.seed))
    shared = only(eachrow(hyper[(hyper.seed .== seed) .& (hyper.model .== "shared"), :]))
    context = only(eachrow(hyper[
        (hyper.seed .== seed) .& (hyper.model .== "context-aware"),
        :,
    ]))
    seed_rates = rates[rates.seed .== seed, :]
    push!(seed_rows, (
        seed=seed,
        trajectories_per_context_ic=context.trajectories_per_context_ic,
        shared_true_hits=shared.true_hits,
        shared_spurious=shared.spurious,
        shared_channel_true_hits=shared.channel_true_hits,
        context_true_hits=context.true_hits,
        context_spurious=context.spurious,
        context_channel_true_hits=context.channel_true_hits,
        A_mre_pct=100 * mean(seed_rates.A_relerr),
        B_mre_pct=100 * mean(seed_rates.B_relerr),
        k2_mre_pct=100 * mean(seed_rates.k2_relerr),
        k4_mre_pct=100 * mean(seed_rates.k4_relerr),
    ))
end
seed_summary = DataFrame(seed_rows)

metric_names = [
    "shared_true_hits",
    "shared_spurious",
    "shared_channel_true_hits",
    "context_true_hits",
    "context_spurious",
    "context_channel_true_hits",
    "A_mre_pct",
    "B_mre_pct",
    "k2_mre_pct",
    "k4_mre_pct",
]
overall_rows = NamedTuple[]
for metric in metric_names
    values = Float64.(seed_summary[!, metric])
    push!(overall_rows, (
        metric=metric,
        mean=mean(values),
        std=length(values) > 1 ? std(values) : 0.0,
        minimum=minimum(values),
        maximum=maximum(values),
        n_seeds=length(values),
    ))
end
overall = DataFrame(overall_rows)

out_dir = get(
    ENV,
    "BRUSS_SINDY_AGGREGATE_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)
mkpath(out_dir)
CSV.write(joinpath(out_dir, "brusselator_sindy_multiseed.csv"), seed_summary)
CSV.write(joinpath(out_dir, "brusselator_sindy_multiseed_overall.csv"), overall)
CSV.write(joinpath(out_dir, "brusselator_sindy_context_rates_multiseed.csv"), rates)
CSV.write(joinpath(out_dir, "brusselator_sindy_hyperparameters_multiseed.csv"), hyper)

println(seed_summary)
println("\n", overall)
println("\nSaved multi-seed Brusselator SINDy summaries to $(out_dir)")
