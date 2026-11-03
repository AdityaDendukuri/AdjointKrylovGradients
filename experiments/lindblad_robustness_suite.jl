"""
Structured Lindblad robustness suite.

This script adds the three highest-value stress tests for the current paper:
  1. observation-budget ablation across multiple linear measurement maps
  2. finite-shot robustness for the partial-observation Pauli setting
  3. seed aggregation for both of the above
"""

using DelimitedFiles, Printf, Statistics

include("lindblad_structured.jl")

const ABLATION_RAW_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_ablation_raw.csv")
const ABLATION_SUMMARY_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_ablation_summary.csv")
const NOISE_RAW_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_noise_raw.csv")
const NOISE_SUMMARY_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_noise_summary.csv")

mean_std(xs) = (mean(xs), length(xs) > 1 ? std(xs) : 0.0)

function write_rows(path, header, rows, formats)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            vals = map(f -> f(row), formats)
            println(io, join(vals, ","))
        end
    end
end

function summarize_rows(rows, key_fields, metric_fields)
    groups = Dict{Tuple, Vector{NamedTuple}}()
    for row in rows
        key = Tuple(getproperty(row, field) for field in key_fields)
        push!(get!(groups, key, NamedTuple[]), row)
    end

    summary = NamedTuple[]
    for key in sort!(collect(keys(groups)))
        grp = groups[key]
        base_pairs = [field => key[i] for (i, field) in enumerate(key_fields)]
        metric_pairs = Pair{Symbol, Any}[]
        for field in metric_fields
            μ, σ = mean_std([getproperty(row, field) for row in grp])
            push!(metric_pairs, Symbol(field, "_mean") => μ)
            push!(metric_pairs, Symbol(field, "_std") => σ)
        end
        push!(summary, (; base_pairs..., metric_pairs...))
    end
    return summary
end

function run_observation_ablation(; runs=[(2, 600), (3, 600), (4, 800)],
                                  obs_modes=[:full_state, :z_local, :pauli_single, :pauli_local],
                                  seeds=[11, 22, 33])
    rows = NamedTuple[]
    println("Structured Lindblad observation-budget ablation")

    for obs_mode in obs_modes
        println("\nObservation mode: $(obs_mode)")
        for (n, epochs) in runs
            for seed in seeds
                model = train_lindblad_model(n; n_epochs=epochs, verbose=false,
                                             obs_mode=obs_mode, seed=seed)
                eval = evaluate_lindblad_model(model)
                row = (
                    n=n,
                    obs_mode=string(obs_mode),
                    seed=seed,
                    observation_dim=eval.observation_dim,
                    observation_count=eval.observation_count,
                    mean_param_err=eval.mean_param_err,
                    mean_prediction_loss=eval.mean_prediction_loss,
                )
                push!(rows, row)
                @printf("  n=%d  seed=%d  obs_dim=%d  err=%.2f%%  loss=%.3e\n",
                        n, seed, row.observation_dim, 100 * row.mean_param_err, row.mean_prediction_loss)
            end
        end
    end

    summary = summarize_rows(rows, [:n, :obs_mode, :observation_dim, :observation_count],
                             [:mean_param_err, :mean_prediction_loss])

    write_rows(ABLATION_RAW_PATH,
               ["n", "obs_mode", "seed", "observation_dim", "observation_count", "mean_param_err", "mean_prediction_loss"],
               rows,
               [r -> string(r.n), r -> r.obs_mode, r -> string(r.seed), r -> string(r.observation_dim),
                r -> string(r.observation_count), r -> @sprintf("%.10f", r.mean_param_err),
                r -> @sprintf("%.10f", r.mean_prediction_loss)])

    write_rows(ABLATION_SUMMARY_PATH,
               ["n", "obs_mode", "observation_dim", "observation_count", "mean_param_err_mean", "mean_param_err_std",
                "mean_prediction_loss_mean", "mean_prediction_loss_std"],
               summary,
               [r -> string(r.n), r -> r.obs_mode, r -> string(r.observation_dim), r -> string(r.observation_count),
                r -> @sprintf("%.10f", r.mean_param_err_mean), r -> @sprintf("%.10f", r.mean_param_err_std),
                r -> @sprintf("%.10f", r.mean_prediction_loss_mean), r -> @sprintf("%.10f", r.mean_prediction_loss_std)])

    println("\nSaved observation ablation → $(ABLATION_SUMMARY_PATH)")
    return rows, summary
end

function run_partial_noise_sweep(; runs=[(2, 600), (3, 600), (4, 800)],
                                 shots_list=[0, 100, 1000, 10000],
                                 seeds=[11, 22, 33], obs_mode=:pauli_local)
    rows = NamedTuple[]
    println("Structured Lindblad finite-shot robustness")

    for shots in shots_list
        println("\nShots per observable: $(shots == 0 ? "noiseless" : string(shots))")
        for (n, epochs) in runs
            for seed in seeds
                model = train_lindblad_model(n; n_epochs=epochs, verbose=false,
                                             obs_mode=obs_mode, seed=seed, shots=shots)
                eval = evaluate_lindblad_model(model)
                row = (
                    n=n,
                    obs_mode=string(obs_mode),
                    shots=shots,
                    seed=seed,
                    observation_dim=eval.observation_dim,
                    observation_count=eval.observation_count,
                    mean_param_err=eval.mean_param_err,
                    mean_prediction_loss=eval.mean_prediction_loss,
                )
                push!(rows, row)
                @printf("  n=%d  seed=%d  shots=%d  err=%.2f%%  loss=%.3e\n",
                        n, seed, shots, 100 * row.mean_param_err, row.mean_prediction_loss)
            end
        end
    end

    summary = summarize_rows(rows, [:n, :obs_mode, :shots, :observation_dim, :observation_count],
                             [:mean_param_err, :mean_prediction_loss])

    write_rows(NOISE_RAW_PATH,
               ["n", "obs_mode", "shots", "seed", "observation_dim", "observation_count", "mean_param_err", "mean_prediction_loss"],
               rows,
               [r -> string(r.n), r -> r.obs_mode, r -> string(r.shots), r -> string(r.seed),
                r -> string(r.observation_dim), r -> string(r.observation_count),
                r -> @sprintf("%.10f", r.mean_param_err), r -> @sprintf("%.10f", r.mean_prediction_loss)])

    write_rows(NOISE_SUMMARY_PATH,
               ["n", "obs_mode", "shots", "observation_dim", "observation_count", "mean_param_err_mean", "mean_param_err_std",
                "mean_prediction_loss_mean", "mean_prediction_loss_std"],
               summary,
               [r -> string(r.n), r -> r.obs_mode, r -> string(r.shots), r -> string(r.observation_dim),
                r -> string(r.observation_count), r -> @sprintf("%.10f", r.mean_param_err_mean),
                r -> @sprintf("%.10f", r.mean_param_err_std),
                r -> @sprintf("%.10f", r.mean_prediction_loss_mean), r -> @sprintf("%.10f", r.mean_prediction_loss_std)])

    println("\nSaved partial-observation noise sweep → $(NOISE_SUMMARY_PATH)")
    return rows, summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_observation_ablation()
    println()
    run_partial_noise_sweep()
end
