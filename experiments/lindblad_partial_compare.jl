"""
Compare full-state and partial-observation structured Lindblad fitting.

The partial-observation mode uses a linear map from vec(ρ_t) to a compact set of
single-site and nearest-neighbour Pauli expectations, while keeping the same
generator family, contexts, and snapshot times as the canonical full-state
experiment.
"""

using DelimitedFiles, Printf

include("lindblad_structured.jl")

const PARTIAL_RESULT_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_observation_compare.csv")

function write_observation_results(path, rows)
    open(path, "w") do io
        println(io, "n,obs_mode,observation_dim,snapshot_count,observation_count,mean_param_err,mean_prediction_loss")
        for row in rows
            @printf(io, "%d,%s,%d,%d,%d,%.10f,%.10f\n",
                    row.n, row.obs_mode, row.observation_dim, row.snapshot_count,
                    row.observation_count, row.mean_param_err, row.mean_prediction_loss)
        end
    end
end

function run_observation_comparison(; runs=[(2, 600), (3, 600), (4, 800)],
                                     obs_modes=[:full_state, :pauli_local])
    rows = NamedTuple[]
    println("Structured Lindblad observation comparison")

    for obs_mode in obs_modes
        println("\nObservation mode: $(obs_mode)")
        for (n, epochs) in runs
            model = train_lindblad_model(n; n_epochs=epochs, verbose=true, obs_mode=obs_mode)
            eval = evaluate_lindblad_model(model)
            row = (
                n=n,
                obs_mode=string(obs_mode),
                observation_dim=eval.observation_dim,
                snapshot_count=eval.snapshot_count,
                observation_count=eval.observation_count,
                mean_param_err=eval.mean_param_err,
                mean_prediction_loss=eval.mean_prediction_loss,
            )
            push!(rows, row)
            @printf("  n=%d  obs_dim=%d  mean param err=%.2f%%  mean loss=%.4e\n",
                    row.n, row.observation_dim, 100 * row.mean_param_err, row.mean_prediction_loss)
        end
    end

    write_observation_results(PARTIAL_RESULT_PATH, rows)
    println("\nSaved observation comparison → $(PARTIAL_RESULT_PATH)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_observation_comparison()
end
