"""
Canonical full-state Lindblad scaling experiment.

This is the source-of-truth experiment for the structured Lindblad results in
the paper. It trains the heterogeneous n-qubit model for n = 2, 3, 4, reports
parameter recovery, and writes a CSV summary consumed by
`experiments/lindblad_figure.jl`.
"""

using DelimitedFiles, Printf

include("lindblad_structured.jl")

const SCALING_RESULT_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_scaling_results.csv")

function write_scaling_results(path, rows)
    open(path, "w") do io
        println(io, "n,d2,M,ambient_dim,representation_ratio,mean_param_err,mean_prediction_loss")
        for row in rows
            @printf(io, "%d,%d,%d,%d,%.10f,%.10f,%.10f\n",
                    row.n, row.d2, row.M, row.ambient_dim,
                    row.representation_ratio, row.mean_param_err, row.mean_prediction_loss)
        end
    end
end

function run_scaling_experiment(; runs=[(2, 600), (3, 600), (4, 800)])
    rows = NamedTuple[]
    println("Structured Lindblad scaling experiment")
    println("Reference metric: unrestricted channel ambient dimension d^4")

    for (n, epochs) in runs
        model = train_lindblad_model(n; n_epochs=epochs, verbose=true)
        eval = evaluate_lindblad_model(model)
        row = (
            n=n,
            d2=model.d2,
            M=model.M,
            ambient_dim=eval.ambient_dim,
            representation_ratio=eval.representation_ratio,
            mean_param_err=eval.mean_param_err,
            mean_prediction_loss=eval.mean_prediction_loss,
        )
        push!(rows, row)
        @printf("  n=%d  d²=%d  M=%d  mean param err=%.2f%%  mean loss=%.4e\n",
                row.n, row.d2, row.M, 100 * row.mean_param_err, row.mean_prediction_loss)
    end

    write_scaling_results(SCALING_RESULT_PATH, rows)
    println("\nSaved scaling results → $(SCALING_RESULT_PATH)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_scaling_experiment()
end
