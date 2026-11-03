"""
Sparse full-state Lindblad parameter recovery beyond four qubits.

Environment variables:
  LINDBLAD_LARGE_NS          comma-separated qubit counts (default: 5,6)
  LINDBLAD_LARGE_SEEDS       comma-separated seeds (default: 42)
  LINDBLAD_LARGE_EPOCHS      default epochs per run (default: 800)
  LINDBLAD_LARGE_EPOCHS_N5   optional per-size override
  LINDBLAD_LARGE_EPOCHS_N6   optional per-size override
  LINDBLAD_LARGE_HIDDEN      encoder width (default: 64)
  LINDBLAD_LARGE_OUTPUT_DIR  result directory (default: paper/figures)

The summary and per-parameter CSVs are checkpointed after every completed run.
"""

using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames

include("lindblad_structured.jl")

const LARGE_NS = parse.(
    Int,
    split(get(ENV, "LINDBLAD_LARGE_NS", "5,6"), ","),
)
const LARGE_SEEDS = parse.(
    Int,
    split(get(ENV, "LINDBLAD_LARGE_SEEDS", "42"), ","),
)
const LARGE_DEFAULT_EPOCHS = parse(
    Int,
    get(ENV, "LINDBLAD_LARGE_EPOCHS", "800"),
)
const LARGE_HIDDEN = parse(Int, get(ENV, "LINDBLAD_LARGE_HIDDEN", "64"))
const LARGE_OUTPUT_DIR = get(
    ENV,
    "LINDBLAD_LARGE_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "paper", "figures"),
)

epochs_for(n) = parse(
    Int,
    get(ENV, "LINDBLAD_LARGE_EPOCHS_N$(n)", string(LARGE_DEFAULT_EPOCHS)),
)

summary_rows = NamedTuple[]
parameter_rows = NamedTuple[]

function checkpoint_results()
    mkpath(LARGE_OUTPUT_DIR)
    CSV.write(
        joinpath(LARGE_OUTPUT_DIR, "lindblad_large_recovery.csv"),
        DataFrame(summary_rows),
    )
    CSV.write(
        joinpath(LARGE_OUTPUT_DIR, "lindblad_large_recovery_parameters.csv"),
        DataFrame(parameter_rows),
    )
end

println("Large sparse Lindblad parameter recovery")
println("  qubits=$(LARGE_NS) seeds=$(LARGE_SEEDS) hidden=$(LARGE_HIDDEN)")

for n in LARGE_NS, seed in LARGE_SEEDS
    n_epochs = epochs_for(n)
    model_ref = Ref{Any}()
    train_seconds = @elapsed begin
        model_ref[] = train_lindblad_model(
            n;
            n_epochs=n_epochs,
            hidden=LARGE_HIDDEN,
            seed=seed,
            verbose=true,
            obs_mode=:full_state,
            sparse_operators=true,
        )
    end
    model = model_ref[]
    @assert issparse(model.A_const)
    @assert all(issparse, model.dirs)

    evaluation = evaluate_lindblad_model(model)
    run_param_errs = [row.rel_err for row in evaluation.param_rows]
    operator_bytes = Base.summarysize(model.A_const) +
                     sum(Base.summarysize, model.dirs; init=0)
    workspace_bytes = sizeof(ComplexF64) * (
        2 * model.d2 * model.mk + 2 * model.mk^2 +
        2 * model.d2 + 2 * model.d2 * 5 + model.d2
    )

    push!(summary_rows, (
        n=n,
        seed=seed,
        hilbert_dim=model.d,
        liouville_dim=model.d2,
        M=model.M,
        epochs=n_epochs,
        hidden=LARGE_HIDDEN,
        krylov_m=model.mk,
        train_contexts=length(model.train_ctxs),
        test_contexts=length(model.test_ctxs),
        observation_times=length(model.abs_times),
        train_seconds=train_seconds,
        mean_param_err=evaluation.mean_param_err,
        median_param_err=median(run_param_errs),
        max_param_err=maximum(run_param_errs),
        mean_prediction_loss=evaluation.mean_prediction_loss,
        sparse_operator_mb=operator_bytes / 2.0^20,
        krylov_workspace_mb=workspace_bytes / 2.0^20,
    ))

    for row in evaluation.param_rows
        push!(parameter_rows, merge(
            (seed=seed, epochs=n_epochs, sparse_operators=true),
            row,
        ))
    end
    checkpoint_results()

    result = summary_rows[end]
    @printf(
        "Completed n=%d seed=%d: param err=%.2f%% median=%.2f%% max=%.2f%% loss=%.3e time=%.1fs\n",
        n, seed,
        100 * result.mean_param_err,
        100 * result.median_param_err,
        100 * result.max_param_err,
        result.mean_prediction_loss,
        result.train_seconds,
    )
    flush(stdout)
    GC.gc()
end

println("\nSaved large-recovery results to $(LARGE_OUTPUT_DIR)")
