"""
Two-qubit visualisation for the canonical structured Lindblad experiment.

This script reuses the heterogeneous full-state model from
`experiments/lindblad_structured.jl` and produces the three-panel figure used in
the paper:
  A. Representative density-matrix element magnitudes at a held-out context
  B. True-vs-predicted structured parameters across held-out contexts
  C. Prediction loss across the context range

Unlike the older version of this script, it does not introduce a separate
quantum-noise task or process-tomography narrative. It is only a 2-qubit view
of the same `M = 2n-1` structured identification problem used in the scaling
experiment.
"""

using CairoMakie, Printf, Statistics

include("lindblad_structured.jl")

const FIG_PATH = joinpath(@__DIR__, "../paper/figures/fig_lindblad.pdf")

function trajectory_loss(model, c, states)
    θ = predict_params(model.net, c, model.n)
    _sg_loss_lindblad(θ, states, model.loss_dts, model.A_const, model.dirs, model.mk,
                      model.obs_model.obs_op, model.obs_model.from_initial)
end

function train_and_plot_context_demo(; n=2, n_epochs=800, c_demo=0.5)
    model = train_lindblad_model(n; n_epochs=n_epochs, verbose=true, obs_mode=:full_state)
    eval = evaluate_lindblad_model(model)

    curves = dense_density_curves(model, c_demo)
    elem_labels = [L"\rho_{11}", L"\rho_{12}", L"\rho_{22}", L"\rho_{14}"]
    elem_idx = [1, 2, 6, 4]
    obs_times = model.abs_times

    train_losses = [
        trajectory_loss(model, c, model.train_data[i]) for (i, c) in enumerate(model.train_ctxs)
    ]
    test_losses = [
        trajectory_loss(model, c, model.test_data[i]) for (i, c) in enumerate(model.test_ctxs)
    ]
    fine_ctxs = collect(range(0.01, 0.99; length=60))
    fine_losses = [
        trajectory_loss(model, c,
                        project_lindblad_states(
                            propagate_lindblad(c, model.n, model.A_const, model.dirs;
                                               abs_times=model.abs_times)[1],
                            model.obs_model))
        for c in fine_ctxs
    ]

    param_groups = [
        ("J1", :royalblue, :circle, 1),
        ("gamma1", :firebrick, :utriangle, 2),
        ("gamma2", :darkorange, :rect, 3),
    ]

    fig = Figure(size=(1200, 380), fontsize=13)

    axA = Axis(fig[1, 1];
        title="(A) Density-matrix elements at c = 0.5",
        xlabel="time",
        ylabel="|ρᵢⱼ(t)|",
        titlesize=14)

    palette = [:royalblue, :firebrick, :darkorange, :seagreen]
    for (lab, idx, col) in zip(elem_labels, elem_idx, palette)
        true_vals = [abs(ρ[idx]) for ρ in curves.ρ_true]
        pred_vals = [abs(ρ[idx]) for ρ in curves.ρ_pred]
        lines!(axA, curves.ts, true_vals; color=col, linewidth=2.2, label="$(lab) true")
        lines!(axA, curves.ts, pred_vals; color=col, linewidth=2.2, linestyle=:dash,
               label="$(lab) pred")
    end
    vlines!(axA, obs_times; color=(:black, 0.25), linestyle=:dot)
    axislegend(axA; position=:rt, labelsize=9, nbanks=2)

    axB = Axis(fig[1, 2];
        title="(B) Structured parameter recovery",
        xlabel="True parameter",
        ylabel="Predicted parameter",
        titlesize=14)

    all_true = Float64[]
    all_pred = Float64[]
    for (label, color, marker, j) in param_groups
        xs = [true_params(c, model.n)[j] for c in model.test_ctxs]
        ys = [predict_params(model.net, c, model.n)[j] for c in model.test_ctxs]
        append!(all_true, xs)
        append!(all_pred, ys)
        scatter!(axB, xs, ys; color=color, marker=marker, markersize=8, label=label)
    end
    lo = min(minimum(all_true), minimum(all_pred)) - 0.03
    hi = max(maximum(all_true), maximum(all_pred)) + 0.03
    lines!(axB, [lo, hi], [lo, hi]; color=:black, linewidth=1.5, linestyle=:dash, label="y = x")
    axislegend(axB; position=:lt, labelsize=10)

    axC = Axis(fig[1, 3];
        title="(C) Prediction loss across context",
        xlabel="Context c",
        ylabel="Prediction loss",
        titlesize=14,
        yscale=log10)

    lines!(axC, fine_ctxs, fine_losses .+ 1e-12; color=:steelblue, linewidth=2.2, label="loss")
    scatter!(axC, model.train_ctxs, train_losses .+ 1e-12;
             color=:royalblue, markersize=9, marker=:circle, label="train")
    scatter!(axC, model.test_ctxs, test_losses .+ 1e-12;
             color=:firebrick, markersize=9, marker=:diamond, label="test")
    axislegend(axC; position=:rt, labelsize=10)

    save(FIG_PATH, fig)
    println("\nSaved $(FIG_PATH)")

    @printf("Mean relative parameter error (n=2): %.2f%%\n", 100 * eval.mean_param_err)
    @printf("Mean held-out prediction loss     : %.4e\n", eval.mean_prediction_loss)
    println("Reference size of unrestricted channel representation: d^4 = $(eval.ambient_dim)")

    return (; model, eval, train_losses, test_losses, fine_ctxs, fine_losses)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_and_plot_context_demo()
end
