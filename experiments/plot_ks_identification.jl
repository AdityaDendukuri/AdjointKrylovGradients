using CairoMakie
using CSV
using DataFrames
using Statistics

root = length(ARGS) >= 1 ? ARGS[1] : error("repository root required")
out = length(ARGS) >= 2 ? ARGS[2] : error("output path required")
data = joinpath(root, "paper", "figures")

composition = CSV.read(
    joinpath(data, "kuramoto_sivashinsky_composable_raw.csv"), DataFrame)
support = CSV.read(
    joinpath(data, "kuramoto_sivashinsky_support_hard_raw.csv"), DataFrame)

set_theme!(theme_latexfonts())
update_theme!(
    fontsize = 13,
    Axis = (
        topspinevisible = false,
        rightspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
    ),
)

fig = Figure(size = (1120, 350))

# (a) The input-state pullback is the only route to the nonlinear coefficient c.
ax = Axis(
    fig[1, 1],
    xlabel = "Coefficient",
    ylabel = "Estimate",
    title = "(a) Composition pullback",
    xticks = (1:3, ["a", "b", "c"]),
)
hlines!(ax, [1.0], color = :gray45, linestyle = :dot, linewidth = 1.5)
for (k, (method, label, marker, shift)) in enumerate([
    ("full_state_pullback", "Full pullback", :circle, -0.10),
    ("zero_state_ablation", "Input cotangent removed", :diamond, 0.10),
])
    sub = filter(row -> row.method == method, composition)
    for (j, name) in enumerate([:estimate_a, :estimate_b, :estimate_c])
        vals = sub[!, name]
        scatter!(
            ax,
            fill(j + shift, length(vals)),
            vals,
            marker = marker,
            color = :white,
            strokecolor = :black,
            strokewidth = 1.2,
            markersize = 7,
            label = j == 1 ? label : nothing,
        )
        lines!(ax, [j + shift - 0.05, j + shift + 0.05], fill(mean(vals), 2),
               color = :black, linewidth = 2.0)
    end
end
axislegend(ax, position = :lb, labelsize = 9)

# (b) Noisy nine-term discovery: all seeds separate active from inactive terms.
ax = Axis(
    fig[1, 2],
    xlabel = "Library direction",
    ylabel = "Discovery coefficient",
    title = "(b) Noisy 3-of-9 discovery",
    xticks = (1:9, ["-Dx", "-Dxx", "DxDxx", "-Dxxxx", "q=2", "q=3", "q=4", "q=5", "q=6"]),
    xticklabelrotation = pi / 4,
)
disc = filter(row -> row.stage == "l1_discovery", support)
for i in 1:nrow(disc)
    vals = [disc[i, Symbol("theta_$(j)")] for j in 1:9]
    scatter!(ax, 1:9, vals, marker = :circle, color = :white,
             strokecolor = :black, strokewidth = 0.8, markersize = 5)
end
means = [mean(disc[!, Symbol("theta_$(j)")]) for j in 1:9]
scatter!(ax, 1:9, means, marker = :diamond, color = :black, markersize = 7)
hlines!(ax, [0.08], color = :black, linestyle = :dash, linewidth = 1.3,
        label = "threshold")
text!(ax, 8.9, 0.10, text = "threshold 0.08", align = (:right, :bottom),
      fontsize = 10)

# (c) Compare the ablation and sparse-library stages on physical error metrics.
full = filter(row -> row.method == "full_state_pullback", composition)
zero = filter(row -> row.method == "zero_state_ablation", composition)
disc = filter(row -> row.stage == "l1_discovery", support)
refit = filter(row -> row.stage == "masked_refit", support)
groups = [full, zero, disc, refit]
labels = ["Full", "No input\ncotangent", "Discovery", "Masked\nrefit"]
parameter = [
    100mean(full.parameter_relerr),
    100mean(zero.parameter_relerr),
    100mean(disc.coefficient_relerr),
    100mean(refit.coefficient_relerr),
]
rollout = [
    100mean(full.heldout_rollout_relerr),
    100mean(zero.heldout_rollout_relerr),
    100mean(disc.heldout_rollout_relerr),
    100mean(refit.heldout_rollout_relerr),
]
ax = Axis(
    fig[1, 3],
    xlabel = "Fit",
    ylabel = "Relative error (%)",
    yscale = log10,
    yticks = ([0.03, 0.1, 0.3, 1, 3, 10, 30],
              ["0.03", "0.1", "0.3", "1", "3", "10", "30"]),
    title = "(c) Parameter and rollout error",
    xticks = (1:4, labels),
)
scatterlines!(ax, (1:4) .- 0.08, parameter, color = :black, marker = :circle,
              markersize = 7, linestyle = :solid, label = "Parameters")
scatterlines!(ax, (1:4) .+ 0.08, rollout, color = :black, marker = :diamond,
              markersize = 7, linestyle = :dash, label = "Held-out rollout")
axislegend(ax, position = :rb, labelsize = 9)

save(out, fig)
