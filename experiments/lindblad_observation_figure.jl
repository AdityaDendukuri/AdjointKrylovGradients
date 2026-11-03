using CairoMakie, DelimitedFiles

const ABLATION_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_ablation_summary.csv")
const NOISE_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_noise_summary.csv")
const OUT_PDF = joinpath(@__DIR__, "../paper/figures/fig_lindblad_observations.pdf")
const OUT_PNG = joinpath(@__DIR__, "../paper/figures/fig_lindblad_observations.png")

for path in (ABLATION_PATH, NOISE_PATH)
    isfile(path) || error("Missing $(path). Run the corresponding Lindblad summary scripts first.")
end

function read_csv_table(path)
    raw, header = readdlm(path, ','; header=true)
    names = vec(String.(header))
    idx = Dict(name => i for (i, name) in enumerate(names))
    return raw, idx
end

ablation_raw, aidx = read_csv_table(ABLATION_PATH)
noise_raw, nidx = read_csv_table(NOISE_PATH)

obs_order = ["full_state", "z_local", "pauli_single", "pauli_local"]
obs_labels = [
    "full state\n(d²)",
    "Z only\n(n+1)",
    "single-site\n(3n+1)",
    "local + edge\n(6n-2)",
]

shot_order = [0, 100, 1000, 10000]
shot_labels = ["noiseless", "100", "1000", "10000"]

ns = [2, 3, 4]
series_labels = ["n = 2", "n = 3", "n = 4"]
series_colors = [:royalblue, :darkorange, :seagreen]
series_markers = [:circle, :rect, :diamond]
offsets = [-0.18, 0.0, 0.18]

fig = Figure(size=(980, 330), fontsize=13, figure_padding=(8, 8, 10, 8))

ax1 = Axis(fig[1, 1],
    title="(A) Observation budget",
    xlabel="Observation set",
    ylabel="Mean relative error (%)",
    xticks=(1:4, obs_labels),
    yscale=log10,
    yticks=([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1000"]),
    xgridvisible=false, ygridvisible=true, ygridcolor=(:gray, 0.2))

for (k, n) in enumerate(ns)
    ys = Float64[]
    yerrs = Float64[]
    xs = Float64[]
    for (j, mode) in enumerate(obs_order)
        mask = (String.(ablation_raw[:, aidx["obs_mode"]]) .== mode) .&
               (Int.(ablation_raw[:, aidx["n"]]) .== n)
        row = ablation_raw[findfirst(mask), :]
        push!(xs, j + offsets[k])
        push!(ys, 100 * Float64(row[aidx["mean_param_err_mean"]]))
        push!(yerrs, 100 * Float64(row[aidx["mean_param_err_std"]]))
    end
    lines!(ax1, xs, ys; color=series_colors[k], linewidth=2.2)
    ylow = max.(ys .- yerrs, 1e-2)
    yhigh = ys .+ yerrs
    for i in eachindex(xs)
        lines!(ax1, [xs[i], xs[i]], [ylow[i], yhigh[i]]; color=series_colors[k], linewidth=1.8)
        lines!(ax1, [xs[i] - 0.04, xs[i] + 0.04], [ylow[i], ylow[i]]; color=series_colors[k], linewidth=1.8)
        lines!(ax1, [xs[i] - 0.04, xs[i] + 0.04], [yhigh[i], yhigh[i]]; color=series_colors[k], linewidth=1.8)
    end
    scatter!(ax1, xs, ys; color=series_colors[k], marker=series_markers[k], markersize=10,
             label=series_labels[k])
end

axislegend(ax1, position=:lt, framevisible=false, labelsize=10)
xlims!(ax1, 0.55, 4.45)

ax2 = Axis(fig[1, 2],
    title="(B) Finite-shot robustness",
    xlabel="Shots per observable",
    ylabel="Mean relative error (%)",
    xticks=(1:4, shot_labels),
    xgridvisible=false, ygridvisible=true, ygridcolor=(:gray, 0.2),
    limits=((0.7, 4.3), (0, 20)))

for (k, n) in enumerate(ns)
    ys = Float64[]
    yerrs = Float64[]
    xs = Float64[]
    for (j, shots) in enumerate(shot_order)
        mask = (Int.(noise_raw[:, nidx["shots"]]) .== shots) .&
               (Int.(noise_raw[:, nidx["n"]]) .== n)
        row = noise_raw[findfirst(mask), :]
        push!(xs, j)
        push!(ys, 100 * Float64(row[nidx["mean_param_err_mean"]]))
        push!(yerrs, 100 * Float64(row[nidx["mean_param_err_std"]]))
    end
    lines!(ax2, xs, ys; color=series_colors[k], linewidth=2.2)
    errorbars!(ax2, xs, ys .- yerrs, ys .+ yerrs; color=series_colors[k], whiskerwidth=10)
    scatter!(ax2, xs, ys; color=series_colors[k], marker=series_markers[k], markersize=10)
end

text!(ax2, 4.05, 2.0;
      text="means ± std over 3 seeds",
      fontsize=10, color=(:gray, 0.65), align=(:right, :bottom))

save(OUT_PDF, fig)
save(OUT_PNG, fig)
println("Saved $(OUT_PDF)")
println("Saved $(OUT_PNG)")
