using CairoMakie, DelimitedFiles

const RESULT_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_scaling_results.csv")

if !isfile(RESULT_PATH)
    error("Missing $(RESULT_PATH). Run experiments/lindblad_scaling.jl first.")
end

raw, header = readdlm(RESULT_PATH, ','; header=true)
header = vec(String.(header))
idx = Dict(name => i for (i, name) in enumerate(header))

ns = Int.(raw[:, idx["n"]])
d2s = Int.(raw[:, idx["d2"]])
Ms = Int.(raw[:, idx["M"]])
ambient_dims = Int.(raw[:, idx["ambient_dim"]])
ratios = Float64.(raw[:, idx["representation_ratio"]])
param_err_pct = 100 .* Float64.(raw[:, idx["mean_param_err"]])

# n=5 analytic continuation for the representation-size ratio.
n5_ambient = 4^10
n5_ratio = n5_ambient / 3

fig = Figure(size=(980, 310), fontsize=13, figure_padding=(8, 8, 8, 8))

ax1 = Axis(fig[1, 1],
    title="(A) Unrestricted channel dimension reference",
    xlabel="Number of qubits n",
    ylabel="Reference size",
    yscale=log10,
    xticks=(2:5, ["2", "3", "4", "5"]),
    yticks=([3, 256, 4096, 65536, 1048576], ["3", "256", "4k", "65k", "1M"]),
    xgridvisible=false, ygridvisible=true, ygridcolor=(:gray, 0.2))

lines!(ax1, ns, ambient_dims, color=:crimson, linewidth=2.5, label="ambient dimension d⁴")
scatter!(ax1, ns, ambient_dims, color=:crimson, markersize=9)
lines!(ax1, [4, 5], [ambient_dims[end], n5_ambient], color=:crimson, linewidth=2.5, linestyle=:dash)
scatter!(ax1, [5], [n5_ambient], color=:crimson, markersize=9, marker=:diamond)
lines!(ax1, [2, 5], [3, 3], color=:steelblue, linewidth=2.5, linestyle=:dash,
       label="3 full-state snapshots")
text!(ax1, 4.1, n5_ambient * 0.55; text="$(round(Int, n5_ambient / 1000))k",
      fontsize=11, color=:crimson)
axislegend(ax1, position=:lt, labelsize=11, framevisible=false)
xlims!(ax1, 1.7, 5.3)

ax2 = Axis(fig[1, 2],
    title="(B) Representation-size ratio",
    xlabel="Number of qubits n",
    ylabel="d⁴ / 3",
    yscale=log10,
    xticks=(2:5, ["2", "3", "4", "5"]),
    yticks=([85, 1365, 21845, 349525], ["85×", "1,365×", "21,845×", "350k×"]),
    xgridvisible=false, ygridvisible=true, ygridcolor=(:gray, 0.2))

lines!(ax2, ns, ratios, color=:steelblue, linewidth=2.5)
scatter!(ax2, ns, ratios, color=:steelblue, markersize=9)
lines!(ax2, [4, 5], [ratios[end], n5_ratio], color=:steelblue, linewidth=2.5, linestyle=:dash)
scatter!(ax2, [5], [n5_ratio], color=:steelblue, markersize=9, marker=:diamond)
for (n, ratio) in zip(ns, ratios)
    text!(ax2, n + 0.08, ratio;
          text=ratio < 10000 ? "$(round(Int, ratio))×" : "$(round(Int, ratio / 1000))k×",
          fontsize=10, color=:steelblue, align=(:left, :center))
end
text!(ax2, 5.08, n5_ratio; text="350k×\n(analytic)", fontsize=10,
      color=(:steelblue, 0.7), align=(:left, :center))
xlims!(ax2, 1.7, 5.5)

ax3 = Axis(fig[1, 3],
    title="(C) Parameter recovery (M = 2n−1 params)",
    xlabel="Number of qubits n",
    ylabel="Mean relative error (%)",
    xticks=(2:4, ["2", "3", "4"]),
    xgridvisible=false, ygridvisible=true,
    ygridcolor=(:gray, 0.2),
    limits=((1.7, 4.5), (0, 10)))

lines!(ax3, ns, param_err_pct, color=:darkorange, linewidth=2.5)
scatter!(ax3, ns, param_err_pct, color=:darkorange, markersize=9)
for (n, err, m) in zip(ns, param_err_pct, Ms)
    text!(ax3, n, err + 0.35; text="M=$(m)\n$(round(err, digits=1))%",
          fontsize=10, color=:darkorange, align=(:center, :bottom))
end
hlines!(ax3, [10], color=(:gray, 0.4), linewidth=1, linestyle=:dot)
text!(ax3, 4.4, 10.2; text="10%", fontsize=9, color=(:gray, 0.6),
      align=(:right, :bottom))

save("paper/figures/fig_lindblad_scaling.pdf", fig)
println("Saved paper/figures/fig_lindblad_scaling.pdf")
