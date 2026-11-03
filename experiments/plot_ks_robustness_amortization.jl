using CairoMakie
using CSV
using DataFrames
using Statistics

root = length(ARGS) >= 1 ? ARGS[1] : error("repository root required")
out = length(ARGS) >= 2 ? ARGS[2] : error("output path required")
data = joinpath(root, "paper", "figures")

noise_tags = ["noise0000", "noise0025", "noise0050", "noise0100"]
support_frames = DataFrame[]
for tag in noise_tags
    push!(
        support_frames,
        CSV.read(
            joinpath(
                data,
                "kuramoto_sivashinsky_support_robust15_$(tag)_raw.csv",
            ),
            DataFrame,
        ),
    )
end
support = vcat(support_frames...; cols=:union)
timing = CSV.read(
    joinpath(data, "kuramoto_sivashinsky_amortization.csv"),
    DataFrame,
)

set_theme!(theme_latexfonts())
update_theme!(
    fontsize=17,
    Axis=(
        topspinevisible=false,
        rightspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
        spinewidth=1.5,
        xtickwidth=1.5,
        ytickwidth=1.5,
        xlabelpadding=9,
        ylabelpadding=9,
    ),
)

fig = Figure(size=(1120, 390))

# (a) Support remains exact while refitting eventually amplifies noise.
ax = Axis(
    fig[1, 1],
    xlabel="Snapshot noise (% RMS)",
    ylabel="Noise-free test-trajectory error (%)",
    title="(a) Noise and refit tradeoff",
    xticks=([0.0, 0.25, 0.5, 1.0], ["0", "0.25", "0.5", "1"]),
)
discovery = filter(row -> row.stage == "l1_discovery", support)
refit = filter(row -> row.stage == "masked_refit", support)
noise_levels = sort(unique(discovery.relative_noise))
for noise in noise_levels
    discovery_subset = filter(row -> row.relative_noise == noise, discovery)
    refit_subset = filter(row -> row.relative_noise == noise, refit)
    x = 100noise
    jitter = collect(range(-0.018, 0.018; length=nrow(discovery_subset)))
    scatter!(
        ax,
        x .- 0.025 .+ jitter,
        100 .* discovery_subset.heldout_rollout_relerr,
        color=:white,
        strokecolor=:black,
        strokewidth=1.4,
        markersize=6.5,
    )
    scatter!(
        ax,
        x .+ 0.025 .+ jitter,
        100 .* refit_subset.heldout_rollout_relerr,
        marker=:diamond,
        color=:white,
        strokecolor=:black,
        strokewidth=1.4,
        markersize=6.5,
    )
    exact = count(identity, discovery_subset.exact_support)
    text!(
        ax,
        x,
        max(
            maximum(100 .* discovery_subset.heldout_rollout_relerr),
            maximum(100 .* refit_subset.heldout_rollout_relerr),
        ) + 0.035,
        text="$(exact)/$(nrow(discovery_subset)) exact",
        align=(:center, :bottom),
        fontsize=13,
        font=:bold,
    )
end
discovery_means = [
    100mean(filter(row -> row.relative_noise == noise, discovery).heldout_rollout_relerr)
    for noise in noise_levels
]
refit_means = [
    100mean(filter(row -> row.relative_noise == noise, refit).heldout_rollout_relerr)
    for noise in noise_levels
]
scatterlines!(
    ax,
    100 .* noise_levels .- 0.025,
    discovery_means,
    color=:black,
    linewidth=2.5,
    markersize=10,
    marker=:circle,
    label="Discovery",
)
scatterlines!(
    ax,
    100 .* noise_levels .+ 0.025,
    refit_means,
    color=:black,
    linewidth=2.5,
    linestyle=:dash,
    marker=:diamond,
    markersize=10,
    label="Masked refit",
)
axislegend(ax, position=:lt, labelsize=13, framecolor=:black, framewidth=1.2)

# (b) Larger-domain amortization: the expensive bases are reused over M.
ax = Axis(
    fig[1, 2],
    xlabel="Affine library size",
    ylabel="Time for all parameter derivatives (ms)",
    yscale=log10,
    title="(b) Shared-Krylov amortization, n=$(first(timing.state_dimension))",
    xticks=(timing.library_size, string.(timing.library_size)),
)
shared_ms = 1e3 .* timing.shared_seconds
repeated_ms = 1e3 .* timing.repeated_seconds
scatterlines!(
    ax,
    timing.library_size,
    shared_ms,
    color=:black,
    marker=:circle,
    linewidth=2.5,
    markersize=10,
    label="Shared bases",
)
scatterlines!(
    ax,
    timing.library_size,
    repeated_ms,
    color=:black,
    marker=:diamond,
    linestyle=:dash,
    linewidth=2.5,
    markersize=10,
    label="Bases rebuilt per direction",
)
for row in eachrow(timing)
    text!(
        ax,
        row.library_size,
        sqrt((1e3 * row.shared_seconds) * (1e3 * row.repeated_seconds)),
        text="$(round(row.speedup; digits=1))×",
        align=(:center, :center),
        fontsize=13,
        font=:bold,
    )
end
axislegend(ax, position=:lt, labelsize=13, framecolor=:black, framewidth=1.2)

colgap!(fig.layout, 42)
save(out, fig)
