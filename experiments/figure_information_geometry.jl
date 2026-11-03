# =============================================================================
# Figure: Jacobian-column scaling and optimization trajectories
#
# Panel layout (2 × 2):
#   (a) Singular-value spectra of J_{S*}  — raw           (all 4 systems)
#   (b) Same after Fisher column scaling                    (conditioning gain)
#   (c) 2D loss slice in (w₁,w₂) + STLS trajectories without scaling (Schlögl)
#   (d) Same contours   + STLS trajectories with scaling (Schlögl)
#
# Contours are an honest 2D landscape visualization in the top two
# active-subspace directions.  Panels (a,b) show the conditioning change.
# Panels (c,d) show how the conditioning difference affects optimizer paths.
#
# Usage: julia --project=experiments experiments/figure_information_geometry.jl
# =============================================================================

using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "bw_style.jl"))
using AdjointKrylovGradients, LinearAlgebra, Printf, Statistics
using CairoMakie, LaTeXStrings
using ExponentialUtilities

set_siam_theme!()

const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

# =============================================================================
# Helpers
# =============================================================================

function cme_forward(A, p0, dt, T; m=30, tol=1e-11)
    P = Vector{Vector{Float64}}(undef, T + 1)
    P[1] = copy(p0)
    for k in 1:T
        P[k+1] = expv(dt, A, P[k]; m=m, tol=tol)
    end
    return P
end

function obs_jacobian(θ_flat, P, R, X, dt, dA, active_idx; feature_map, K)
    n = length(X); T = length(P) - 1
    θ = unflatten_params(θ_flat, R, K)
    A = build_generator(X, R, θ; feature_map=feature_map)
    param_list = [(ζ, f) for ζ in sort(collect(R)) for f in 1:K]
    s = length(active_idx)
    J = zeros(n * T, s)
    for k in 1:T, (col, pidx) in enumerate(active_idx)
        ζ, f = param_list[pidx]
        J[(k-1)*n+1:k*n, col] .= frechet_action(A * dt, dA[ζ][f] * dt, P[k])
    end
    return J
end

function active_subspace(J)
    C  = J'J
    F  = eigen(Symmetric(C))
    ord = sortperm(F.values, rev=true)
    return (; C, λ=F.values[ord], W=F.vectors[:, ord])
end

function loss_2d(θf, act_idx, P_data, R, X, dt, w1, w2;
                 feature_map, K, ns=30, srange=0.55)
    sg = range(-srange, srange, length=ns)
    L  = zeros(ns, ns)
    for (i, s1) in enumerate(sg), (j, s2) in enumerate(sg)
        θn = copy(θf)
        θn[act_idx] .*= exp.(s1 .* w1 .+ s2 .* w2)
        L[i, j] = objective(P_data, R, unflatten_params(θn, R, K), X, dt;
                            feature_map=feature_map)
    end
    Lmax = maximum(L[isfinite.(L)])
    return collect(sg), L ./ Lmax
end

"""Project θ_snapshots onto (w₁,w₂) log-ratio coordinates relative to θ*.
Zeroed active params are replaced with a tiny floor so trajectories that lose
a param are pushed far off-screen (large negative log-ratio) rather than dropped.
Makie clips those segments at the axis boundary, showing the optimizer leaving the
displayed region."""
function project_trajectory(snapshots, θf_true, act_idx, w1, w2)
    coords = Tuple{Float64,Float64}[]
    θ_true_act = θf_true[act_idx]
    for θs in snapshots
        vals = θs[act_idx]
        all(isfinite, vals) || continue
        # Replace non-positive values with a small floor for log-ratio visualization.
        vals_vis = max.(vals, 1e-8 .* θ_true_act)
        δ = log.(vals_vis ./ θ_true_act)
        all(isfinite, δ) || continue
        push!(coords, (dot(w1, δ), dot(w2, δ)))
    end
    return coords
end

function fisher_scale_jacobian(J)
    col_norms = [norm(J[:, j]) for j in 1:size(J, 2)]
    col_norms[col_norms .< 1e-30] .= 1.0
    return J ./ col_norms'
end

# =============================================================================
# System setup
# =============================================================================

# ── Michaelis–Menten ─────────────────────────────────────────────────────────
R_MM = [(-1,-1,1,0), (1,1,-1,0), (0,1,-1,1)]; K_MM = feature_dimension(4)
X_mm  = [(50-se-p, 10-se, se, p) for se in 0:10 for p in 0:(50-se)]
idx_mm = Dict(x => i for (i,x) in enumerate(X_mm))
θ_mm  = Dict(ζ => zeros(K_MM) for ζ in R_MM)
θ_mm[(-1,-1,1,0)][10]=0.01; θ_mm[(1,1,-1,0)][4]=0.1; θ_mm[(0,1,-1,1)][4]=0.1
θf_mm = flatten_params(θ_mm); act_mm = findall(!iszero, θf_mm)
A_mm  = build_generator(X_mm, R_MM, θ_mm; feature_map=quad_features)
p0_mm = zeros(length(X_mm)); p0_mm[idx_mm[(50,10,0,0)]] = 1.0
dA_mm = build_derivatives(X_mm, R_MM; feature_map=quad_features, K=K_MM)

# ── Lotka–Volterra ────────────────────────────────────────────────────────────
R_LV = [(1,0), (-1,1), (0,-1)]; K_LV = feature_dimension(2)
X_lv  = [(x,y) for x in 0:60 for y in 0:40]
idx_lv = Dict(x => i for (i,x) in enumerate(X_lv))
θ_lv  = Dict(ζ => zeros(K_LV) for ζ in R_LV)
θ_lv[(1,0)][1]=1.0; θ_lv[(-1,1)][6]=0.01; θ_lv[(0,-1)][3]=0.5
θf_lv = flatten_params(θ_lv); act_lv = findall(!iszero, θf_lv)
A_lv  = build_generator(X_lv, R_LV, θ_lv; feature_map=quad_features)
p0_lv = zeros(length(X_lv)); p0_lv[idx_lv[(50,20)]] = 1.0
dA_lv = build_derivatives(X_lv, R_LV; feature_map=quad_features, K=K_LV)

# ── Schlögl ───────────────────────────────────────────────────────────────────
R_SCH = [(1,), (-1,)]; K_SCH = cubic_feature_dimension(1)
X_sch   = [(x,) for x in 0:80]
idx_sch = Dict(x => i for (i,x) in enumerate(X_sch))
θ_sch   = Dict(ζ => zeros(K_SCH) for ζ in R_SCH)
θ_sch[(1,)][1]=1.7; θ_sch[(1,)][3]=0.017; θ_sch[(-1,)][2]=0.2; θ_sch[(-1,)][4]=0.0006
θf_sch  = flatten_params(θ_sch); act_sch = findall(!iszero, θf_sch)
A_sch   = build_generator(X_sch, R_SCH, θ_sch; feature_map=cubic_features)
p0_sch  = zeros(length(X_sch)); p0_sch[idx_sch[(25,)]] = 1.0
dA_sch  = build_derivatives(X_sch, R_SCH; feature_map=cubic_features, K=K_SCH)

# ── Brusselator ───────────────────────────────────────────────────────────────
R_BRUS = [(1,0),(1,-1),(-1,1),(-1,0)]; K_BRUS = cubic_feature_dimension(2)
X_brus   = [(x,y) for x in 0:20 for y in 0:25]
idx_brus = Dict(x => i for (i,x) in enumerate(X_brus))
θ_brus   = Dict(ζ => zeros(K_BRUS) for ζ in R_BRUS)
θ_brus[(1,0)][1]=1.0; θ_brus[(1,-1)][9]=1.0; θ_brus[(-1,1)][2]=3.0; θ_brus[(-1,0)][2]=1.0
θf_brus  = flatten_params(θ_brus); act_brus = findall(!iszero, θf_brus)
A_brus   = build_generator(X_brus, R_BRUS, θ_brus; feature_map=cubic_features)
p0_brus  = zeros(length(X_brus)); p0_brus[idx_brus[(1,1)]] = 1.0
dA_brus  = build_derivatives(X_brus, R_BRUS; feature_map=cubic_features, K=K_BRUS)

# =============================================================================
# Forward solves and Jacobians
# =============================================================================

println("Forward solves..."); flush(stdout)
P_mm   = cme_forward(A_mm,   p0_mm,   1.0, 10)
P_lv   = cme_forward(A_lv,   p0_lv,   1.0, 20)
P_sch  = cme_forward(A_sch,  p0_sch,  0.5, 40)
P_brus = cme_forward(A_brus, p0_brus, 1.0, 25)

println("Jacobians..."); flush(stdout)
J_mm   = obs_jacobian(θf_mm,  P_mm,  R_MM,  X_mm,  1.0, dA_mm,  act_mm;  feature_map=quad_features,  K=K_MM)
J_lv   = obs_jacobian(θf_lv,  P_lv,  R_LV,  X_lv,  1.0, dA_lv,  act_lv;  feature_map=quad_features,  K=K_LV)
J_sch  = obs_jacobian(θf_sch, P_sch, R_SCH, X_sch, 0.5, dA_sch, act_sch; feature_map=cubic_features, K=K_SCH)
J_brus = obs_jacobian(θf_brus,P_brus,R_BRUS,X_brus,1.0, dA_brus,act_brus;feature_map=cubic_features, K=K_BRUS)

as_sch = active_subspace(J_sch)
w1_sch = as_sch.W[:, 1]
w2_sch = as_sch.W[:, 2]

# =============================================================================
# Loss landscape (Schlögl, active-subspace coordinates)
# =============================================================================

println("Loss landscape..."); flush(stdout)
sg, L_sch = loss_2d(θf_sch, act_sch, P_sch, R_SCH, X_sch, 0.5, w1_sch, w2_sch;
                    feature_map=cubic_features, K=K_SCH)

# =============================================================================
# STLS trajectories: 4 initializations × {no PRFP, PRFP}
# =============================================================================

println("STLS trajectories..."); flush(stdout)

init_coords = [(0.38, 0.22), (-0.38, 0.22), (0.25, -0.38), (-0.28, -0.32)]

function make_init(s1, s2)
    θ_init_flat = copy(θf_sch)
    θ_init_flat[act_sch] .*= exp.(s1 .* w1_sch .+ s2 .* w2_sch)
    return unflatten_params(θ_init_flat, R_SCH, K_SCH)
end

function run_trajectory(θ_init; fisher_precond)
    result = optimize_stls(
        [(P_sch, X_sch)], R_SCH, 0.5, θ_init;
        τ=0.01, outer_iter=40, polish_iter=80,
        verbose=false, verbose_history=true,
        feature_map=cubic_features,
        fisher_precond=fisher_precond,
        per_reaction_threshold=fisher_precond
    )
    snaps        = get(result.history, "θ_snapshots",        Vector{Float64}[])
    polish_snaps = get(result.history, "polish_θ_snapshots", Vector{Float64}[])
    return project_trajectory(vcat(snaps, polish_snaps), θf_sch, act_sch, w1_sch, w2_sch)
end

trajs_no = Vector{Vector{Tuple{Float64,Float64}}}()
trajs_fp = Vector{Vector{Tuple{Float64,Float64}}}()

for (s1, s2) in init_coords
    θ0 = make_init(s1, s2)
    push!(trajs_no, run_trajectory(θ0; fisher_precond=false))
    push!(trajs_fp, run_trajectory(θ0; fisher_precond=true))
    @printf("  init (%.2f, %.2f) done\n", s1, s2); flush(stdout)
end

# =============================================================================
# Plot
# =============================================================================

Js        = [J_mm, J_lv, J_sch, J_brus]
Js_scaled = fisher_scale_jacobian.(Js)
sys_labels  = ["MM", "LV", "Schl.", "Brus."]
markers     = [:circle, :rect, :diamond, :utriangle]
spec_styles = [:solid, :dash, :dot, :dashdot]

fig = Figure(size=(680, 600), figure_padding=12)

ax_raw = Axis(fig[1,1], xlabel=L"i", ylabel=L"\sigma_i(\mathbf{J}_{S^*})",
              title="(a) raw spectra", yscale=log10, xticks=1:4)
ax_fsh = Axis(fig[1,2], xlabel=L"i", ylabel=L"\sigma_i(\mathbf{J}_{S^*}\mathbf{D}^{-1})",
              title="(b) Fisher-scaled spectra", yscale=log10, xticks=1:4)
ax_no  = Axis(fig[2,1], xlabel=L"s_1\;(\mathbf{w}_1)",
              ylabel=L"s_2\;(\mathbf{w}_2)", title="(c) without scaling", aspect=DataAspect())
ax_fp  = Axis(fig[2,2], xlabel=L"s_1\;(\mathbf{w}_1)",
              ylabel=L"s_2\;(\mathbf{w}_2)", title="(d) with scaling",    aspect=DataAspect())

# Spectra ─────────────────────────────────────────────────────────────────────
for (J, Jsc, lbl, mk, ls) in zip(Js, Js_scaled, sys_labels, markers, spec_styles)
    s_raw = sort(svdvals(J),   rev=true)[1:min(4,size(J,2))]
    s_fsh = sort(svdvals(Jsc), rev=true)[1:min(4,size(Jsc,2))]
    xs = 1:length(s_raw)
    scatterlines!(ax_raw, xs, s_raw; label=lbl, marker=mk, markersize=7, linewidth=1.2,
                  color=:black, linestyle=ls)
    scatterlines!(ax_fsh, xs, s_fsh; label=lbl, marker=mk, markersize=7, linewidth=1.2,
                  color=:black, linestyle=ls)
end
axislegend(ax_raw, position=:rt, labelsize=9, framevisible=false)
axislegend(ax_fsh, position=:rt, labelsize=9, framevisible=false)

# Contours + trajectories ─────────────────────────────────────────────────────
clev = range(0.0, 1.0, length=16)
gray_levels = range(0.55, 0.05, length=length(init_coords))

traj_styles = [:solid, :dash, :dot, :dashdot]

for (ax, trajs) in [(ax_no, trajs_no), (ax_fp, trajs_fp)]
    contour!(ax, sg, sg, L_sch'; levels=clev, color=(:black, 0.20), linewidth=0.6)
    # θ* marker
    scatter!(ax, [0.0], [0.0]; marker=:star5, markersize=14,
             color=:white, strokecolor=:black, strokewidth=1.2)

    for (traj, ls) in zip(trajs, traj_styles)
        isempty(traj) && continue
        xs = first.(traj); ys = last.(traj)
        scatter!(ax, [xs[1]], [ys[1]]; marker=:circle, markersize=7,
                 color=:black, strokecolor=:black, strokewidth=0.8)
        lines!(ax, xs, ys; color=:black, linewidth=1.8, linestyle=ls)
        if length(traj) >= 2 && (xs[end] != xs[end-1] || ys[end] != ys[end-1])
            dx = xs[end] - xs[end-1]; dy = ys[end] - ys[end-1]
            arrows2d!(ax, [xs[end-1]], [ys[end-1]], [dx], [dy];
                      tipwidth=0.025, shaftlength=0.0, color=:black)
        end
    end
    xlims!(ax, sg[1], sg[end]); ylims!(ax, sg[1], sg[end])
end

colgap!(fig.layout, 1, 10)
rowgap!(fig.layout, 1, 10)

outpath = joinpath(FIGDIR, "information_geometry.pdf")
save(outpath, fig)
println("Saved → $outpath")
