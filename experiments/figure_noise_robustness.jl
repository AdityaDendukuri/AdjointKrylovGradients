# figure_noise_robustness.jl
# Shows thresholded-optimization trajectories under different initialization noise.
#
# Uses verbose_history=true to get:
#   - per-outer-iteration objectives from the thresholding phase
#   - per-inner-step objectives from the polish phase (via extended_trace)
# Concatenates both to show the full gradient-step trajectory.
#
# Columns : MM  |  LV  |  Brusselator
# Row 1   : objective L(θ) vs gradient step (log scale)
# Row 2   : relative rate error ‖θ̂-θ*‖/‖θ*‖ vs gradient step (log scale)
# Output: paper/figures/noise_robustness.pdf

using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "bw_style.jl"))
using AdjointKrylovGradients, Catalyst, JumpProcesses, Printf, Random, Statistics, LinearAlgebra
using LaTeXStrings

Random.seed!(42)
set_siam_theme!()

const NOISE_LEVELS = [0.3, 1.0, 2.0, 5.0]
const LINESTYLES   = [:solid, :dash, :dot, :dashdot]
const GRAYS        = ["black", "#444444", "#777777", "#aaaaaa"]
const NOISE_LABELS = ["σ = 0.3", "σ = 1.0", "σ = 2.0", "σ = 5.0"]

function full_history(res, θ0_flat, obj0, rate_fn, true_vec, R, K)
    # STLS phase: one objective per outer iteration
    stls_objs = vcat(obj0, res.history["objective"])
    # Polish phase: per-L-BFGS-step objectives (stored in polish_objectives)
    pol_objs  = get(res.history, "polish_objectives", Float64[])
    all_objs  = vcat(stls_objs, pol_objs)

    # Rate errors: STLS snapshots
    stls_snaps = get(res.history, "θ_snapshots", Vector{Float64}[])
    stls_errs  = [begin
        θd = unflatten_params(s, R, K)
        rec = rate_fn(θd)
        norm(abs.(rec .- true_vec) ./ true_vec)
    end for s in stls_snaps]
    # Polish snapshots
    pol_snaps  = get(res.history, "polish_θ_snapshots", Vector{Float64}[])
    pol_errs   = [begin
        θd = unflatten_params(s, R, K)
        rec = rate_fn(θd)
        norm(abs.(rec .- true_vec) ./ true_vec)
    end for s in pol_snaps]

    # Initial error from θ0
    err0 = norm(abs.(rate_fn(unflatten_params(θ0_flat, R, K)) .- true_vec) ./ true_vec)
    all_errs = vcat(err0, stls_errs, pol_errs)

    # x-axis: cumulative gradient steps
    n_stls = length(stls_objs)   # includes the t=0 initial value
    n_pol  = length(pol_objs)
    n_err  = length(all_errs)
    xs_obj = 0:n_stls+n_pol-1
    xs_err = range(0, n_stls+n_pol-1, length=n_err)
    stls_end_x = n_stls - 1   # x value where polish begins

    return xs_obj, all_objs, xs_err, all_errs, stls_end_x
end

# ─── Michaelis-Menten ─────────────────────────────────────────────────────────
println("="^50, "\nGenerating MM data..."); flush(stdout)
K_ON=0.01; K_OFF=0.1; K_CAT=0.1
R_mm  = [(-1,-1,1,0),(1,1,-1,0),(0,1,-1,1)]
rn_mm = @reaction_network begin
    k1, S+E --> SE; k2, SE --> S+E; k3, SE --> P+E
end
trajs_mm = [begin
    dprob = DiscreteProblem(rn_mm,[:S=>50,:E=>10,:SE=>0,:P=>0],(0.0,10.0),
                            [:k1=>K_ON,:k2=>K_OFF,:k3=>K_CAT])
    solve(JumpProblem(rn_mm,dprob,Direct()),SSAStepper())
end for _ in 1:1000]
times_mm = collect(0.0:1.0:10.0)
X_mm = build_state_space(trajs_mm,times_mm,R_mm;expansion=2,verbose=false)
P_mm = build_distributions(trajs_mm,times_mm,X_mm)
td_mm = count_transitions(trajs_mm,R_mm;verbose=false)
K_mm = feature_dimension(4)
mm_rate_fn(θd) = [θd[(-1,-1,1,0)][10], θd[(1,1,-1,0)][4], θd[(0,1,-1,1)][4]]
mm_true = [K_ON, K_OFF, K_CAT]

mm_data = []
for σ in NOISE_LEVELS
    @printf("  MM σ=%.1f...\n",σ); flush(stdout)
    θ0  = empirical_initialization(td_mm,R_mm;verbose=false,noise_scale=σ)
    obj0 = objective(P_mm,R_mm,θ0,X_mm,1.0)
    res  = optimize_stls(P_mm,R_mm,X_mm,1.0,θ0;
                         τ=0.02,outer_iter=40,polish_iter=80,
                         verbose=false,verbose_history=true,
                         gradient_fn=gradient_adjoint_krylov!)
    push!(mm_data, full_history(res,flatten_params(θ0),obj0,mm_rate_fn,mm_true,R_mm,K_mm))
end

# ─── Lotka-Volterra ───────────────────────────────────────────────────────────
println("Generating LV data..."); flush(stdout)
ALPHA=1.0; BETA=0.01; GAMMA=0.5
R_lv  = [(1,0),(-1,1),(0,-1)]
rn_lv = @reaction_network begin
    alpha, 0-->X; beta, X+Y-->2Y; gamma, Y-->0
end
trajs_lv = [begin
    dprob = DiscreteProblem(rn_lv,[:X=>50,:Y=>20],(0.0,50.0),
                            [:alpha=>ALPHA,:beta=>BETA,:gamma=>GAMMA])
    solve(JumpProblem(rn_lv,dprob,Direct()),SSAStepper())
end for _ in 1:500]
times_lv = collect(0.0:1.0:50.0)
X_lv = build_state_space(trajs_lv,times_lv,R_lv;expansion=2,verbose=false)
P_lv = build_distributions(trajs_lv,times_lv,X_lv)
td_lv = count_transitions(trajs_lv,R_lv;verbose=false)
K_lv = feature_dimension(2)
rs_lv = Dict((1,0)=>(0,0),(-1,1)=>(1,1),(0,-1)=>(0,1))
lv_rate_fn(θd) = [θd[(1,0)][1], θd[(-1,1)][6], θd[(0,-1)][3]]
lv_true = [ALPHA, BETA, GAMMA]

lv_data = []
for σ in NOISE_LEVELS
    @printf("  LV σ=%.1f...\n",σ); flush(stdout)
    θ0  = empirical_initialization(td_lv,R_lv;verbose=false,noise_scale=σ,
                                    reactant_stoich=rs_lv)
    obj0 = objective(P_lv,R_lv,θ0,X_lv,1.0)
    res  = optimize_stls(P_lv,R_lv,X_lv,1.0,θ0;
                         τ=0.005,outer_iter=40,polish_iter=80,
                         verbose=false,verbose_history=true,
                         gradient_fn=gradient_adjoint_krylov!)
    push!(lv_data, full_history(res,flatten_params(θ0),obj0,lv_rate_fn,lv_true,R_lv,K_lv))
end

# ─── Brusselator ─────────────────────────────────────────────────────────────
println("Generating Brusselator data..."); flush(stdout)
struct XYProj; sol; end
Base.getproperty(p::XYProj,s::Symbol) =
    s==:sol ? getfield(p,:sol) : getproperty(getfield(p,:sol),s)
(p::XYProj)(t) = (Int(round(p.sol(t)[1])),Int(round(p.sol(t)[2])))
K1A=1.0;K2B=1.0;K3B=3.0;K4B=1.0
R_bru = [(1,0),(1,-1),(-1,1),(-1,0)]
rn_bru = @reaction_network begin
    @species X(t) Y(t) D(t) E(t); @parameters k1 k2 k3 k4 A B
    k1*A, 0-->X; k2, 2X+Y-->3X; k3*B, X-->Y+D; k4, X-->E
end
raw_bru = [solve(JumpProblem(JumpInputs(rn_bru,[:X=>1,:Y=>1,:D=>0,:E=>0],(0.0,50.0),
    [:k1=>1.0,:k2=>1.0,:k3=>1.0,:k4=>1.0,:A=>1.0,:B=>3.0])),SSAStepper()) for _ in 1:500]
trajs_bru = [XYProj(s) for s in raw_bru]
times_bru = collect(0.0:1.0:50.0)
X_bru = build_state_space(trajs_bru,times_bru,R_bru;expansion=2,verbose=false)
P_bru = build_distributions(trajs_bru,times_bru,X_bru)
td_bru = count_transitions(trajs_bru,R_bru;verbose=false)
K_bru = cubic_feature_dimension(2)
k2_base = let
    wexp = sum(et*cubic_features(x)[9] for (x,et) in td_bru.exposure)
    wexp > 0 ? td_bru.counts[(1,-1)] / wexp : 1.0
end
bru_rate_fn(θd) = [θd[(1,0)][1],θd[(1,-1)][9],θd[(-1,1)][2],θd[(-1,0)][2]]
bru_true = [K1A,K2B,K3B,K4B]

bru_data = []
for σ in NOISE_LEVELS
    @printf("  Bru σ=%.1f...\n",σ); flush(stdout)
    θ0 = empirical_initialization(td_bru,R_bru;verbose=false,noise_scale=σ,
                                   feature_map=cubic_features)
    wexp = sum(et*cubic_features(x)[9] for (x,et) in td_bru.exposure)
    if wexp > 0
        θ0[(1,-1)] = zeros(K_bru)
        θ0[(1,-1)][9] = k2_base*max(0.1,abs(1.0+σ*randn()))
    end
    obj0 = objective(P_bru,R_bru,θ0,X_bru,1.0;feature_map=cubic_features)
    res  = optimize_stls(P_bru,R_bru,X_bru,1.0,θ0;
                         τ=0.02,outer_iter=40,polish_iter=80,
                         verbose=false,verbose_history=true,
                         feature_map=cubic_features,
                         gradient_fn=gradient_adjoint_krylov!)
    push!(bru_data, full_history(res,flatten_params(θ0),obj0,bru_rate_fn,bru_true,R_bru,K_bru))
end

# ─── Schlögl ─────────────────────────────────────────────────────────────────
println("Generating Schlögl data..."); flush(stdout)
K1_SCH=0.017; K2_SCH=0.0006; K3_SCH=1.7; K4_SCH=0.2
R_sch  = [(1,),(-1,)]
rn_sch = @reaction_network begin
    k1, 2X-->3X; k2, 3X-->2X; k3, 0-->X; k4, X-->0
end
trajs_sch = [begin
    dprob = DiscreteProblem(rn_sch,[:X=>25],(0.0,50.0),
                            [:k1=>K1_SCH,:k2=>K2_SCH,:k3=>K3_SCH,:k4=>K4_SCH])
    solve(JumpProblem(rn_sch,dprob,Direct()),SSAStepper())
end for _ in 1:1000]
times_sch = collect(0.0:0.5:50.0)
X_sch = build_state_space(trajs_sch,times_sch,R_sch;expansion=3,verbose=false)
P_sch = build_distributions(trajs_sch,times_sch,X_sch)
K_sch = cubic_feature_dimension(1)

# Pre-compute WLS base estimates (noise-free), then scale with σ per noise level
let
    bc=Dict{NTuple{1,Int},Int}(); dc=Dict{NTuple{1,Int},Int}()
    em=Dict{NTuple{1,Int},Float64}()
    for traj in trajs_sch
        ts=traj.t; us=traj.u
        for i in 1:length(ts)-1
            xs=(Int(us[i][1]),); dt_s=ts[i+1]-ts[i]; Δ=Int(us[i+1][1])-Int(us[i][1])
            em[xs]=get(em,xs,0.0)+dt_s
            if Δ==1; bc[xs]=get(bc,xs,0)+1; elseif Δ==-1; dc[xs]=get(dc,xs,0)+1; end
        end
    end
    xs_v=sort([x[1] for x in keys(em)])
    Ab=[]; bb=[]; wb=[]; Ad=[]; bd=[]; wd=[]
    for xv in xs_v
        x=(xv,); e=get(em,x,0.0); e<1.0 && continue
        φ=cubic_features(x)
        push!(Ab,φ[1],φ[3]); push!(bb,get(bc,x,0)/e); push!(wb,sqrt(e))
        push!(Ad,φ[2],φ[4]); push!(bd,get(dc,x,0)/e); push!(wd,sqrt(e))
    end
    np=length(bb)
    k3k1=max.(reshape(Ab,2,np)'.* wb \ (bb.*wb), 1e-8)
    k4k2=max.(reshape(Ad,2,np)'.* wd \ (bd.*wd), 1e-8)
    global sch_k3_base=k3k1[1]; global sch_k1_base=k3k1[2]
    global sch_k4_base=k4k2[1]; global sch_k2_base=k4k2[2]
end

sch_rate_fn(θd) = [θd[(1,)][3], θd[(-1,)][4], θd[(1,)][1], θd[(-1,)][2]]
sch_true = [K1_SCH, K2_SCH, K3_SCH, K4_SCH]

sch_data = []
for σ in NOISE_LEVELS
    @printf("  Sch σ=%.1f...\n",σ); flush(stdout)
    θ0 = Dict(ζ=>zeros(K_sch) for ζ in R_sch)
    θ0[(1,)][1]  = sch_k3_base * max(0.1, abs(1.0+σ*randn()))
    θ0[(1,)][3]  = sch_k1_base * max(0.1, abs(1.0+σ*randn()))
    θ0[(-1,)][2] = sch_k4_base * max(0.1, abs(1.0+σ*randn()))
    θ0[(-1,)][4] = sch_k2_base * max(0.1, abs(1.0+σ*randn()))
    obj0 = objective(P_sch,R_sch,θ0,X_sch,0.5;feature_map=cubic_features)
    res  = optimize_stls(P_sch,R_sch,X_sch,0.5,θ0;
                         τ=0.01, fisher_precond=true, per_reaction_threshold=true,
                         outer_iter=50,polish_iter=80,
                         verbose=false,verbose_history=true,
                         feature_map=cubic_features,
                         gradient_fn=gradient_adjoint_krylov!)
    push!(sch_data, full_history(res,flatten_params(θ0),obj0,sch_rate_fn,sch_true,R_sch,K_sch))
end

# ─── Plot ─────────────────────────────────────────────────────────────────────
println("Plotting..."); flush(stdout)
fig = Figure(size=(SW2, SH_HALF*2 + 30))

titles   = ["Michaelis-Menten","Lotka-Volterra","Brusselator","Schlögl"]
datasets = [mm_data, lv_data, bru_data, sch_data]

for (col,(title,data)) in enumerate(zip(titles,datasets))
    ax_top = Axis(fig[1,col], title=title,
                  ylabel = col==1 ? L"\mathcal{L}(\theta)" : "",
                  yscale=log10, yticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                  xticklabelsvisible=false)
    ax_bot = Axis(fig[2,col],
                  xlabel="Gradient step",
                  ylabel = col==1 ? "Relative rate error" : "",
                  yscale=log10, yticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0])

    for (i,(σ,d)) in enumerate(zip(NOISE_LEVELS,data))
        xs_o, objs, xs_e, errs, stls_end = d

        # Draw vertical line at STLS→polish boundary (only first noise level to avoid clutter)
        if i == 1
            vlines!(ax_top, [stls_end+0.5]; color="gray", linestyle=:dot, linewidth=0.8)
            vlines!(ax_bot, [stls_end+0.5]; color="gray", linestyle=:dot, linewidth=0.8)
        end

        lines!(ax_top, collect(xs_o), max.(objs,1e-16);
               linestyle=LINESTYLES[i], color=GRAYS[i], linewidth=1.5)
        lines!(ax_bot, collect(xs_e), max.(errs,1e-4);
               linestyle=LINESTYLES[i], color=GRAYS[i], linewidth=1.5,
               label=NOISE_LABELS[i])
    end

    if col == length(datasets)
        axislegend(ax_bot; position=:rt, labelsize=9, framevisible=true)
    end
end

Label(fig[1,1,TopLeft()], "Thresholding", fontsize=8, color="gray")

rowgap!(fig.layout, 8)
colgap!(fig.layout, 10)

out_path = joinpath(@__DIR__,"../paper/figures/noise_robustness.pdf")
save(out_path, fig)
println("Saved: $out_path")
