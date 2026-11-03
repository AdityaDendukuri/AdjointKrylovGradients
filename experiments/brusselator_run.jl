# =============================================================================
# Brusselator — comprehensive inference run
#
# Reactions (X,Y projected): ∅→X (k1A=1), 2X+Y→3X (k2=1), X→Y (k3B=3), X→∅ (k4=1)
# Saves results/brusselator_results.jld2
#
# Usage:  julia --project=experiments experiments/brusselator_run.jl
# =============================================================================

using Pkg; Pkg.activate(@__DIR__)
using AdjointKrylovGradients, Catalyst, JumpProcesses
using LinearAlgebra, Printf, Random, Statistics, JLD2
using ExponentialUtilities: expv_timestep

# Threshold rule: :relative reproduces the published configuration;
# :significance uses the Gauss-Newton t-statistic (see Section on
# approximation error). Defaults preserve existing behaviour.
const INIT_NOISE_MODEL = Symbol(get(ENV, "INIT_NOISE_MODEL", "lognormal"))
const INIT_NOISE_SCALE = parse(Float64, get(ENV, "INIT_NOISE_SCALE", "0.5"))
const RUN_SEED = parse(Int, get(ENV, "RUN_SEED", "42"))
const STLS_MODE = Symbol(get(ENV, "STLS_MODE", "relative"))
const STLS_SIGLEVEL = parse(Float64, get(ENV, "STLS_SIGLEVEL", "5.0"))

Random.seed!(RUN_SEED)

const K1A_TRUE = 1.0; const K2_TRUE = 1.0
const K3B_TRUE = 3.0; const K4_TRUE = 1.0
const R_BRUS = [(1,0), (1,-1), (-1,1), (-1,0)]
const R_LABELS = Dict(
    (1,0)  => "∅→X (k1A)",
    (1,-1) => "2X+Y→3X (k2)",
    (-1,1) => "X→Y (k3B)",
    (-1,0) => "X→∅ (k4)",
)
const TRUE_RATES = Dict(
    (1,0)  => K1A_TRUE,
    (1,-1) => K2_TRUE,
    (-1,1) => K3B_TRUE,
    (-1,0) => K4_TRUE,
)
# cubic_features(2) = [1,X,Y,X²/2,Y²/2,XY,X³/6,Y³/6,X²Y/2,XY²/2]
const TRUE_FEAT = Dict(
    (1,0)  => 1,   # constant
    (1,-1) => 9,   # X²Y/2 (autocatalytic)
    (-1,1) => 2,   # X
    (-1,0) => 2,   # X
)

const N_TRAJ = 500
const T_END  = 50.0
const DT     = 1.0

println("="^60); println("Brusselator comprehensive inference run")
println("="^60)

# ── Data generation ──────────────────────────────────────────────────────────
rn = @reaction_network begin
    @species X(t) Y(t) D(t) E(t)
    @parameters k1 k2 k3 k4 A B
    k1*A, 0 --> X
    k2,   2X + Y --> 3X
    k3*B, X --> Y + D
    k4,   X --> E
end
u0 = [:X=>1, :Y=>1, :D=>0, :E=>0]
ps = [:k1=>1.0, :k2=>1.0, :k3=>1.0, :k4=>1.0, :A=>1.0, :B=>3.0]

print("Generating $N_TRAJ SSA trajectories... "); flush(stdout)
t_traj = @elapsed raw_trajs = [begin
    jinput = JumpInputs(rn, u0, (0.0, T_END), ps)
    solve(JumpProblem(jinput), SSAStepper())
end for _ in 1:N_TRAJ]
@printf("done (%.1f s)\n", t_traj)

# Project to (X,Y)
struct XYProj; sol; end
Base.getproperty(p::XYProj, s::Symbol) =
    s == :sol ? getfield(p, :sol) : getproperty(getfield(p, :sol), s)
(p::XYProj)(t) = (Int(round(p.sol(t)[1])), Int(round(p.sol(t)[2])))
trajs = [XYProj(s) for s in raw_trajs]

times   = collect(0.0:DT:T_END)
species = [:X, :Y]

snapshots = Dict{Float64,Matrix{Int}}()
for t in times
    mat = zeros(Int, N_TRAJ, 2)
    for i in 1:N_TRAJ
        xy = trajs[i](t); mat[i,1] = xy[1]; mat[i,2] = xy[2]
    end
    snapshots[t] = mat
end

# ── State space + distributions ──────────────────────────────────────────────
print("Building state space... "); flush(stdout)
t_ss = @elapsed begin
    X  = build_state_space(trajs, times, R_BRUS; expansion=2, verbose=false)
    P  = build_distributions(trajs, times, X)
end
@printf("done  %d states, %d time points (%.1f s)\n", length(X), length(times), t_ss)

# ── Initialisation ───────────────────────────────────────────────────────────
td = count_transitions(trajs, R_BRUS; verbose=false)
θ0 = empirical_initialization(td, R_BRUS; verbose=false, noise_scale=5.0,
                               feature_map=cubic_features)

# Fix autocatalytic init: (1,-1) needs X²Y/2 (index 9)
if haskey(θ0, (1,-1))
    n_trans = td.counts[(1,-1)]
    wexp = sum(exp_t * cubic_features(x)[9] for (x, exp_t) in td.exposure)
    if wexp > 0
        θ0[(1,-1)] = zeros(cubic_feature_dimension(2))
        θ0[(1,-1)][9] = (n_trans / wexp) * max(0.1, abs(1.0 + 5.0 * randn()))
    end
end

obj0 = objective(P, R_BRUS, θ0, X, DT; feature_map=cubic_features)
@printf("Initial objective: %.4e\n", obj0)

# ── Gradient timing ──────────────────────────────────────────────────────────
K_cub   = cubic_feature_dimension(2)
dA      = build_derivatives(X, R_BRUS; feature_map=cubic_features, K=K_cub)
θ0_flat = flatten_params(θ0)
g_buf   = zeros(length(θ0_flat))

print("Timing adjoint gradient (3 calls)... "); flush(stdout)
t_adj = @elapsed for _ in 1:3
    gradient_adjoint_krylov!(g_buf, θ0_flat, P, R_BRUS, X, DT, dA, 0.0;
                             feature_map=cubic_features, K=K_cub)
end
t_adj_mean = t_adj / 3
@printf("%.3f s/call\n", t_adj_mean)

K_total = length(θ0_flat)
t_1obj  = @elapsed objective(P, R_BRUS, θ0, X, DT; feature_map=cubic_features)
t_fd_estimate = (K_total + 1) * t_1obj

# ── STLS optimisation ────────────────────────────────────────────────────────
println("\nRunning STLS optimisation...")

t_opt = @elapsed result = optimize_stls(P, R_BRUS, X, DT, θ0;
    threshold_mode=STLS_MODE, significance_level=STLS_SIGLEVEL,
    τ=0.02, outer_iter=40, polish_iter=100,
    verbose=true, verbose_history=true,
    feature_map=cubic_features,
    gradient_fn=gradient_adjoint_krylov!)

@printf("Optimisation: %.1f s  (%d iters, converged=%s)\n",
    t_opt, result.n_iterations, string(result.converged))

trace_obj  = Float64.(result.history["objective"])
polish_obj = haskey(result.history, "polish_objectives") ?
             Float64.(result.history["polish_objectives"]) : Float64[]

# ── Fitted distributions ─────────────────────────────────────────────────────
print("Computing fitted CME distributions... "); flush(stdout)
A_opt    = build_generator(X, R_BRUS, result.θ_opt; feature_map=cubic_features)
A_opnorm = opnorm(A_opt, Inf)
P_fit    = Vector{Vector{Float64}}(undef, length(times))
P_fit[1] = copy(P[1])
for k in 1:length(times)-1
    P_fit[k+1] = expv_timestep(DT, A_opt, P_fit[k];
                               m=25, tol=1e-8, adaptive=true, opnorm=A_opnorm)
end
println("done")

# ── Parameter recovery ───────────────────────────────────────────────────────
println("\n", "="^60, "\nPARAMETER RECOVERY\n", "="^60)
@printf("%-16s  %8s  %10s  %7s\n", "Reaction", "True", "Inferred", "Err%")
println("-"^50)
for ζ in R_BRUS
    k_true = TRUE_RATES[ζ]
    k_inf  = max(0.0, result.θ_opt[ζ][TRUE_FEAT[ζ]])
    err    = abs(k_inf - k_true) / k_true * 100
    @printf("%-16s  %8.4f  %10.4f  %7.1f\n", R_LABELS[ζ], k_true, k_inf, err)
end

# ── Save to JLD2 ─────────────────────────────────────────────────────────────
outdir  = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
outfile = joinpath(outdir, "brusselator_results.jld2")

jldsave(outfile;
    system        = "brusselator",
    description   = "∅→X (1), 2X+Y→3X (1), X→Y (3), X→∅ (1) — oscillatory 2D",
    species       = string.(species),
    reactions     = [string(ζ) for ζ in R_BRUS],
    reaction_labels = Dict(string(k) => v for (k,v) in R_LABELS),
    times         = times,
    N_traj        = N_TRAJ,
    snapshots     = snapshots,
    X             = X,
    P_empirical   = P,
    true_rates    = Dict(string(k) => v for (k,v) in TRUE_RATES),
    true_feat_idx = Dict(string(k) => v for (k,v) in TRUE_FEAT),
    theta_init    = Dict(string(k) => copy(v) for (k,v) in θ0),
    theta_opt     = Dict(string(k) => copy(v) for (k,v) in result.θ_opt),
    obj_init      = obj0,
    obj_opt       = result.objective,
    converged     = result.converged,
    n_iters       = result.n_iterations,
    trace_obj     = trace_obj,
    polish_obj    = polish_obj,
    P_fit         = P_fit,
    n_states      = length(X),
    K_total       = K_total,
    t_per_gradient_adj_s    = t_adj_mean,
    t_per_gradient_fd_est_s = t_fd_estimate,
    t_optimization_s        = t_opt,
    t_state_space_s         = t_ss,
    t_trajectories_s        = t_traj,
)

println("\nSaved: $outfile")
println("  n_states=$(length(X))  t_grad_adj=$(round(t_adj_mean,digits=3))s  t_opt=$(round(t_opt,digits=1))s")
