# =============================================================================
# Lotka-Volterra — comprehensive inference run
#
# Reactions: ∅→X (α=1.0), X+Y→2Y (β=0.01), Y→∅ (γ=0.5)
# Saves results/lv_results.jld2
#
# Usage:  julia --project=experiments experiments/lv_run.jl
# =============================================================================

using Pkg; Pkg.activate(@__DIR__)
using AdjointKrylovGradients, Catalyst, JumpProcesses
using LinearAlgebra, Printf, Random, Statistics, JLD2
using ExponentialUtilities: expv_timestep

# Threshold rule: :relative reproduces the published configuration;
# :significance uses the Gauss-Newton t-statistic (see Section on
# approximation error). Defaults preserve existing behaviour.
const LV_NQ = parse(Int, get(ENV, "LV_NQ", "16"))
const LV_MK = parse(Int, get(ENV, "LV_MK", "25"))
const INIT_NOISE_MODEL = Symbol(get(ENV, "INIT_NOISE_MODEL", "lognormal"))
const INIT_NOISE_SCALE = parse(Float64, get(ENV, "INIT_NOISE_SCALE", "0.5"))
const RUN_SEED = parse(Int, get(ENV, "RUN_SEED", "42"))
const STLS_MODE = Symbol(get(ENV, "STLS_MODE", "relative"))
const STLS_SIGLEVEL = parse(Float64, get(ENV, "STLS_SIGLEVEL", "5.0"))

Random.seed!(RUN_SEED)

const ALPHA_TRUE = 1.0; const BETA_TRUE = 0.01; const GAMMA_TRUE = 0.5
const R_LV = [(1,0), (-1,1), (0,-1)]
const R_LABELS = Dict(
    (1,0)  => "∅→X (α)",
    (-1,1) => "X+Y→2Y (β)",
    (0,-1) => "Y→∅ (γ)",
)
const TRUE_RATES = Dict(
    (1,0)  => ALPHA_TRUE,
    (-1,1) => BETA_TRUE,
    (0,-1) => GAMMA_TRUE,
)
# Feature indices for quad_features(2): [1, X, Y, X²/2, Y²/2, XY]
const TRUE_FEAT = Dict(
    (1,0)  => 1,   # constant
    (-1,1) => 6,   # XY
    (0,-1) => 3,   # Y
)

const N_TRAJ = 500
const T_END  = 50.0
const DT     = 1.0
const NOISE  = 5.0

println("="^60); println("Lotka-Volterra comprehensive inference run")
println("="^60)

# ── Data generation ──────────────────────────────────────────────────────────
rn = @reaction_network begin
    alpha, 0 --> X
    beta,  X + Y --> 2Y
    gamma, Y --> 0
end
u0 = [:X=>50, :Y=>20]
ps = [:alpha=>ALPHA_TRUE, :beta=>BETA_TRUE, :gamma=>GAMMA_TRUE]

print("Generating $N_TRAJ SSA trajectories... "); flush(stdout)
t_traj = @elapsed trajs = [begin
    dprob = DiscreteProblem(rn, u0, (0.0, T_END), ps)
    solve(JumpProblem(rn, dprob, Direct()), SSAStepper())
end for _ in 1:N_TRAJ]
@printf("done (%.1f s)\n", t_traj)

times   = collect(0.0:DT:T_END)
species = [:X, :Y]

snapshots = Dict{Float64,Matrix{Int}}()
for t in times
    snapshots[t] = hcat([Int.(round.(s(t))) for s in trajs]...)' |> Matrix{Int}
end

# ── State space + distributions ──────────────────────────────────────────────
print("Building state space... "); flush(stdout)
t_ss = @elapsed begin
    X  = build_state_space(trajs, times, R_LV; expansion=2, verbose=false)
    P  = build_distributions(trajs, times, X)
end
@printf("done  %d states, %d time points (%.1f s)\n", length(X), length(times), t_ss)

# ── Initialisation ───────────────────────────────────────────────────────────
td  = count_transitions(trajs, R_LV; verbose=false)
reactant_stoich = Dict((1,0)=>(0,0), (-1,1)=>(1,1), (0,-1)=>(0,1))
θ0  = empirical_initialization(td, R_LV; verbose=false, noise_scale=INIT_NOISE_SCALE, noise_model=INIT_NOISE_MODEL,
                                reactant_stoich=reactant_stoich)
obj0 = objective(P, R_LV, θ0, X, DT)
@printf("Initial objective: %.4e\n", obj0)

# ── Gradient timing ──────────────────────────────────────────────────────────
K_feat  = feature_dimension(length(first(X)))
dA      = build_derivatives(X, R_LV; feature_map=quad_features, K=K_feat)
θ0_flat = flatten_params(θ0)
g_buf   = zeros(length(θ0_flat))

print("Timing adjoint gradient (3 calls)... "); flush(stdout)
t_adj = @elapsed for _ in 1:3
    gradient_adjoint_krylov!(g_buf, θ0_flat, P, R_LV, X, DT, dA, 0.0)
end
t_adj_mean = t_adj / 3
@printf("%.3f s/call\n", t_adj_mean)

K_total    = length(θ0_flat)
t_1obj     = @elapsed objective(P, R_LV, θ0, X, DT)
t_fd_estimate = (K_total + 1) * t_1obj
@printf("FD estimate: ~%.3f s/call  (%d params)\n", t_fd_estimate, K_total)

# ── STLS optimisation ────────────────────────────────────────────────────────
println("\nRunning STLS optimisation...")

t_opt = @elapsed result = optimize_stls(P, R_LV, X, DT, θ0;
    threshold_mode=STLS_MODE, significance_level=STLS_SIGLEVEL,
    n_quadrature=LV_NQ, m_krylov=LV_MK,
    τ=0.005, outer_iter=40, polish_iter=100,
    verbose=true, verbose_history=true,
    gradient_fn=gradient_adjoint_krylov!)

@printf("Optimisation: %.1f s  (%d iters, converged=%s)\n",
    t_opt, result.n_iterations, string(result.converged))

trace_obj  = Float64.(result.history["objective"])
polish_obj = haskey(result.history, "polish_objectives") ?
             Float64.(result.history["polish_objectives"]) : Float64[]

# ── Fitted distributions ─────────────────────────────────────────────────────
print("Computing fitted CME distributions... "); flush(stdout)
A_opt = build_generator(X, R_LV, result.θ_opt)
A_opnorm = opnorm(A_opt, Inf)
P_fit = Vector{Vector{Float64}}(undef, length(times))
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
for ζ in R_LV
    k_true = TRUE_RATES[ζ]
    k_inf  = max(0.0, result.θ_opt[ζ][TRUE_FEAT[ζ]])
    err    = abs(k_inf - k_true) / k_true * 100
    @printf("%-16s  %8.4f  %10.4f  %7.1f\n", R_LABELS[ζ], k_true, k_inf, err)
end

# ── Save to JLD2 ─────────────────────────────────────────────────────────────
outdir  = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
outfile = joinpath(outdir, "lv_results.jld2")

jldsave(outfile;
    system        = "lotka_volterra",
    description   = "∅→X (α=1), X+Y→2Y (β=0.01), Y→∅ (γ=0.5)",
    species       = string.(species),
    reactions     = [string(ζ) for ζ in R_LV],
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
