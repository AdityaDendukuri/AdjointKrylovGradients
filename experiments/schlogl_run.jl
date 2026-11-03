# =============================================================================
# Schlögl — comprehensive inference run
#
# Bistable 1-species system: 2X⇌3X, ∅⇌X
# Saves results/schlogl_results.jld2
#
# Usage:  julia --project=experiments experiments/schlogl_run.jl
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

const K1_TRUE = 0.017; const K2_TRUE = 0.0006
const K3_TRUE = 1.7;   const K4_TRUE = 0.2
const R_SCH = [(1,), (-1,)]
const R_LABELS = Dict(
    (1,)  => "2X→3X (k1) / ∅→X (k3)",
    (-1,) => "3X→2X (k2) / X→∅ (k4)",
)
# cubic_features(1) = [1, x, x(x-1)/2, x(x-1)(x-2)/6]
# birth (+1): index 1=k3, index 3=k1
# death (-1): index 2=k4, index 4=k2
const TRUE_PARAMS = Dict(
    (1,)  => Dict(1=>K3_TRUE, 3=>K1_TRUE),
    (-1,) => Dict(2=>K4_TRUE, 4=>K2_TRUE),
)

const N_TRAJ = 1000
const T_END  = 50.0
const DT     = 0.5

println("="^60); println("Schlögl comprehensive inference run")
println("="^60)

# ── Data generation ──────────────────────────────────────────────────────────
rn = @reaction_network begin
    k1, 2X --> 3X; k2, 3X --> 2X; k3, 0 --> X; k4, X --> 0
end
u0 = [:X => 25]
ps = [:k1=>K1_TRUE, :k2=>K2_TRUE, :k3=>K3_TRUE, :k4=>K4_TRUE]

print("Generating $N_TRAJ SSA trajectories... "); flush(stdout)
t_traj = @elapsed trajs = [begin
    dprob = DiscreteProblem(rn, u0, (0.0, T_END), ps)
    solve(JumpProblem(rn, dprob, Direct()), SSAStepper())
end for _ in 1:N_TRAJ]
@printf("done (%.1f s)\n", t_traj)

times   = collect(0.0:DT:T_END)
species = [:X]

snapshots = Dict{Float64,Matrix{Int}}()
for t in times
    snapshots[t] = hcat([Int.(round.(s(t))) for s in trajs]...)' |> Matrix{Int}
end

# Mode fractions
xs_final = [Int(round(trajs[i](T_END)[1])) for i in 1:N_TRAJ]
frac_lower = mean(xs_final .< 30)
@printf("Mode fractions at T=%.0f: lower=%.1f%% upper=%.1f%%\n",
    T_END, frac_lower*100, (1-frac_lower)*100)

# ── State space + distributions ──────────────────────────────────────────────
print("Building state space... "); flush(stdout)
t_ss = @elapsed begin
    X  = build_state_space(trajs, times, R_SCH; expansion=3, verbose=false)
    P  = build_distributions(trajs, times, X)
end
x_vals = [s[1] for s in X]
@printf("done  %d states, x∈[%d,%d], %d time points (%.1f s)\n",
    length(X), minimum(x_vals), maximum(x_vals), length(times), t_ss)

# ── True parameter vector ────────────────────────────────────────────────────
K_cub = cubic_feature_dimension(1)
θ_true = Dict(ζ => zeros(K_cub) for ζ in R_SCH)
θ_true[(1,)][1] = K3_TRUE; θ_true[(1,)][3]  = K1_TRUE
θ_true[(-1,)][2] = K4_TRUE; θ_true[(-1,)][4] = K2_TRUE
obj_ref = objective(P, R_SCH, θ_true, X, DT; feature_map=cubic_features)
@printf("Reference objective at true θ: %.4e\n", obj_ref)

# ── Initialisation (per-state WLS regression) ────────────────────────────────
print("Building empirical init (WLS regression)... "); flush(stdout)
birth_counts = Dict{NTuple{1,Int}, Int}()
death_counts = Dict{NTuple{1,Int}, Int}()
exposure_map = Dict{NTuple{1,Int}, Float64}()

for traj in trajs
    ts = traj.t; us = traj.u
    for i in 1:length(ts)-1
        x_state = (Int(us[i][1]),)
        Δ = Int(us[i+1][1]) - Int(us[i][1])
        dt = ts[i+1] - ts[i]
        exposure_map[x_state] = get(exposure_map, x_state, 0.0) + dt
        if Δ == 1
            birth_counts[x_state] = get(birth_counts, x_state, 0) + 1
        elseif Δ == -1
            death_counts[x_state] = get(death_counts, x_state, 0) + 1
        end
    end
end

xs_visited = sort([x[1] for x in keys(exposure_map)])
A_birth = Float64[]; b_birth = Float64[]; w_birth = Float64[]
A_death = Float64[]; b_death = Float64[]; w_death = Float64[]
for xv in xs_visited
    x = (xv,); exp_t = get(exposure_map, x, 0.0)
    exp_t < 1.0 && continue
    φ = cubic_features(x)
    push!(A_birth, φ[1], φ[3]); push!(b_birth, get(birth_counts, x, 0) / exp_t)
    push!(w_birth, sqrt(exp_t))
    push!(A_death, φ[2], φ[4]); push!(b_death, get(death_counts, x, 0) / exp_t)
    push!(w_death, sqrt(exp_t))
end

n_pts = length(b_birth)
A_b = reshape(A_birth, 2, n_pts)' .* w_birth
A_d = reshape(A_death, 2, n_pts)' .* w_death
k3k1 = max.(A_b \ (b_birth .* w_birth), 1e-8)
k4k2 = max.(A_d \ (b_death .* w_death), 1e-8)

# Lognormal perturbation, matching empirical_initialization. The previous form,
# r * max(0.1, abs(1 + 3*randn())), never deleted a direction because of the abs
# and the relative floor, so Schlogl was not affected by the deletion defect that
# hit the other three benchmarks; it is changed here only so all four share one
# perturbation model.
jitter(r) = r * exp(INIT_NOISE_SCALE * randn())

θ0 = Dict(ζ => zeros(K_cub) for ζ in R_SCH)
θ0[(1,)][1]  = jitter(k3k1[1]); θ0[(1,)][3]  = jitter(k3k1[2])
θ0[(-1,)][2] = jitter(k4k2[1]); θ0[(-1,)][4] = jitter(k4k2[2])
obj0 = objective(P, R_SCH, θ0, X, DT; feature_map=cubic_features)
@printf("done  obj_init=%.4e\n", obj0)

# ── Gradient timing ──────────────────────────────────────────────────────────
dA = build_derivatives(X, R_SCH; feature_map=cubic_features, K=K_cub)
θ0_flat = flatten_params(θ0)
g_buf   = zeros(length(θ0_flat))

print("Timing adjoint gradient (3 calls)... "); flush(stdout)
t_adj = @elapsed for _ in 1:3
    gradient_adjoint_krylov!(g_buf, θ0_flat, P, R_SCH, X, DT, dA, 0.0;
                             feature_map=cubic_features, K=K_cub)
end
t_adj_mean = t_adj / 3
@printf("%.3f s/call\n", t_adj_mean)

K_total = length(θ0_flat)
t_1obj  = @elapsed objective(P, R_SCH, θ0, X, DT; feature_map=cubic_features)
t_fd_estimate = (K_total + 1) * t_1obj

# ── STLS optimisation ────────────────────────────────────────────────────────
println("\nRunning STLS optimisation...")

t_opt = @elapsed result = optimize_stls(P, R_SCH, X, DT, θ0;
    threshold_mode=STLS_MODE, significance_level=STLS_SIGLEVEL,
    τ=0.01, outer_iter=50, polish_iter=150,
    verbose=true, verbose_history=true,
    feature_map=cubic_features,
    fisher_precond=true, per_reaction_threshold=true,
    gradient_fn=gradient_adjoint_krylov!)

@printf("Optimisation: %.1f s  (%d iters, converged=%s)\n",
    t_opt, result.n_iterations, string(result.converged))

trace_obj  = Float64.(result.history["objective"])
polish_obj = haskey(result.history, "polish_objectives") ?
             Float64.(result.history["polish_objectives"]) : Float64[]

# ── Fitted distributions ─────────────────────────────────────────────────────
print("Computing fitted CME distributions... "); flush(stdout)
A_opt    = build_generator(X, R_SCH, result.θ_opt; feature_map=cubic_features)
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
@printf("%-6s  %10s  %10s  %7s\n", "Rate", "True", "Inferred", "Err%")
println("-"^40)
for (name, ζ, idx, k_true) in [
        ("k1", (1,),  3, K1_TRUE), ("k3", (1,),  1, K3_TRUE),
        ("k2", (-1,), 4, K2_TRUE), ("k4", (-1,), 2, K4_TRUE)]
    k_inf = max(0.0, result.θ_opt[ζ][idx])
    err   = abs(k_inf - k_true) / k_true * 100
    @printf("%-6s  %10.5f  %10.6f  %7.1f\n", name, k_true, k_inf, err)
end

# ── Save to JLD2 ─────────────────────────────────────────────────────────────
outdir  = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
outfile = joinpath(outdir, "schlogl_results.jld2")

true_rates_flat = Dict(
    "k1" => K1_TRUE, "k2" => K2_TRUE,
    "k3" => K3_TRUE, "k4" => K4_TRUE,
)
true_feat_flat = Dict(
    "(1,)_k3" => 1, "(1,)_k1" => 3,
    "(-1,)_k4" => 2, "(-1,)_k2" => 4,
)

jldsave(outfile;
    system        = "schlogl",
    description   = "2X⇌3X, ∅⇌X — bistable 1D system",
    species       = ["X"],
    reactions     = [string(ζ) for ζ in R_SCH],
    reaction_labels = Dict(string(k) => v for (k,v) in R_LABELS),
    times         = times,
    N_traj        = N_TRAJ,
    snapshots     = snapshots,
    X             = X,
    P_empirical   = P,
    true_rates    = true_rates_flat,
    true_feat_idx = true_feat_flat,
    theta_init    = Dict(string(k) => copy(v) for (k,v) in θ0),
    theta_opt     = Dict(string(k) => copy(v) for (k,v) in result.θ_opt),
    obj_init      = obj0,
    obj_opt       = result.objective,
    obj_ref       = obj_ref,
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
    frac_lower_mode         = frac_lower,
)

println("\nSaved: $outfile")
println("  n_states=$(length(X))  t_grad_adj=$(round(t_adj_mean,digits=3))s  t_opt=$(round(t_opt,digits=1))s")
