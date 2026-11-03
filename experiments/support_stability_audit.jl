"""
Is STLS support selection stable under the gradient's own discretization?

On a toy birth-death problem, changing only the quadrature order flipped the
recovered support (4 vs 5 nonzeros) even though the two gradients agreed far
below the Krylov truncation error. If that reproduces on a real benchmark it is
a robustness caveat the paper has to state, and it would mean re-run numbers are
sensitive to settings that are supposed to be immaterial.

This audit fixes the data and the initial point, then sweeps only the solver
settings (Krylov depth and quadrature order) that should not change the answer.
Schlogl is the hardest of the four benchmarks for this question: its true rates
span 0.0006 to 1.7, so thresholding has to separate coefficients across nearly
four orders of magnitude, which is exactly what Fisher preconditioning exists to
handle.

  SUPPORT_STAB_TRAJ    SSA trajectories        (default 1000)
  SUPPORT_STAB_SEEDS   initializer seeds       (default 3)
  SUPPORT_STAB_OUT     output CSV
"""

using Pkg; Pkg.activate(@__DIR__)
using AdjointKrylovGradients, Catalyst, JumpProcesses
using LinearAlgebra, Printf, Random, Statistics
using CSV, DataFrames

const N_TRAJ = parse(Int, get(ENV, "SUPPORT_STAB_TRAJ", "1000"))
const NSEEDS = parse(Int, get(ENV, "SUPPORT_STAB_SEEDS", "3"))
# Defaults match experiments/schlogl_run.jl, the configuration behind the paper's
# Schlogl row. An earlier version of this audit used tau=0.05, five times more
# aggressive, which removed k3 and produced spurious failures.
const TAU    = parse(Float64, get(ENV, "SUPPORT_STAB_TAU", "0.01"))
const OUTER  = parse(Int, get(ENV, "SUPPORT_STAB_OUTER", "50"))
const POLISH = parse(Int, get(ENV, "SUPPORT_STAB_POLISH", "150"))
const PATIENCE = parse(Int, get(ENV, "SUPPORT_STAB_PATIENCE", "1"))
const TMODE    = Symbol(get(ENV, "SUPPORT_STAB_MODE", "relative"))
const SIGLVL   = parse(Float64, get(ENV, "SUPPORT_STAB_SIGLEVEL", "3.0"))
const OUT    = get(ENV, "SUPPORT_STAB_OUT",
                   joinpath(@__DIR__, "..", "benchmark_results", "support_stability.csv"))

const K1_TRUE = 0.017; const K2_TRUE = 0.0006
const K3_TRUE = 1.7;   const K4_TRUE = 0.2
const R_SCH   = [(1,), (-1,)]
const T_END   = 50.0
const DT      = 0.5

Random.seed!(42)
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

times = collect(0.0:DT:T_END)
X = build_state_space(trajs, times, R_SCH; expansion=3, verbose=false)
P = build_distributions(trajs, times, X)
@printf("state space: %d states, %d time points\n", length(X), length(times))
@printf("STLS settings: tau=%.3g outer=%d polish=%d patience=%d mode=%s siglevel=%.3g\n", TAU, OUTER, POLISH, PATIENCE, TMODE, SIGLVL)

K_cub  = cubic_feature_dimension(1)
θ_true = Dict(ζ => zeros(K_cub) for ζ in R_SCH)
θ_true[(1,)][1]  = K3_TRUE; θ_true[(1,)][3]  = K1_TRUE
θ_true[(-1,)][2] = K4_TRUE; θ_true[(-1,)][4] = K2_TRUE
θ_true_flat  = flatten_params(θ_true)
true_support = findall(!iszero, θ_true_flat)
@printf("true support: %s (of %d), rate range %.4g .. %.4g\n",
        string(true_support), length(θ_true_flat),
        minimum(θ_true_flat[true_support]), maximum(θ_true_flat[true_support]))

# Empirical initializer, jittered per seed. The initializer is the only thing
# that varies between seeds; the data is fixed.
function make_init(seed)
    rng = MersenneTwister(seed)
    θ0 = Dict(ζ => zeros(K_cub) for ζ in R_SCH)
    jitter(r) = r * max(0.1, abs(1.0 + 1.5 * randn(rng)))
    θ0[(1,)][1]  = jitter(K3_TRUE); θ0[(1,)][3]  = jitter(K1_TRUE)
    θ0[(-1,)][2] = jitter(K4_TRUE); θ0[(-1,)][4] = jitter(K2_TRUE)
    θ0
end

rows = NamedTuple[]
println("\n seed |  m  | n_q |          recovered support | rate err | matches truth")
println("------+-----+-----+----------------------------+----------+--------------")

for seed in 1:NSEEDS
    θ0 = make_init(seed)
    for m in (25, 40), nq in (5, 8, 16, 24)
        res = optimize_stls(P, R_SCH, X, DT, θ0;
                            τ=TAU, outer_iter=OUTER, polish_iter=POLISH,
                            verbose=false, feature_map=cubic_features,
                            fisher_precond=true, per_reaction_threshold=true,
                            m_krylov=m, n_quadrature=nq,
                            elimination_patience=PATIENCE,
                            threshold_mode=TMODE, significance_level=SIGLVL)
        v   = flatten_params(res.θ_opt)
        sup = findall(!iszero, v)
        err = norm(v[true_support] - θ_true_flat[true_support]) /
              norm(θ_true_flat[true_support])
        exact = Set(sup) == Set(true_support)
        push!(rows, (seed=seed, m_krylov=m, n_quad=nq,
                     support=join(sup, "-"), support_size=length(sup),
                     exact_support=exact, rate_relerr=err, objective=res.objective))
        @printf("%5d | %3d | %3d | %26s | %8.2e | %s\n",
                seed, m, nq, string(sup), err, exact)
    end
end

df = DataFrame(rows)
mkpath(dirname(OUT)); CSV.write(OUT, df)

println("\n" * "="^72)
println("STABILITY VERDICT")
println("="^72)
for seed in 1:NSEEDS
    sub = df[df.seed .== seed, :]
    uniq = unique(sub.support)
    @printf("seed %d: %d distinct support(s) across 8 solver settings%s\n",
            seed, length(uniq), length(uniq) == 1 ? "  [STABLE]" : "  [UNSTABLE]")
    length(uniq) > 1 && for u in uniq
        @printf("           %-24s from %s\n", u,
                join(["m=$(r.m_krylov),nq=$(r.n_quad)" for r in eachrow(sub[sub.support .== u, :])], "; "))
    end
end
@printf("\noverall: %d/%d runs recovered the exact true support\n",
        count(df.exact_support), nrow(df))
@printf("rate error spread within identical supports: %.2e .. %.2e\n",
        minimum(df.rate_relerr), maximum(df.rate_relerr))
println("\nSaved $(OUT)")
