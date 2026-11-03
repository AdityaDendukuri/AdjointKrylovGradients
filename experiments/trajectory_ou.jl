"""
Trajectory-based SDE parameter identification via SemigroupLayer.

Demonstrates that the same SemigroupLayer used for density observations
works directly with individual sample trajectories via the negative
log-likelihood loss:

    L(θ) = -∑_{trajectories} ∑_{steps} log p(X_{t+1} | X_t, θ, Δt)

where  p(X_{t+1}=j | X_t=i, θ, Δt) = [e^{A(θ)Δt}]_{j,i}

The semigroup is applied to a unit vector e_i (point mass at X_t),
giving the full transition distribution from that state.  The upstream
gradient from log(ρ_pred[j]) is -e_j / ρ_pred[j], which flows
through the SemigroupLayer.rrule just like any other loss.

System: 1D Ornstein-Uhlenbeck
  dXt = -θ(c)·Xt dt + σ dWt,    θ(c) = 0.5 + 0.8c,   σ = 0.4
Grid:  x ∈ [-3, 3],  n = 120 points
Data:  N_TRAJ trajectories per context, K steps at Δt = 0.25

Compare: density-MSE loss vs trajectory log-likelihood loss.
"""

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Flux, Zygote
using ExponentialUtilities: expv
using StochasticDiffEq

include("../src/shared_krylov.jl")
include("../src/fpe.jl")

# ── Grid & library ─────────────────────────────────────────────────────────────
const N_GRID  = 120
const XS      = collect(range(-3.0, 3.0; length=N_GRID))
const H_GRID  = XS[2] - XS[1]
const D_DIFF  = 0.08     # σ²/2
const A0_OU, DIRS_OU = fpe_library_1d(XS, [x -> x]; D=D_DIFF)  # drift = θ·x
const SL_OU   = SemigroupLayer(A0_OU, DIRS_OU; m_krylov=20)

# True map: θ(c) = 0.5 + 0.8c  (linearly context-dependent decay rate)
θ_true(c) = 0.5 + 0.8*c

# ── Trajectory generator from the FPE transition matrix ───────────────────────
# Sample directly from P[:,i] = exp(A·dt)·e_i (the true model distribution).
# This avoids continuous→grid discretisation error: the data is exactly
# consistent with our semigroup model, so the NLL surface is well-defined.
function build_transition_matrix(c, dt)
    θ = θ_true(c)
    A = A0_OU + θ * DIRS_OU[1]
    P = Matrix{Float64}(undef, N_GRID, N_GRID)
    for i in 1:N_GRID
        col = expv(dt, A, UNIT_VECS[i]; m=20)
        P[:, i] .= max.(real.(col), 0.0)
        P[:, i] ./= sum(P[:, i])   # renormalize to exact simplex
    end
    P
end

function generate_trajectories(c, n_traj, k_steps, dt; rng=Random.default_rng())
    P = build_transition_matrix(c, dt)
    trajs = Vector{Vector{Int}}(undef, n_traj)
    for t in 1:n_traj
        # Start near centre of grid (x ≈ 0)
        i = clamp(round(Int, N_GRID/2) + rand(rng, -5:5), 1, N_GRID)
        path = [i]
        for _ in 1:k_steps
            weights = P[:, path[end]]
            i = searchsortedfirst(cumsum(weights), rand(rng))
            push!(path, clamp(i, 1, N_GRID))
        end
        trajs[t] = path
    end
    trajs
end

# Convert index-trajectories to Dict{start_idx => [next_idx...]}
function traj_to_transition_counts(trajs)
    counts = Dict{Int, Vector{Int}}()
    for path in trajs
        for t in 1:length(path)-1
            push!(get!(counts, path[t], Int[]), path[t+1])
        end
    end
    counts
end

# ── Precomputed unit vectors (constants — no gradient needed for p0) ────────────
# Zygote differentiates through θ → SL(e_i, θ, dt) only; rrule returns
# ZeroTangent() for p0, so e_i being a pre-built dense array is fine.
const UNIT_VECS = [begin e = zeros(N_GRID); e[k] = 1.0; e end for k in 1:N_GRID]

# ── Losses ─────────────────────────────────────────────────────────────────────

# Trajectory NLL via SemigroupLayer — batched by starting state.
# One SemigroupLayer call per unique starting grid cell (not per trajectory).
# ρ_pred = e^{A(θ)dt} e_i gives the full transition distribution from cell i;
# we sum -log ρ_pred[j] over all observed next-states j from that cell.
function nll_loss(sl, transition_counts, θ_vec, dt)
    total = 0.0
    n_obs = 0
    for (i, js) in transition_counts
        ρ_pred = sl(UNIT_VECS[i], θ_vec, dt)   # ONE Krylov call per unique start
        for j in js
            total -= log(max(real(ρ_pred[j]), 1e-30))
        end
        n_obs += length(js)
    end
    total / n_obs
end

# Density MSE: observe histogram of X_t → propagate → compare to histogram of X_{t+1}
function density_mse_loss(sl, p_data, θ_vec, dts)
    total = 0.0
    for t in eachindex(dts)
        p_pred  = sl(p_data[t], θ_vec, dts[t])
        total  += 0.5 * sum(abs2, p_data[t+1] - p_pred)
    end
    total
end

# ── Dataset ────────────────────────────────────────────────────────────────────
const DT       = 0.25
const K_STEPS  = 4
const N_TRAJ   = 200
const TRAIN_CS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
const TEST_CS  = [0.15, 0.35, 0.55, 0.75, 0.95]

Random.seed!(42)
println("Generating trajectories…")
train_trajs = Dict(c => traj_to_transition_counts(generate_trajectories(c, N_TRAJ, K_STEPS, DT))
                   for c in TRAIN_CS)
test_trajs  = Dict(c => traj_to_transition_counts(generate_trajectories(c, N_TRAJ, K_STEPS, DT))
                   for c in TEST_CS)
n_unique_starts = mean(length(train_trajs[c]) for c in TRAIN_CS)
println("  Unique starting cells per context: ~$(round(Int, n_unique_starts)) (vs $(N_TRAJ*K_STEPS) raw pairs)")

# Also generate density data for comparison
function exact_density_traj(c)
    θ = θ_true(c); A = A0_OU + θ * DIRS_OU[1]
    p0 = exp.(-(XS .- 0.0).^2 ./ 0.5); p0 ./= sum(p0)
    ps = [p0]; for _ in 1:K_STEPS; push!(ps, expv(DT, A, ps[end])); end
    ps, fill(DT, K_STEPS)
end
train_density = Dict(c => exact_density_traj(c) for c in TRAIN_CS)
test_density  = Dict(c => exact_density_traj(c) for c in TEST_CS)

# ── Training ───────────────────────────────────────────────────────────────────
N_EPOCHS = 400
LR       = 3e-3

function make_enc()
    Chain(Dense(1=>16,tanh), Dense(16=>16,tanh),
          Dense(16=>1; bias=false)) |> f64
end

println("\nTraining — Trajectory NLL loss")
enc_nll = make_enc()
opt_nll = Flux.setup(Adam(LR), enc_nll)

println("Training — Density MSE loss (comparison)")
enc_mse = make_enc()
opt_mse = Flux.setup(Adam(LR), enc_mse)

@printf "\n%-6s  %-22s  %-22s\n" "epoch" "NLL  (train/test)" "MSE  (train/test)"
for epoch in 1:N_EPOCHS
    # NLL gradient step
    _, g_nll = Zygote.withgradient(enc_nll) do m
        sum(begin
                θ_v = softplus.(vec(m([c]))) .+ 0.1
                nll_loss(SL_OU, train_trajs[c], θ_v, DT)
            end for c in TRAIN_CS)
    end
    Flux.update!(opt_nll, enc_nll, g_nll[1])

    # MSE gradient step
    _, g_mse = Zygote.withgradient(enc_mse) do m
        sum(begin
                θ_v = softplus.(vec(m([c]))) .+ 0.1
                ps, dts = train_density[c]
                density_mse_loss(SL_OU, ps, θ_v, dts)
            end for c in TRAIN_CS)
    end
    Flux.update!(opt_mse, enc_mse, g_mse[1])

    if epoch % 50 == 0 || epoch == 1
        nll_tr = mean(nll_loss(SL_OU, train_trajs[c],
                               softplus.(vec(enc_nll([c]))) .+ 0.1, DT) for c in TRAIN_CS)
        nll_te = mean(nll_loss(SL_OU, test_trajs[c],
                               softplus.(vec(enc_nll([c]))) .+ 0.1, DT) for c in TEST_CS)
        mse_tr = mean(density_mse_loss(SL_OU, train_density[c][1],
                                        softplus.(vec(enc_mse([c]))) .+ 0.1,
                                        train_density[c][2]) for c in TRAIN_CS)
        mse_te = mean(density_mse_loss(SL_OU, test_density[c][1],
                                        softplus.(vec(enc_mse([c]))) .+ 0.1,
                                        test_density[c][2]) for c in TEST_CS)
        @printf "%-6d  NLL %7.4f / %7.4f     MSE %9.6f / %9.6f\n" epoch nll_tr nll_te mse_tr mse_te
        flush(stdout)
    end
end

# ── Evaluation ─────────────────────────────────────────────────────────────────
println("\n━━━ Parameter recovery at test contexts ━━━")
@printf "%-6s  %-8s  %-10s  %-10s  %-10s  %-10s\n" "c" "θ_true" "NLL_pred" "NLL_err%" "MSE_pred" "MSE_err%"
nll_errs = Float64[]; mse_errs = Float64[]
for c in vcat(TRAIN_CS, TEST_CS)
    θ_t   = θ_true(c)
    θ_nll = (softplus.(vec(enc_nll([c]))) .+ 0.1)[1]
    θ_mse = (softplus.(vec(enc_mse([c]))) .+ 0.1)[1]
    e_nll = abs(θ_nll - θ_t) / θ_t * 100
    e_mse = abs(θ_mse - θ_t) / θ_t * 100
    push!(nll_errs, e_nll); push!(mse_errs, e_mse)
    marker = c ∈ TEST_CS ? " ←test" : ""
    @printf "%-6.2f  %-8.3f  %-10.3f  %-10.1f  %-10.3f  %-10.1f%s\n" c θ_t θ_nll e_nll θ_mse e_mse marker
end
@printf "\nMean rel err — NLL: %.1f%%   MSE: %.1f%%\n" mean(nll_errs) mean(mse_errs)
println("\nSame SemigroupLayer, different loss — trajectory vs density.")

# ── Save results for figure generation ────────────────────────────────────────
using CSV, DataFrames
rows = NamedTuple[]
for c in vcat(TRAIN_CS, TEST_CS)
    θ_t   = θ_true(c)
    θ_nll = (softplus.(vec(enc_nll([c]))) .+ 0.1)[1]
    θ_mse = (softplus.(vec(enc_mse([c]))) .+ 0.1)[1]
    push!(rows, (
        c         = c,
        theta_true = θ_t,
        theta_nll  = θ_nll,
        theta_mse  = θ_mse,
        err_nll    = abs(θ_nll - θ_t) / θ_t * 100,
        err_mse    = abs(θ_mse - θ_t) / θ_t * 100,
        is_test    = c ∈ TEST_CS,
    ))
end
CSV.write("paper/figures/trajectory_ou_results.csv", DataFrame(rows))
println("Saved paper/figures/trajectory_ou_results.csv")
