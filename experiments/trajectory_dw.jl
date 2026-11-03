"""
Double-well SDE: trajectory NLL via SemigroupLayer vs Euler-Maruyama approximation.

Key comparison at coarse Δt=0.5:
  - EM-NLL approximates p(x_{t+1}|x_t) ≈ N(x_t + b(x_t)Δt, 2DΔt).
    For nonlinear drift, this Gaussian is wrong: near the unstable fixed
    point x=0 the true transition is bimodal (weight on both wells), while
    the EM Gaussian is unimodal and underestimates the spread.
    Result: biased θ estimates, especially for smaller barrier heights where
    the inter-well transition rate is high.
  - SemigroupLayer NLL computes p(x_{t+1}=j | x_t=i, θ, Δt) = [e^{A(θ)Δt}]_{ji}
    exactly (Krylov approximation), capturing the full non-Gaussian transition.
    Result: accurate θ at the same Δt.

System:  dXt = (-Xt³ + c·Xt) dt + √(2D) dWt
  c ∈ [0.3, 2.4]: barrier height.  c < 1 → single well; c > 1 → double well.
  True rate: θ(c) = c (one context-dependent direction).
  D = 0.5, grid x ∈ [-2.5, 2.5], n = 150.

Data:  Individual SDE trajectories from fine EM (dt_fine=1e-3), sub-sampled
       at Δt_obs.  Diverse starting positions across the grid.
       N_TRAJ = 300 trajectories per context, K_STEPS = 4 observation steps.
"""

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Flux, Zygote
using ExponentialUtilities: expv
using CSV, DataFrames

include("../src/shared_krylov.jl")
include("../src/fpe.jl")

# ── Grid and library ───────────────────────────────────────────────────────────
const N_DW   = 150
const XS_DW  = collect(range(-2.5, 2.5; length=N_DW))
const H_DW   = XS_DW[2] - XS_DW[1]
const D_DW   = 0.5

# A(c) = A₀[b=-x³] + c · D_lin[b=+x]
const A0_DW_BASE, DIRS_DW_BASE = fpe_library_1d(XS_DW, [x -> -x^3, x -> x]; D=D_DW)
const A0_DW  = A0_DW_BASE + 1.0 * DIRS_DW_BASE[1]   # fixed cubic part
const D_LIN  = DIRS_DW_BASE[2]                        # learnable: b = +θ·x
const SL_DW  = SemigroupLayer(A0_DW, [D_LIN]; m_krylov=30)

θ_true_dw(c) = Float64(c)

# ── Unit vectors (precomputed; Zygote returns ZeroTangent() for p0) ────────────
const UVECS_DW = [begin e = zeros(N_DW); e[k] = 1.0; e end for k in 1:N_DW]

# ── Data generation: individual SDE trajectories ──────────────────────────────
# Simulate the continuous SDE via fine EM, snap each observation to the grid.
# Starting positions spread across the grid so both wells are visited.
function gen_dw_trajectories(c; n_traj=300, k_steps=4, dt_obs=0.5,
                              dt_fine=1e-3, rng=Random.default_rng())
    n_sub = round(Int, dt_obs / dt_fine)
    σ_dw  = sqrt(2 * D_DW)
    b(x)  = -x^3 + c * x

    function snap(x)
        clamp(round(Int, (x - XS_DW[1]) / H_DW) + 1, 1, N_DW)
    end

    trajs = Vector{Vector{Int}}(undef, n_traj)
    for t in 1:n_traj
        # Start uniformly across the grid to cover both wells and the unstable point.
        x = XS_DW[1] + H_DW + rand(rng) * (XS_DW[end] - XS_DW[1] - 2H_DW)
        path = [snap(x)]
        for _ in 1:k_steps
            for _ in 1:n_sub
                x += b(x) * dt_fine + σ_dw * sqrt(dt_fine) * randn(rng)
                x  = clamp(x, XS_DW[1] + H_DW, XS_DW[end] - H_DW)
            end
            push!(path, snap(x))
        end
        trajs[t] = path
    end
    trajs
end

# Aggregate individual trajectories into start→[ends] transition counts.
function traj_to_counts(trajs)
    counts = Dict{Int, Vector{Int}}()
    for path in trajs
        for t in 1:length(path)-1
            push!(get!(counts, path[t], Int[]), path[t+1])
        end
    end
    counts
end

# ── Losses ─────────────────────────────────────────────────────────────────────

# SemigroupLayer NLL — one Krylov call per unique starting cell.
function sl_nll(counts, θ_vec, dt)
    total = 0.0; n = 0
    for (i, js) in counts
        ρ = SL_DW(UVECS_DW[i], θ_vec, dt)
        for j in js
            total -= log(max(real(ρ[j]), 1e-30))
        end
        n += length(js)
    end
    total / n
end

# Euler-Maruyama NLL — Gaussian approximation p(x_{t+1}|x_t) ≈ N(μ_EM, σ²_EM).
# At coarse Δt this is wrong for nonlinear drift: the true transition near x=0
# is bimodal (barrier crossing), but the EM Gaussian is unimodal.
function em_nll(counts, θ, dt)
    var_em = 2 * D_DW * dt   # EM variance (constant, independent of θ)
    total  = 0.0; n = 0
    for (i, js) in counts
        x_i  = XS_DW[i]
        μ_em = x_i + (-x_i^3 + θ[1] * x_i) * dt
        for j in js
            x_j    = XS_DW[j]
            total -= -0.5 * (x_j - μ_em)^2 / var_em - 0.5 * log(2π * var_em)
        end
        n += length(js)
    end
    total / n
end

# ── Dataset ────────────────────────────────────────────────────────────────────
const DT_COARSE = 0.5    # coarse Δt — EM fails here
const DT_FINE   = 0.1    # fine Δt — EM is approximately correct (reference)
const K_STEPS   = 4
const N_TRAJ    = 300

const TRAIN_CS_DW = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
const TEST_CS_DW  = [0.45, 0.75, 1.05, 1.35, 1.65, 1.95]

Random.seed!(42)
println("Generating double-well SDE trajectories (fine EM, dt_fine=1e-3)…")
println("  Δt_coarse=$(DT_COARSE), Δt_fine=$(DT_FINE), N_TRAJ=$(N_TRAJ), K=$(K_STEPS)")

train_counts_c = Dict(c => traj_to_counts(gen_dw_trajectories(c; n_traj=N_TRAJ, k_steps=K_STEPS, dt_obs=DT_COARSE)) for c in TRAIN_CS_DW)
test_counts_c  = Dict(c => traj_to_counts(gen_dw_trajectories(c; n_traj=N_TRAJ, k_steps=K_STEPS, dt_obs=DT_COARSE)) for c in TEST_CS_DW)
train_counts_f = Dict(c => traj_to_counts(gen_dw_trajectories(c; n_traj=N_TRAJ, k_steps=K_STEPS, dt_obs=DT_FINE))   for c in TRAIN_CS_DW)
test_counts_f  = Dict(c => traj_to_counts(gen_dw_trajectories(c; n_traj=N_TRAJ, k_steps=K_STEPS, dt_obs=DT_FINE))   for c in TEST_CS_DW)

n_pairs_c = mean(sum(length(v) for v in values(train_counts_c[c])) for c in TRAIN_CS_DW)
n_starts_c= mean(length(train_counts_c[c]) for c in TRAIN_CS_DW)
@printf "  Coarse: ~%.0f obs pairs, ~%.0f unique starts per context\n" n_pairs_c n_starts_c

# ── Encoder ────────────────────────────────────────────────────────────────────
make_enc_dw() = Chain(Dense(1=>16, tanh), Dense(16=>16, tanh),
                      Dense(16=>1; bias=false)) |> f64
enc_θ(m, c) = softplus(only(vec(m([Float64(c)])))) * 3.0   # bounded [0, 3]

# ── Training ───────────────────────────────────────────────────────────────────
N_EPOCHS_DW = 600
LR_DW       = 3e-3

function train!(enc, opt, counts_train, dt, loss_fn, label)
    println("\nTraining — $label (Δt=$dt)")
    @printf "%-6s  %-14s\n" "epoch" "train NLL"
    for epoch in 1:N_EPOCHS_DW
        _, g = Zygote.withgradient(enc) do m
            sum(loss_fn(counts_train[c], [enc_θ(m, c)], dt) for c in TRAIN_CS_DW)
        end
        Flux.update!(opt, enc, g[1])
        if epoch % 100 == 0 || epoch == 1
            tr = mean(loss_fn(counts_train[c], [enc_θ(enc, c)], dt) for c in TRAIN_CS_DW)
            @printf "%-6d  %-14.4f\n" epoch tr
            flush(stdout)
        end
    end
end

# Coarse Δt: SemigroupLayer NLL
enc_sl_c = make_enc_dw(); opt_sl_c = Flux.setup(Adam(LR_DW), enc_sl_c)
train!(enc_sl_c, opt_sl_c, train_counts_c, DT_COARSE, sl_nll, "SemigroupLayer NLL, Δt_coarse")

# Coarse Δt: EM NLL
enc_em_c = make_enc_dw(); opt_em_c = Flux.setup(Adam(LR_DW), enc_em_c)
train!(enc_em_c, opt_em_c, train_counts_c, DT_COARSE, em_nll, "EM NLL, Δt_coarse")

# Fine Δt: SemigroupLayer NLL (reference — both methods should work here)
enc_sl_f = make_enc_dw(); opt_sl_f = Flux.setup(Adam(LR_DW), enc_sl_f)
train!(enc_sl_f, opt_sl_f, train_counts_f, DT_FINE, sl_nll, "SemigroupLayer NLL, Δt_fine")

# Fine Δt: EM NLL (reference)
enc_em_f = make_enc_dw(); opt_em_f = Flux.setup(Adam(LR_DW), enc_em_f)
train!(enc_em_f, opt_em_f, train_counts_f, DT_FINE, em_nll, "EM NLL, Δt_fine")

# ── Evaluation ─────────────────────────────────────────────────────────────────
println("\n━━━ Parameter recovery (coarse Δt=$(DT_COARSE)) ━━━")
@printf "%-6s  %-8s  %-10s  %-10s  %-10s  %-10s\n" "c" "θ_true" "SL_pred" "SL_err%" "EM_pred" "EM_err%"

rows = NamedTuple[]
for dt_tag in (:coarse, :fine), c in sort(vcat(TRAIN_CS_DW, TEST_CS_DW))
    dt      = dt_tag == :coarse ? DT_COARSE : DT_FINE
    enc_sl  = dt_tag == :coarse ? enc_sl_c  : enc_sl_f
    enc_em  = dt_tag == :coarse ? enc_em_c  : enc_em_f
    θ_t     = θ_true_dw(c)
    θ_sl    = enc_θ(enc_sl, c)
    θ_em    = enc_θ(enc_em, c)
    e_sl    = abs(θ_sl - θ_t) / θ_t * 100
    e_em    = abs(θ_em - θ_t) / θ_t * 100
    push!(rows, (c=c, dt=dt, theta_true=θ_t,
                 theta_sl=θ_sl, theta_em=θ_em,
                 err_sl=e_sl, err_em=e_em,
                 is_test=(c ∈ TEST_CS_DW)))
    if dt_tag == :coarse
        marker = c ∈ TEST_CS_DW ? " ←test" : ""
        @printf "%-6.2f  %-8.3f  %-10.3f  %-10.1f  %-10.3f  %-10.1f%s\n" c θ_t θ_sl e_sl θ_em e_em marker
    end
end

# Summary
sl_c_errs = [r.err_sl for r in rows if r.dt == DT_COARSE]
em_c_errs = [r.err_em for r in rows if r.dt == DT_COARSE]
sl_f_errs = [r.err_sl for r in rows if r.dt == DT_FINE]
em_f_errs = [r.err_em for r in rows if r.dt == DT_FINE]

println("\n━━━ Mean relative errors ━━━")
@printf "  Δt=%.1f  SemigroupLayer: %.1f%%   EM: %.1f%%\n" DT_COARSE mean(sl_c_errs) mean(em_c_errs)
@printf "  Δt=%.1f  SemigroupLayer: %.1f%%   EM: %.1f%%\n" DT_FINE   mean(sl_f_errs) mean(em_f_errs)

CSV.write("paper/figures/trajectory_dw_results.csv", DataFrame(rows))
println("\nSaved paper/figures/trajectory_dw_results.csv")
