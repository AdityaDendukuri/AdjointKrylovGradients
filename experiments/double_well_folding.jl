"""
Context-conditioned double-well FPE: closest-to-real-world SDE benchmark.

Physical analogue: single-molecule protein folding observed via FRET.
  - Reaction coordinate x ∈ [-2,2] (end-to-end distance, normalised)
  - Drift: b(x;c) = -x³ + c·x   (double-well potential V = x⁴/4 - c·x²/2)
  - Context c = denaturant concentration  (c↑ raises barrier → deeper double well)
  - Diffusion: D = 0.5 (thermal noise, fixed)

Why this is hard for standard methods at Δt = 0.5:
  - Euler-Maruyama approximation assumes p(x₁|x₀) ≈ N(x₀+b·Δt, 2D·Δt) → wrong
    because the nonlinear drift changes sign within a single step.
  - MSD analysis assumes pure diffusion → completely ignores drift.
  - Both fail when Δt is comparable to the relaxation time 1/c.

Our approach: learn A(c) = A_const + c·D_linear from density snapshots.
  Generator is linear in c with M=1 direction — one SemigroupLayer call per step.

Data: N_SIM particles simulated via fine Euler-Maruyama (dt_fine=1e-3),
      sub-sampled at Δt_obs to produce empirical histograms on the FPE grid.
      The histograms are the *data*; the continuous SDE is the ground truth.
"""

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Flux, Zygote
using ExponentialUtilities: expv

include("../src/shared_krylov.jl")
include("../src/fpe.jl")

# ── Grid & library ─────────────────────────────────────────────────────────────
const N_GRID   = 150
const XS       = collect(range(-2.5, 2.5; length=N_GRID))
const H_GRID   = XS[2] - XS[1]
const D_DIFF   = 0.5

# A(c) = A_const[b=-x³] + c * D_linear[b=+x]
# A_const captures the fixed cubic part; D_linear is the one context-dependent direction.
const A_CONST_DW, DIRS_DW = fpe_library_1d(XS, [x -> -x^3, x -> x]; D=D_DIFF)
# DIRS_DW[1] = generator for drift -x³  → goes into A_const (θ=1 always)
# DIRS_DW[2] = generator for drift +x   → θ = c (context-dependent)
const A0_DW   = A_CONST_DW + 1.0 * DIRS_DW[1]   # fixed cubic part
const D_LINEAR = DIRS_DW[2]                        # context-dependent direction
const SL_DW   = SemigroupLayer(A0_DW, [D_LINEAR]; m_krylov=30)

# True parameter: θ(c) = c  (barrier height = context)
θ_true(c) = Float64(c)

# ── Data generation from continuous SDE ───────────────────────────────────────
# Simulate the true nonlinear SDE and aggregate into density histograms.
# This is the "real-world" regime: data comes from a continuous process,
# not from our discretized model.
function sim_sde_histogram(c, n_sim, dt_obs; dt_fine=1e-3, rng=Random.default_rng())
    n_substeps = round(Int, dt_obs / dt_fine)
    σ = sqrt(2 * D_DIFF)

    # Start from a Gaussian centred at 0 — diverse initial condition,
    # avoids clustering near potential minima which biases histograms.
    σ_init = 0.8
    p_init = exp.(-(XS).^2 ./ (2σ_init^2))
    p_init ./= sum(p_init)

    # Simulate forward from the same Gaussian distribution
    xs_sim = zeros(n_sim)
    for k in 1:n_sim
        x = σ_init * randn(rng)
        x = clamp(x, XS[1]+H_GRID, XS[end]-H_GRID)
        for _ in 1:n_substeps
            x = x + (-x^3 + c*x)*dt_fine + σ*sqrt(dt_fine)*randn(rng)
            x = clamp(x, XS[1]+H_GRID, XS[end]-H_GRID)
        end
        xs_sim[k] = x
    end

    # Histogram
    p_next = zeros(N_GRID)
    for x in xs_sim
        i = clamp(round(Int, (x - XS[1]) / H_GRID) + 1, 1, N_GRID)
        p_next[i] += 1.0
    end
    p_next ./= sum(p_next)
    return p_init, p_next
end

# ── Euler-Maruyama approximation baseline ─────────────────────────────────────
# At large Δt, this approximates p(x₁|x₀) ≈ N(x₀ + b(x₀)Δt, 2D·Δt).
# We fit c by minimising the KL between this approximation and the data.
# For a library with M=1 direction, this becomes fitting θ from the
# expected drift E[ΔX]/Δt ≈ b(E[x_t]; θ).
function em_estimate(p_init, p_next, dt_obs)
    # Empirical mean displacement
    mean_x0   = dot(XS, p_init)
    mean_x1   = dot(XS, p_next)
    mean_dx   = mean_x1 - mean_x0
    # EM approximation: E[ΔX] ≈ b(mean_x0) Δt = (-mean_x0³ + θ·mean_x0) Δt
    # Solve: θ = (mean_dx/Δt + mean_x0³) / mean_x0
    abs(mean_x0) < 1e-3 && return NaN
    (mean_dx / dt_obs + mean_x0^3) / mean_x0
end

# ── Dataset ───────────────────────────────────────────────────────────────────
const DTS_OBS  = [0.1, 0.25, 0.5]   # smaller Δt → tighter FPE discretisation
const N_SIM    = 5000               # more particles → less histogram noise
const TRAIN_CS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
const TEST_CS  = [0.45, 0.75, 1.05, 1.35, 1.65, 1.95]

Random.seed!(42)
println("Simulating double-well SDE data (ground truth from continuous SDE)…")
println("  $(N_SIM) particles per context, Δt_obs ∈ $DTS_OBS, dt_fine=1e-3")

# For each context: list of (p_t, p_{t+1}, dt) density snapshot pairs
function gen_context_data(c)
    data = Tuple{Vector{Float64}, Vector{Float64}, Float64}[]
    for dt in DTS_OBS
        p0, p1 = sim_sde_histogram(c, N_SIM, dt)
        push!(data, (p0, p1, dt))
    end
    data
end

train_data = Dict(c => gen_context_data(c) for c in TRAIN_CS)
test_data  = Dict(c => gen_context_data(c) for c in TEST_CS)

for c in sort(TRAIN_CS)
    θ_em = mean(em_estimate(d[1], d[2], d[3]) for d in train_data[c])
    @printf "  c=%.2f  θ_true=%.2f  EM_estimate=%.2f (err=%.1f%%)\n" c θ_true(c) θ_em 100*abs(θ_em-θ_true(c))/θ_true(c)
end

# ── Model ──────────────────────────────────────────────────────────────────────
enc = Chain(Dense(1=>16, tanh), Dense(16=>16, tanh),
            Dense(16=>1; bias=false)) |> f64
enc_to_θ(m, c) = softplus(only(vec(m([Float64(c)])))) * 3.0  # bounded [0, 3]

opt = Flux.setup(Adam(3e-3), enc)

function total_loss(model, data_dict, ctxs)
    sum(begin
        θ_v = [enc_to_θ(model, c)]
        sum(0.5 * sum(abs2, d[2] - SL_DW(d[1], θ_v, d[3])) for d in data_dict[c])
    end for c in ctxs)
end

# ── Training ───────────────────────────────────────────────────────────────────
N_EPOCHS = 800
println("\nTraining encoder (SemigroupLayer, density MSE on SDE-generated histograms)…")
@printf "\n%-6s  %-16s  %-16s\n" "epoch" "train loss" "test loss"

for epoch in 1:N_EPOCHS
    _, g = Zygote.withgradient(enc) do m
        total_loss(m, train_data, TRAIN_CS)
    end
    Flux.update!(opt, enc, g[1])

    if epoch % 100 == 0 || epoch == 1
        tr = total_loss(enc, train_data, TRAIN_CS)
        te = total_loss(enc, test_data,  TEST_CS)
        @printf "%-6d  %-16.6f  %-16.6f\n" epoch tr te
        flush(stdout)
    end
end

# ── Evaluation ─────────────────────────────────────────────────────────────────
println("\n━━━ Parameter recovery ━━━")
@printf "%-6s  %-8s  %-12s  %-10s  %-12s  %-10s\n" "c" "θ_true" "SG_pred" "SG_err%" "EM_pred" "EM_err%"

sg_errs = Float64[]; em_errs = Float64[]
for c in sort(vcat(TRAIN_CS, TEST_CS))
    θ_t    = θ_true(c)
    θ_sg   = enc_to_θ(enc, c)
    data_c = c ∈ TEST_CS ? test_data[c] : train_data[c]
    em_vals = filter(isfinite, [em_estimate(d[1], d[2], d[3]) for d in data_c])
    θ_em   = isempty(em_vals) ? NaN : mean(em_vals)
    e_sg   = abs(θ_sg - θ_t) / θ_t * 100
    e_em   = isfinite(θ_em) ? abs(θ_em - θ_t) / θ_t * 100 : NaN
    push!(sg_errs, e_sg)
    isfinite(e_em) && push!(em_errs, e_em)
    marker = c ∈ TEST_CS ? " ←test" : ""
    @printf "%-6.2f  %-8.3f  %-12.3f  %-10.1f  %-12.3f  %-10.1f%s\n" c θ_t θ_sg e_sg θ_em e_em marker
end
@printf "\nMean rel err — SemigroupLayer: %.1f%%   EM baseline: %.1f%%\n" mean(sg_errs) mean(em_errs)
println("\nNote: data generated from continuous SDE (not our discretized model).")
println("SemigroupLayer learns from density snapshots at Δt ∈ $DTS_OBS.")
println("EM baseline uses moment-matching at each Δt — fails at large Δt/nonlinear drift.")

# ── Save results for figure generation ────────────────────────────────────────
using CSV, DataFrames
rows = NamedTuple[]
for c in sort(vcat(TRAIN_CS, TEST_CS))
    θ_t  = θ_true(c)
    θ_sg = enc_to_θ(enc, c)
    data_c = c ∈ TEST_CS ? test_data[c] : train_data[c]
    em_vals = filter(isfinite, [em_estimate(d[1], d[2], d[3]) for d in data_c])
    θ_em = isempty(em_vals) ? NaN : mean(em_vals)
    push!(rows, (
        c       = c,
        theta_true = θ_t,
        theta_sg   = θ_sg,
        theta_em   = θ_em,
        err_sg     = abs(θ_sg - θ_t) / θ_t * 100,
        err_em     = isfinite(θ_em) ? abs(θ_em - θ_t) / θ_t * 100 : NaN,
        is_test    = c ∈ TEST_CS,
    ))
end
CSV.write("paper/figures/double_well_results.csv", DataFrame(rows))
println("\nSaved paper/figures/double_well_results.csv")
