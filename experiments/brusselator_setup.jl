# Shared setup for Brusselator CME sparse identification experiments.
# Safe to include multiple times — guards against re-running.
isdefined(Main, :DIRS_BS) && return

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Flux, Zygote, ChainRulesCore
using ExponentialUtilities: expv
using Catalyst, JumpProcesses

include("../src/shared_krylov.jl")

# ── System size Ω ─────────────────────────────────────────────────────────────
# Standard Brusselator with volume Ω:
#   k1A·Ω  :  ∅ → X    (production scales with Ω)
#   k4     :  X → ∅    (degradation, linear)
#   k3B    :  X → Y    (conversion, linear)
#   k2/Ω²  :  2X+Y→3X  (autocatalysis, bimolecular propensity C(x,2)·y)
# Fixed point:  X_ss = A·Ω,  Y_ss = B/A
# Larger Ω → larger X_ss → C(x,2) active → autocatalysis identifiable.
const OMEGA_BS = parse(Int, get(ENV, "BRUSS_OMEGA", "1"))

# ── State space ────────────────────────────────────────────────────────────────
const X_MAX_BS = 40
const Y_MAX_BS = 40
const N_BS     = (X_MAX_BS + 1) * (Y_MAX_BS + 1)

lin_bs(x, y) = x * (Y_MAX_BS + 1) + y + 1

# Initial condition: x=3 (C(x,2)=3 activates autocatalysis), y=y_init.
# No single y works for all channels:
#   y=0 → kills C(y,1), C(x,2)·y etc. — discriminates Channel 1 but blinds Channel 4
#   y=2 → activates autocatalysis (C(x,2)·y=6) — needed for Channel 4
# Solution: train on BOTH y=0 and y=2 trajectories per context (see discovery script).
function p0_bs(A_ctx=1.0, B_ctx=3.0; y_init=2)
    p  = zeros(N_BS)
    x0 = clamp(max(3, round(Int, A_ctx * OMEGA_BS)), 0, X_MAX_BS)
    p[lin_bs(x0, y_init)] = 1.0
    p
end

# ── Mass-action polynomial library ────────────────────────────────────────────
const CHANNELS_BS       = [(1, 0), (-1, 0), (-1, 1), (1, -1)]
const MONOMIALS_BS      = [(a, d - a) for d in 0:3 for a in 0:d]   # 10 entries
const MONOMIAL_DEGREES_BS = Float64[a + b for (a, b) in MONOMIALS_BS]  # [0,1,1,2,2,2,3,3,3,3]
const K_BS         = length(MONOMIALS_BS)

function comb_bs(x::Int, a::Int)
    a == 0 && return 1.0
    x < a  && return 0.0
    r = 1.0; for k in 0:(a-1); r *= (x-k); end; r / factorial(a)
end

function build_cme_library(X_max, Y_max)
    n = (X_max+1)*(Y_max+1); lin = (x,y) -> x*(Y_max+1)+y+1
    dirs = SparseMatrixCSC{Float64,Int}[]; labels = String[]
    for (dx,dy) in CHANNELS_BS, (a,b) in MONOMIALS_BS
        rs=Int[]; cs=Int[]; vs=Float64[]
        for x in 0:X_max, y in 0:Y_max
            (x<a||y<b) && continue
            xn=x+dx; yn=y+dy
            (xn<0||xn>X_max||yn<0||yn>Y_max) && continue
            prop=comb_bs(x,a)*comb_bs(y,b); iszero(prop) && continue
            ci=lin(x,y); ni=lin(xn,yn)
            push!(rs,ni); push!(cs,ci); push!(vs,prop)
            push!(rs,ci); push!(cs,ci); push!(vs,-prop)
        end
        isempty(rs) && continue
        push!(dirs, sparse(rs,cs,vs,n,n))
        push!(labels, "($(dx≥0 ? "+" : "")$dx,$(dy≥0 ? "+" : "")$dy)·C(x,$a)·C(y,$b)")
    end
    dirs, labels
end

println("Building CME mass-action library...")
const DIRS_BS, LABELS_BS = build_cme_library(X_MAX_BS, Y_MAX_BS)
const M_BS = length(DIRS_BS)
@printf "  %d directions, n=%d  (Ω=%d)\n" M_BS N_BS OMEGA_BS

# ── True model ─────────────────────────────────────────────────────────────────
# Indices: channel k, monomial m → direction (k-1)*10 + m
#   (+1,0)×1        → dir 1    ∅→X,     rate k1A·Ω
#   (-1,0)×x        → dir 13   X→∅,     rate k4=1
#   (-1,+1)×x       → dir 23   X→Y,     rate k3B=B
#   (+1,-1)×C(x,2)y → dir 39   2X+Y→3X, rate k2/Ω²
const TRUE_SUPPORT_BS = [1, 13, 23, 39]
# THETA_MAX covers max rate: k1A·Ω ≤ A_max·Ω, k3B ≤ B_max — use 2·Ω·A_max+margin
const THETA_MAX_BS = Float64(max(5, 2 * OMEGA_BS))

function true_theta_bs(c)
    A_ctx, B_ctx = c
    θ = zeros(M_BS)
    θ[1]  = Float64(A_ctx) * OMEGA_BS   # k1A·Ω  (production)
    θ[13] = 1.0                           # k4=1   (degradation)
    θ[23] = Float64(B_ctx)               # k3B=B  (conversion)
    θ[39] = 1.0 / OMEGA_BS^2             # k2/Ω²  (autocatalysis)
    θ
end

ctx_vec_bs(c) = Float64[c[1], c[2]]

let c=(1.0,3.0)
    A_check  = sum(true_theta_bs(c)[j]*DIRS_BS[j] for j in 1:M_BS)
    entry    = A_check[lin_bs(3,0), lin_bs(2,1)]
    expected = 1.0 / OMEGA_BS^2   # θ[39]·C(2,2)·C(1,1) = (k2/Ω²)·1·1
    @printf "\nGenerator sanity check (c=(1,3)): col-sum err=%.2e\n" maximum(abs.(vec(sum(A_check;dims=1))))
    @printf "  A[(3,0),(2,1)] via autocatalysis = %.4f (expect %.4f)\n" entry expected
    @assert abs(entry - expected) < 1e-10 "Index mismatch in TRUE_SUPPORT_BS"
end

true_support_bs(c)  = Set(j for j in 1:M_BS if abs(true_theta_bs(c)[j]) > 1e-8)
union_support_bs(cs)= sort!(collect(reduce(union,(true_support_bs(c) for c in cs))))

# ── SSA trajectories (Brusselator, project to X,Y) ────────────────────────────
const BRUS_RN_BS = @reaction_network begin
    k1A, 0 --> X
    k2,  2X + Y --> 3X
    k3B, X --> Y
    k4,  X --> 0
end

const DTS_BS    = [0.05, 0.2, 0.5]   # all times: x≥2, C(x,2)≥1 active, no aliasing at y<3
const T_SAVE_BS = cumsum(DTS_BS)
const N_SSA_BS  = parse(Int, get(ENV, "BRUSS_N_SSA", "2000"))

function ssa_trajectory_bs(c; n_traj=N_SSA_BS, y_init=2)
    A_ctx, B_ctx = c
    x0 = clamp(max(3, round(Int, A_ctx * OMEGA_BS)), 0, X_MAX_BS)
    u0 = [:X => x0, :Y => y_init]
    ps = [:k1A => Float64(A_ctx) * OMEGA_BS,
          :k2  => 1.0 / OMEGA_BS^2,
          :k3B => Float64(B_ctx),
          :k4  => 1.0]
    t_save = collect(T_SAVE_BS)
    pvs    = [zeros(N_BS) for _ in T_SAVE_BS]

    lk = ReentrantLock()
    Threads.@threads for _ in 1:n_traj
        sol = solve(JumpProblem(BRUS_RN_BS, u0, (0.0, maximum(T_SAVE_BS)), ps),
                    SSAStepper())
        for (k, t) in enumerate(t_save)
            state = sol(t)
            x = clamp(Int(round(state[1])), 0, X_MAX_BS)
            y = clamp(Int(round(state[2])), 0, Y_MAX_BS)
            lock(lk) do; pvs[k][lin_bs(x, y)] += 1.0; end
        end
    end

    vcat([copy(p0_bs(A_ctx, B_ctx; y_init=y_init))], [pv ./ n_traj for pv in pvs]), DTS_BS
end

function exact_trajectory_bs(c)
    A_ctx, B_ctx = c
    A_gen = sum(true_theta_bs(c)[j]*DIRS_BS[j] for j in 1:M_BS)
    ps = [copy(p0_bs(A_ctx, B_ctx))]
    for dt in DTS_BS
        p = expv(dt, A_gen, ps[end]; m=60); p=max.(p,0.0); p./=sum(p); push!(ps,p)
    end
    ps, DTS_BS
end

Random.seed!(42)
const TRAIN_A_BS = [0.80, 1.00, 1.20, 1.40]
const TRAIN_B_BS = [2.40, 2.80, 3.20, 3.60]
train_ctxs_bs = vec([(A, B) for A in TRAIN_A_BS, B in TRAIN_B_BS])
test_ctxs_bs  = [(0.90, 2.50), (1.10, 2.90), (1.00, 3.40), (1.30, 3.10), (1.25, 3.75)]

println("\nGenerating training data ($N_SSA_BS SSA trajectories per context)...")
flush(stdout)
train_data_bs = [(d = ssa_trajectory_bs(c); @printf "  train ctx (%.2f,%.2f) done\n" c[1] c[2]; flush(stdout); d) for c in train_ctxs_bs]
println("Generating test data...")
flush(stdout)
test_data_bs  = [(d = ssa_trajectory_bs(c); @printf "  test  ctx (%.2f,%.2f) done\n" c[1] c[2]; flush(stdout); d) for c in test_ctxs_bs]
exact_data_bs = [exact_trajectory_bs(c) for c in test_ctxs_bs]

println("\nTrue union support: ", union_support_bs(vcat(train_ctxs_bs, test_ctxs_bs)))

# ── Fisher (propensity flux) scales ───────────────────────────────────────────
# s[j] = mean over training states of E_p[propensity_j] = ||D_j*p||_1 / 2.
# Reparametrising α̃[j] = θ[j]*s[j] (encoder output) with D̃[j] = D[j]/s[j]
# (SemigroupLayer) keeps the generator unchanged but makes all α̃ O(1) regardless
# of Ω — fixing the sigmoid-saturation / gradient-scale issue at large Ω.
println("Computing Fisher scales...")
const FISHER_S_BS = let
    all_ps = [p for (ps, _) in train_data_bs for p in ps]
    s = zeros(M_BS)
    for j in 1:M_BS
        s[j] = mean(sum(abs, DIRS_BS[j] * p) / 2 for p in all_ps)
    end
    max.(s, 1e-10)   # guard zero-flux directions
end
let lo=minimum(FISHER_S_BS), hi=maximum(FISHER_S_BS)
    @printf "  scale range: [%.2e, %.2e]  (ratio %.0f×)\n" lo hi hi/lo
end

# Scaled library and SemigroupLayer used during training
const DIRS_BS_S = [DIRS_BS[j] / FISHER_S_BS[j] for j in 1:M_BS]
const SL_BS_S   = SemigroupLayer(spzeros(N_BS, N_BS), DIRS_BS_S; m_krylov=60)

# THETA_MAX for the α̃ space: 2× the largest true scaled rate across all contexts
const THETA_MAX_BS_S = let
    all_ctxs = vcat(train_ctxs_bs, test_ctxs_bs)
    2.0 * maximum(maximum(abs.(true_theta_bs(c) .* FISHER_S_BS)) for c in all_ctxs)
end
@printf "  THETA_MAX (Fisher space): %.3f\n" THETA_MAX_BS_S
for c in test_ctxs_bs
    θ=true_theta_bs(c)
    @printf "  c=(%.2f,%.2f)  θ₁=%.2f  θ₁₃=%.2f  θ₂₃=%.2f  θ₃₉=%.4f\n" c[1] c[2] θ[1] θ[13] θ[23] θ[39]
end

# ── SemigroupLayer for CME ────────────────────────────────────────────────────
# A0 = 0 for Brusselator: the full generator A(θ) = Σⱼ θⱼ Dⱼ with no constant term.
const SL_BS = SemigroupLayer(spzeros(N_BS, N_BS), DIRS_BS; m_krylov=60)

# Thin wrapper for evaluation / STLS — not used in the Zygote training graph.
function semigroup_loss_bs(θ_vec, p_data, dts)
    A = sum(θ_vec[j]*DIRS_BS[j] for j in 1:M_BS)
    _, loss = shared_gradient(A, DIRS_BS, p_data, dts; m_krylov=60)
    loss
end

# ── Channel-factorised encoder ────────────────────────────────────────────────
# θ_{r,k}(c) = THETA_MAX · σ(amp_r(c)) · softmax(dist_r(c))[k] · σ(filter_{r,k})
const N_CH_BS = length(CHANNELS_BS)   # 4

struct ChannelFactorizedEncoder{E, V}
    encoder::E        # 2 → 64 → 64 → (N_CH_BS + N_CH_BS*K_BS)
    gate_logits::V    # M=40 global gates (frozen at init value)
end
Flux.@layer ChannelFactorizedEncoder trainable=(encoder, gate_logits)

function (m::ChannelFactorizedEncoder)(c)
    out   = vec(m.encoder(ctx_vec_bs(c)))
    amps  = THETA_MAX_BS .* sigmoid.(out[1:N_CH_BS])
    dists = out[N_CH_BS+1:end]
    gates = sigmoid.(m.gate_logits)
    vcat([let idxs = (r-1)*K_BS+1:r*K_BS
              amps[r] .* softmax(dists[idxs]) .* gates[idxs]
          end for r in 1:N_CH_BS]...)
end

struct MaskedChannelEncoder{E, V}
    encoder::E
    support_mask::V
end
Flux.@layer MaskedChannelEncoder trainable=(encoder,)

function (m::MaskedChannelEncoder)(c)
    out  = vec(m.encoder(ctx_vec_bs(c)))
    amps = THETA_MAX_BS .* sigmoid.(out[1:N_CH_BS])
    dists = out[N_CH_BS+1:end]
    vcat([let idxs = (r-1)*K_BS+1:r*K_BS
              amps[r] .* softmax(dists[idxs]) .* m.support_mask[idxs]
          end for r in 1:N_CH_BS]...)
end

function base_encoder_cme()
    Chain(
        Dense(2 => 64, tanh),
        Dense(64 => 64, tanh),
        Dense(64 => N_CH_BS + N_CH_BS*K_BS; init=Flux.zeros32),
    ) |> f64
end

# ── Global-simplex encoder (Step 2 architecture) ──────────────────────────────
# θ_{r,k}(c) = THETA_MAX · σ(a_r(c)) · softmax(β_r)[k]
#
# Key change from ChannelFactorizedEncoder:
#   - basis_logits β_r is GLOBAL (shared across all contexts)
#   - encoder outputs only N_CH_BS amplitudes, not N_CH_BS + N_CH_BS*K_BS
#   - no gate_logits — single sparsity source (entropy penalty on π_r)
#
# This eliminates the trilinear scale ambiguity and ensures basis selection
# is consistent across contexts.

struct GlobalSimplexEncoder{E}
    encoder      :: E                    # 2 → 64 → 64 → N_CH_BS
    basis_logits :: Matrix{Float64}      # N_CH_BS × K_BS, global
    theta_max    :: Float64              # per-instance scale (supports Fisher reparametrisation)
end
Flux.@layer GlobalSimplexEncoder trainable=(encoder, basis_logits)

function (m::GlobalSimplexEncoder)(c, τ::Float64=1.0)
    amps = m.theta_max .* sigmoid.(vec(m.encoder(c)))
    vcat([amps[r] .* softmax(m.basis_logits[r, :] ./ τ) for r in 1:N_CH_BS]...)
end

# Stage 2: amplitude-only refit with hard support mask (one-hot per channel).
struct MaskedGlobalSimplexEncoder{E}
    encoder      :: E
    support_mask :: Vector{Float64}    # one-hot per channel, frozen
    theta_max    :: Float64
end
Flux.@layer MaskedGlobalSimplexEncoder trainable=(encoder,)

function (m::MaskedGlobalSimplexEncoder)(c, τ::Float64=1.0)   # τ unused, hard mask
    amps = m.theta_max .* sigmoid.(vec(m.encoder(c)))
    vcat([amps[r] .* m.support_mask[(r-1)*K_BS+1:r*K_BS] for r in 1:N_CH_BS]...)
end

# Slim encoder: outputs only N_CH_BS amplitude logits.
function base_encoder_cme_slim()
    Chain(
        Dense(2 => 64, tanh),
        Dense(64 => 64, tanh),
        Dense(64 => N_CH_BS; init=Flux.zeros32),
    ) |> f64
end
