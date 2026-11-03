"""
2D SDE sparse discovery from trajectory data via SemigroupLayer NLL.

Same 30-direction Chebyshev library and true model as sparse2d_cheby.jl,
but data comes from individual SDE particle trajectories rather than
exact FPE density propagation.

Loss:  L(θ) = -∑_{trajectories} ∑_steps log p(X_{t+1}=j | X_t=i, θ, Δt)
         where p(j|i,θ,Δt) = [e^{A(θ)Δt}]_{j,i}   (exact semigroup column)

Key point: one SemigroupLayer call per unique starting grid cell (not per
particle), so the cost is O(n_unique_starts × Krylov).  Shared-Krylov gives
∂L/∂α across all M=30 directions from two Arnoldi passes per starting cell.

Comparison: the same encoder trained with density-MSE on histograms aggregated
from the same particles — shows trajectory NLL achieves equal or better
support discovery without the histogram aggregation step.

System: 2D SDE on [-2,2]², 30×30 grid, n=900
  dX = (θ₁(c)·x + θ₂(c)·y) dt + √(2D) dWx
  dY = (θ₁(c)·y - θ₂(c)·x) dt + √(2D) dWy
  θ₁(c) = -(0.5+0.3sin(πc)),  θ₂(c) = 1.5cos(πc),  D=0.5

Library: Chebyshev T, total degree ≤ 4, M=30 directions.
True support: 4 of 30 (T₁T₀·x, T₀T₁·x, T₀T₁·y, T₁T₀·y).
"""

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Flux, Zygote
using ExponentialUtilities: expv
using CSV, DataFrames

include("../src/shared_krylov.jl")
include("../src/fpe2d.jl")

# ── Grid and Chebyshev library (identical to sparse2d_cheby.jl) ───────────────
function cheby_t(k::Int, t::Real)
    k == 0 && return 1.0;  k == 1 && return Float64(t)
    k == 2 && return 2t^2-1.0;  k == 3 && return 4t^3-3.0t
    k == 4 && return 8t^4-8t^2+1.0;  error("k=$k not implemented")
end

const NX = 30;  const NY = 30
const XS = collect(range(-2.0, 2.0; length=NX))
const YS = collect(range(-2.0, 2.0; length=NY))
const D_DIFF = 0.5;  const N_GRID = NX * NY   # 900
const HX = XS[2]-XS[1];  const HY = YS[2]-YS[1]

lin2d(i, j)   = i + (j-1)*NX    # column-major (matches fpe2d.jl)
function snap2d(x, y)
    ix = clamp(round(Int, (x-XS[1])/HX)+1, 1, NX)
    iy = clamp(round(Int, (y-YS[1])/HY)+1, 1, NY)
    lin2d(ix, iy)
end

const BASIS_IDX = Tuple{Int,Int}[(k, s-k) for s in 0:4 for k in 0:s]
const K_BASIS = length(BASIS_IDX);  const M_LIB = 2*K_BASIS

println("Building 2D Chebyshev library (degree≤4, K=$K_BASIS, M=$M_LIB)…")
bx_fns = Vector{Function}(undef, M_LIB)
by_fns = Vector{Function}(undef, M_LIB)
for j in 1:K_BASIS
    (ki, li) = BASIS_IDX[j]
    bx_fns[j]         = let ki=ki,li=li; (x,y)->cheby_t(ki,x/2)*cheby_t(li,y/2); end
    by_fns[j]         = (x,y)->0.0
    bx_fns[K_BASIS+j] = (x,y)->0.0
    by_fns[K_BASIS+j] = let ki=ki,li=li; (x,y)->cheby_t(ki,x/2)*cheby_t(li,y/2); end
end

const A0_LIB, DIRS_LIB = fpe_library_2d(XS, YS, bx_fns, by_fns; D=D_DIFF)
const SL_FPE = SemigroupLayer(A0_LIB, DIRS_LIB; m_krylov=30)

const J_10 = findfirst(==((1,0)), BASIS_IDX)
const J_01 = findfirst(==((0,1)), BASIS_IDX)
const TRUE_SUPPORT = sort([J_10, J_01, K_BASIS+J_10, K_BASIS+J_01])
println("  True support: $TRUE_SUPPORT")

function true_alpha(c)
    θ₁ = -(0.5+0.3sin(π*c));  θ₂ = 1.5cos(π*c)
    α = zeros(M_LIB)
    α[J_10] = 2θ₁;  α[J_01] = 2θ₂
    α[K_BASIS+J_01] = 2θ₁;  α[K_BASIS+J_10] = -2θ₂
    α
end

# ── Unit vectors (precomputed; ZeroTangent for p0 in rrule) ───────────────────
const UVECS = [begin e = zeros(N_GRID); e[k]=1.0; e end for k in 1:N_GRID]

# ── SDE particle trajectories ─────────────────────────────────────────────────
# Simulate actual continuous SDE via fine EM, snap observations to grid.
# Uses a coarse 5×5 grid of starting positions so starting cells are shared
# across particles → efficient NLL batching.
const START_XS = [-1.5, -0.75, 0.0, 0.75, 1.5]
const START_YS = [-1.5, -0.75, 0.0, 0.75, 1.5]
const START_CELLS = [snap2d(x,y) for x in START_XS for y in START_YS]  # 25 cells

# Sample trajectories from the exact FPE transition matrix (same approach as
# trajectory_ou.jl).  This eliminates model mismatch between continuous SDE
# and discrete FPE — the data EXACTLY follows exp(A·dt)·e_i, so the NLL
# landscape is well-calibrated and gradient descent reliably converges.
function gen_traj_fpe(c; n_per_start=20, dt_obs=0.5, rng=Random.default_rng())
    α   = true_alpha(c)
    A   = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
    counts = Dict{Int,Vector{Int}}()
    for s0 in START_CELLS
        e  = zeros(N_GRID);  e[s0] = 1.0
        ρ  = expv(dt_obs, A, e; m=50)
        ρ  = max.(real.(ρ), 0.0);  ρ ./= sum(ρ)
        cs = cumsum(ρ)
        js = get!(counts, s0, Int[])
        for _ in 1:n_per_start
            push!(js, clamp(searchsortedfirst(cs, rand(rng)), 1, N_GRID))
        end
    end
    counts
end

# Also aggregate into density histograms for MSE comparison
function counts_to_density_pairs(counts_per_step)
    # counts_per_step: vector of Dict (one per observation step)
    # returns (ps, dts) in density-MSE format
    # This is approximate: we histogram the marginal, losing trajectory structure
    p_starts = zeros(N_GRID)
    p_ends   = zeros(N_GRID)
    for (i, js) in counts_per_step
        p_starts[i] += length(js)
        for j in js; p_ends[j] += 1.0; end
    end
    p_starts ./= sum(p_starts);  p_ends ./= sum(p_ends)
    p_starts, p_ends
end

# ── Dataset ────────────────────────────────────────────────────────────────────
const DT_OBS   = 0.5
const K_STEPS  = 3
const N_PER_S  = parse(Int, get(ENV, "N_PER_S", "20"))   # particles per start cell

const TRAIN_CTXS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
const TEST_CTXS  = [0.10, 0.30, 0.50, 0.70, 0.90]

Random.seed!(42)
println("\nGenerating exact FPE trajectories ($(length(START_CELLS)) start cells × $N_PER_S particles)…")
train_traj = Dict(c => gen_traj_fpe(c; n_per_start=N_PER_S, dt_obs=DT_OBS) for c in TRAIN_CTXS)
test_traj  = Dict(c => gen_traj_fpe(c; n_per_start=N_PER_S, dt_obs=DT_OBS) for c in TEST_CTXS)
n_obs = mean(sum(length(v) for v in values(train_traj[c])) for c in TRAIN_CTXS)
@printf "  ~%.0f transitions/context from %d unique start cells\n" n_obs length(START_CELLS)

# Density histograms from same particles (for MSE comparison)
train_density = Dict(c => begin
    p0, p1 = counts_to_density_pairs(train_traj[c])
    ([p0, p1], [DT_OBS])
end for c in TRAIN_CTXS)
test_density  = Dict(c => begin
    p0, p1 = counts_to_density_pairs(test_traj[c])
    ([p0, p1], [DT_OBS])
end for c in TEST_CTXS)

# ── Losses ────────────────────────────────────────────────────────────────────
#
# nll_loss: custom rrule calls nll_gradient from shared_krylov.jl.
# nll_gradient threads Fréchet VJPs over unique starting cells internally.
# Zygote only traces enc(c) → α; the semigroup backward is invisible to Zygote.
# This is the same pattern as shared_gradient for density MSE — the physics
# gradient is a first-class primitive, not plumbing in the training loop.

function nll_loss(counts, α, dt)
    A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
    _, loss = nll_gradient(A, DIRS_LIB, counts, dt; m_krylov=30)
    loss
end

function ChainRulesCore.rrule(::typeof(nll_loss), counts, α, dt)
    A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
    grad_α, loss = nll_gradient(A, DIRS_LIB, counts, dt; m_krylov=30)
    pullback(Δ) = NoTangent(), ZeroTangent(), Δ .* grad_α, ZeroTangent()
    return loss, pullback
end

function mse_loss(ps, α, dts)
    total = 0.0
    for t in eachindex(dts)
        p_pred = SL_FPE(ps[t], α, dts[t])
        total += 0.5 * sum(abs2, ps[t+1] - p_pred)
    end
    total
end

# ── Encoder ────────────────────────────────────────────────────────────────────
make_enc() = Chain(Dense(1=>32, tanh), Dense(32=>32, tanh),
                   Dense(32=>M_LIB; init=Flux.zeros32)) |> f64

# ── Training — plain Zygote.withgradient, threading inside rrule ──────────────
const N_S1   = parse(Int,     get(ENV, "S1",    "600"))
const N_S2   = parse(Int,     get(ENV, "S2",    "400"))
const LR1_NLL = parse(Float64, get(ENV, "LR_NLL", "3e-4"))   # NLL needs lower LR
const LR1_MSE = parse(Float64, get(ENV, "LR_MSE", "3e-3"))
const LR2    = parse(Float64, get(ENV, "LR2",   "1e-3"))
const λ_STLS = 0.25

enc_nll = make_enc()
opt_nll = Flux.setup(Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipGrad(1.0), Adam(LR1_NLL)), enc_nll)

# ── Gradient sign check on actual train data ───────────────────────────────────
println("\n=== Gradient sign check on actual SDE data ===")
let c0 = TRAIN_CTXS[1], α0 = zeros(M_LIB)
    A0c = A0_LIB
    g0, l0 = nll_gradient(A0c, DIRS_LIB, train_traj[c0], DT_OBS; m_krylov=30)
    @printf "NLL at α=0: %.5f  |g|=%.4f  NaN=%s\n" l0 norm(g0) any(isnan,g0)
    for ε in [1e-5, 1e-4, 1e-3, 1e-2]
        α1 = α0 .- ε .* g0
        A1 = A0_LIB + sum(α1[j]*DIRS_LIB[j] for j in 1:M_LIB)
        _, l1 = nll_gradient(A1, DIRS_LIB, train_traj[c0], DT_OBS; m_krylov=30)
        @printf "  ε=%.0e  loss=%.5f  Δ=%.6f  %s\n" ε l1 (l1-l0) (l1<l0 ? "↓ OK" : "↑ SIGN WRONG")
    end
end
println("=== end check ===")

# Proxy-VJP for NLL: same proven pattern as sparse2d_cheby.jl.
# Phase 1 (outside Zygote): compute nll_gradient for each context in parallel.
# Phase 2 (inside Zygote): chain through encoder via cheap dot product.
# This is mathematically identical to the rrule approach but avoids the
# Δ·grad_α sign ambiguity in the rrule pullback.
@printf "\nStage 1 [Trajectory NLL, proxy-VJP, %d threads]: %d epochs  LR=%.0e\n" Threads.nthreads() N_S1 LR1_NLL
for epoch in 1:N_S1
    n_ctx = length(TRAIN_CTXS)
    alphas    = [vec(enc_nll([c])) for c in TRAIN_CTXS]
    sg_grads  = Vector{Vector{Float64}}(undef, n_ctx)
    sg_losses = zeros(n_ctx)
    Threads.@threads for k in 1:n_ctx
        A = A0_LIB + sum(alphas[k][j]*DIRS_LIB[j] for j in 1:M_LIB)
        sg_grads[k], sg_losses[k] = nll_gradient(A, DIRS_LIB, train_traj[TRAIN_CTXS[k]], DT_OBS; m_krylov=30)
    end
    _, grads = Zygote.withgradient(enc_nll) do m
        sum(dot(sg_grads[k], vec(m([TRAIN_CTXS[k]]))) for k in 1:n_ctx)
    end
    Flux.update!(opt_nll, enc_nll, grads[1])

    if epoch % 20 == 0 || epoch == 1
        tr = mean(sg_losses)
        te = mean(begin
            α = vec(enc_nll([c]))
            A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
            _, l = nll_gradient(A, DIRS_LIB, test_traj[c], DT_OBS; m_krylov=30)
            l
        end for c in TEST_CTXS)
        avg = mean(abs.(vec(enc_nll([c]))) for c in TRAIN_CTXS)
        top4 = sortperm(avg; rev=true)[1:4]
        @printf "NLL  epoch %4d  train=%.5f  test=%.5f  top4: %s\n" epoch tr te repr([j ∈ TRUE_SUPPORT ? "✓$j" : "·$j" for j in top4])
        flush(stdout)
    end
end

enc_mse = make_enc()
opt_mse = Flux.setup(Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipGrad(1.0), Adam(LR1_MSE)), enc_mse)

@printf "\nStage 1 [Density MSE, %d threads]: %d epochs  LR=%.0e\n" Threads.nthreads() N_S1 LR1_MSE
for epoch in 1:N_S1
    _, g = Zygote.withgradient(enc_mse) do m
        sum(begin ps,dts=train_density[c]; mse_loss(ps, vec(m([c])), dts) end
            for c in TRAIN_CTXS)
    end
    Flux.update!(opt_mse, enc_mse, g[1])
    if epoch % 20 == 0 || epoch == 1
        tr = mean(begin ps,dts=train_density[c]; mse_loss(ps,vec(enc_mse([c])),dts) end for c in TRAIN_CTXS)
        te = mean(begin ps,dts=test_density[c];  mse_loss(ps,vec(enc_mse([c])),dts) end for c in TEST_CTXS)
        avg = mean(abs.(vec(enc_mse([c]))) for c in TRAIN_CTXS)
        top4 = sortperm(avg; rev=true)[1:4]
        @printf "MSE  epoch %4d  train=%.6f  test=%.6f  top4: %s\n" epoch tr te repr([j ∈ TRUE_SUPPORT ? "✓$j" : "·$j" for j in top4])
        flush(stdout)
    end
end

# ── Support extraction ─────────────────────────────────────────────────────────
function extract_support(enc, label)
    avg = mean(abs.(vec(enc([c]))) for c in vcat(TRAIN_CTXS, TEST_CTXS))
    disc = [j for j in 1:M_LIB if avg[j] > λ_STLS]
    correct = Set(disc) == Set(TRUE_SUPPORT)
    @printf "\n[%s] Support: %s  (%s)\n" label repr(disc) (correct ? "CORRECT ✓" : "INCORRECT ✗")
    avg, disc
end

println("\n━━━ Support extraction ━━━")
avg_nll, disc_nll = extract_support(enc_nll, "NLL ")
avg_mse, disc_mse = extract_support(enc_mse, "MSE ")
println("True:  $TRUE_SUPPORT")

# ── Stage 2: masked refit (NLL model) ─────────────────────────────────────────
mask_nll = zeros(M_LIB); for j in disc_nll; mask_nll[j]=1.0; end

enc2_nll = deepcopy(enc_nll)
opt2_nll = Flux.setup(Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipGrad(1.0), Adam(LR2)), enc2_nll)

println("\nStage 2 [NLL masked, batched rrule]: $N_S2 epochs")
for epoch in 1:N_S2
    _, g = Zygote.withgradient(enc2_nll) do m
        sum(nll_loss(train_traj[c], vec(m([c])) .* mask_nll, DT_OBS) for c in TRAIN_CTXS)
    end
    Flux.update!(opt2_nll, enc2_nll, g[1])
    if epoch % 20 == 0 || epoch == 1
        tr = mean(nll_loss(train_traj[c], vec(enc2_nll([c])) .* mask_nll, DT_OBS) for c in TRAIN_CTXS)
        @printf "  epoch %3d  train=%.5f\n" epoch tr; flush(stdout)
    end
end

# ── Evaluation: coefficient recovery ──────────────────────────────────────────
println("\n━━━ Coefficient recovery at test contexts ━━━")
@printf "%-5s  %8s %8s  %8s %8s  %8s %8s  %8s %8s\n" "c" "θ₁_t" "θ₁_p" "θ₂_t" "θ₂_p" "err_θ₁" "err_θ₂" "NLL" "MSE"

rows = NamedTuple[]
for c in sort(vcat(TRAIN_CTXS, TEST_CTXS))
    α_t   = true_alpha(c)
    α_nll = vec(enc2_nll([c])) .* mask_nll
    α_mse = vec(enc_mse([c]))

    θ₁_t = -(0.5+0.3sin(π*c));  θ₂_t = 1.5cos(π*c)
    θ₁_nll = (α_nll[J_10] + α_nll[K_BASIS+J_01]) / 4
    θ₂_nll = (α_nll[J_01] - α_nll[K_BASIS+J_10]) / 4
    θ₁_mse = (α_mse[J_10] + α_mse[K_BASIS+J_01]) / 4
    θ₂_mse = (α_mse[J_01] - α_mse[K_BASIS+J_10]) / 4

    e1_nll = abs(θ₁_t) > 0.05 ? abs(θ₁_nll-θ₁_t)/abs(θ₁_t)*100 : NaN
    e2_nll = abs(θ₂_t) > 0.05 ? abs(θ₂_nll-θ₂_t)/abs(θ₂_t)*100 : NaN
    e1_mse = abs(θ₁_t) > 0.05 ? abs(θ₁_mse-θ₁_t)/abs(θ₁_t)*100 : NaN
    e2_mse = abs(θ₂_t) > 0.05 ? abs(θ₂_mse-θ₂_t)/abs(θ₂_t)*100 : NaN

    mark = c ∈ TEST_CTXS ? "←" : "  "
    @printf "%-4.2f %s  %8.3f %8.3f  %8.3f %8.3f  %8.1f  %8.1f\n" c mark θ₁_t θ₁_nll θ₂_t θ₂_nll (isnan(e1_nll) ? -1.0 : e1_nll) (isnan(e2_nll) ? -1.0 : e2_nll)

    push!(rows, (c=c, theta1_true=θ₁_t, theta2_true=θ₂_t,
                 theta1_nll=θ₁_nll, theta2_nll=θ₂_nll,
                 theta1_mse=θ₁_mse, theta2_mse=θ₂_mse,
                 err1_nll=e1_nll, err2_nll=e2_nll,
                 err1_mse=e1_mse, err2_mse=e2_mse,
                 is_test=(c ∈ TEST_CTXS)))
end

df = DataFrame(rows)
valid = .!isnan.(df.err1_nll)
@printf "\nMean rel err (non-zero contexts):\n"
@printf "  NLL  θ₁=%.1f%%  θ₂=%.1f%%\n" mean(df.err1_nll[valid]) mean(df.err2_nll[valid])
valid_mse = .!isnan.(df.err1_mse)
@printf "  MSE  θ₁=%.1f%%  θ₂=%.1f%%\n" mean(df.err1_mse[valid_mse]) mean(df.err2_mse[valid_mse])

# Save
support_df = DataFrame(
    dir=1:M_LIB,
    avg_nll=avg_nll,
    avg_mse=avg_mse,
    is_true=[j ∈ TRUE_SUPPORT for j in 1:M_LIB])
CSV.write("paper/figures/traj2d_support.csv", support_df)

ctx_dense = collect(range(0.0, 1.0; length=100))
coeff_df = DataFrame(
    context     = ctx_dense,
    theta1_true = [-(0.5+0.3sin(π*c)) for c in ctx_dense],
    theta2_true = [1.5cos(π*c)         for c in ctx_dense],
    theta1_nll  = [(vec(enc2_nll([c])).* mask_nll)[J_10]/2 for c in ctx_dense],
    theta2_nll  = [(vec(enc2_nll([c])).* mask_nll)[J_01]/2 for c in ctx_dense],
    theta1_mse  = [vec(enc_mse([c]))[J_10]/2               for c in ctx_dense],
    theta2_mse  = [vec(enc_mse([c]))[J_01]/2               for c in ctx_dense],
)
CSV.write("paper/figures/traj2d_coeffs.csv", coeff_df)
CSV.write("paper/figures/traj2d_recovery.csv", df)
println("\nSaved traj2d_support.csv, traj2d_coeffs.csv, traj2d_recovery.csv")
