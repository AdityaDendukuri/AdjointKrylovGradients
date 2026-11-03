"""
2D sparse FPE identification with Chebyshev basis.

System: 2D SDE on [-2,2]², 30×30 grid, n=900
  dX = b(X;c)dt + √(2D)dW

Library: Chebyshev T polynomials, total degree ≤ 4
  φ_{k,l}(x,y) = Tₖ(x/2)·Tₗ(y/2),  k+l ≤ 4
  K = 15 scalar basis functions
  M = 30 directions: j=1..K → drift (φⱼ,0); j=K+1..2K → drift (0,φⱼ)

True drift: θ₁(c)·(x,y) + θ₂(c)·(y,-x)
  θ₁(c) = -(0.5 + 0.3·sin(πc))    contraction
  θ₂(c) =  1.5·cos(πc)             rotation

In Chebyshev coordinates, 4 active directions out of 30:
  φ_{1,0} in x-dir: coeff  2θ₁(c)   [x contraction]
  φ_{0,1} in x-dir: coeff  2θ₂(c)   [y→x rotation]
  φ_{0,1} in y-dir: coeff  2θ₁(c)   [y contraction]
  φ_{1,0} in y-dir: coeff -2θ₂(c)   [x→y rotation]

Training: SemigroupLayer rrule (single-threaded) or proxy-VJP (multi-threaded).

Run: julia -t 8 --project=experiments experiments/sparse2d_cheby.jl
"""

#=
TODO(aditya): This FPE experiment is already an SDE identification problem.
The drift b(x,y;c) = theta_1(c) * (x,y) + theta_2(c) * (y,-x) is a 2D OU
family; we infer its SDE parameters from density snapshots rather than
trajectories.

The same linear-generator view covers CIR/state-dependent diffusion:
  A(a,b,sigma^2) = a*A_drift_const + b*A_drift_linear
                 + sigma^2*A_diffusion_state_dep.

Future extension: trajectory observations. Replace the density loss with
  log p(x1 | x0, theta) = log [exp(A(theta) * dt)]_{x1,x0}.
Same semigroup in principle, but much noisier gradients and likely a different
Krylov estimator for individual propagator entries.
=#

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Flux, Zygote, ChainRulesCore
using ExponentialUtilities: expv
using CairoMakie, Printf

include("../src/shared_krylov.jl")
include("../src/fpe2d.jl")

# ── Chebyshev T polynomials ────────────────────────────────────────────────────
function cheby_t(k::Int, t::Real)
    k == 0 && return 1.0
    k == 1 && return Float64(t)
    k == 2 && return 2t^2 - 1.0
    k == 3 && return 4t^3 - 3.0t
    k == 4 && return 8t^4 - 8t^2 + 1.0
    error("degree k=$k not implemented")
end

# ── Grid ───────────────────────────────────────────────────────────────────────
const NX = 30;  const NY = 30
const XS = collect(range(-2.0, 2.0; length=NX))
const YS = collect(range(-2.0, 2.0; length=NY))
const D_DIFF = 0.5
const N_GRID = NX * NY   # 900

# ── Chebyshev library, total degree ≤ 4 ───────────────────────────────────────
# Enumerate (k,l) with k+l ≤ 4, grouped by total degree
const BASIS_IDX = Tuple{Int,Int}[(k, s-k) for s in 0:4 for k in 0:s]
# (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)
const K_BASIS = length(BASIS_IDX)   # 15
const M_LIB   = 2 * K_BASIS         # 30

println("Building 2D Chebyshev library  (degree≤4, K=$K_BASIS, M=$M_LIB)…")

# Build drift function lists: first K are (φⱼ,0), next K are (0,φⱼ)
bx_fns = Vector{Function}(undef, M_LIB)
by_fns = Vector{Function}(undef, M_LIB)
for j in 1:K_BASIS
    (ki, li) = BASIS_IDX[j]
    bx_fns[j]         = let ki=ki, li=li; (x,y) -> cheby_t(ki, x/2) * cheby_t(li, y/2); end
    by_fns[j]         = (x,y) -> 0.0
    bx_fns[K_BASIS+j] = (x,y) -> 0.0
    by_fns[K_BASIS+j] = let ki=ki, li=li; (x,y) -> cheby_t(ki, x/2) * cheby_t(li, y/2); end
end

const A0_LIB, DIRS_LIB = fpe_library_2d(XS, YS, bx_fns, by_fns; D=D_DIFF)

# True active direction indices (1-indexed)
const J_10 = findfirst(==((1,0)), BASIS_IDX)   # φ_{1,0} = x/2
const J_01 = findfirst(==((0,1)), BASIS_IDX)   # φ_{0,1} = y/2
# Active: J_10 (x-dir), J_01 (x-dir), K+J_10 (y-dir), K+J_01 (y-dir)
const TRUE_SUPPORT = sort([J_10, J_01, K_BASIS+J_10, K_BASIS+J_01])
println("  J_10=$(J_10)  J_01=$(J_01)  True support: $TRUE_SUPPORT")

# ── True parameter map ─────────────────────────────────────────────────────────
# Chebyshev coefficients for true drift θ₁(c)·(x,y) + θ₂(c)·(y,-x):
#   bx = θ₁x + θ₂y  = 2θ₁·φ_{1,0} + 2θ₂·φ_{0,1}
#   by = θ₁y - θ₂x  = 2θ₁·φ_{0,1} - 2θ₂·φ_{1,0}
function true_alpha(c)
    θ₁ = -(0.5 + 0.3*sin(π*c))
    θ₂ =  1.5*cos(π*c)
    α = zeros(M_LIB)
    α[J_10]            =  2θ₁
    α[J_01]            =  2θ₂
    α[K_BASIS + J_01]  =  2θ₁
    α[K_BASIS + J_10]  = -2θ₂
    α
end

# Sanity check: generator column sums should be ≈ 0
let α = true_alpha(0.3)
    A = Matrix(A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB))
    @printf "Column-sum error at c=0.3: %.2e  (should be < 1e-10)\n" maximum(abs.(sum(A; dims=1)))
end

# ── Data generation ────────────────────────────────────────────────────────────
function gaussian_ic(; μx=0.0, μy=0.0, σ=0.5)
    p = [exp(-((XS[i]-μx)^2 + (YS[j]-μy)^2)/(2σ^2))
         for i in 1:NX, j in 1:NY]
    p = vec(p);  p ./= sum(p)
end

const DTS_TRAIN = [0.5, 1.0, 2.0]
const DT_EXTRA  = 6.0   # 3× longest training step

function gen_extra(c; p0=gaussian_ic())
    α = true_alpha(c)
    A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
    p1 = expv(DT_EXTRA, A, p0; m=50)
    p1 = max.(p1, 0.0);  p1 ./= sum(p1)
    [p0, p1], [DT_EXTRA]
end

function gen_traj(c; dts=DTS_TRAIN, p0=gaussian_ic())
    α = true_alpha(c)
    A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
    ps = [p0]
    for dt in dts
        p = expv(dt, A, ps[end]; m=50)
        p = max.(p, 0.0);  p ./= sum(p)
        push!(ps, p)
    end
    ps, dts
end

Random.seed!(42)
const TRAIN_CTXS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
const TEST_CTXS  = [0.10, 0.30, 0.50, 0.70, 0.90]

println("Generating data…")
train_data  = [gen_traj(c) for c in TRAIN_CTXS]
test_data   = [gen_traj(c) for c in TEST_CTXS]
extra_data  = [gen_extra(c) for c in TEST_CTXS]
@printf "  n=%d  M=%d  K=%d  train=%d  test=%d\n" N_GRID M_LIB K_BASIS length(TRAIN_CTXS) length(TEST_CTXS)

# Quick check: true alpha magnitudes at a few contexts
println("\nTrue Chebyshev coefficients at selected contexts:")
@printf "  %-6s  %-12s  %-12s  %-12s  %-12s\n" "c" "α[J10,x]" "α[J01,x]" "α[J01,y]" "α[J10,y]"
for c in [0.1, 0.3, 0.5, 0.7, 0.9]
    α = true_alpha(c)
    @printf "  %-6.2f  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n" c α[J_10] α[J_01] α[K_BASIS+J_01] α[K_BASIS+J_10]
end

# ── Hyperparameters ────────────────────────────────────────────────────────────
const N_STAGE1 = parse(Int,     get(ENV, "S1",   "800"))
const N_STAGE2 = parse(Int,     get(ENV, "S2",   "600"))
const LR_S1    = parse(Float64, get(ENV, "LR1",  "3e-3"))
const LR_S2    = parse(Float64, get(ENV, "LR2",  "1e-3"))
const λ_L1     = parse(Float64, get(ENV, "L1",   "5e-3"))
const λ_STLS   = parse(Float64, get(ENV, "STLS", "0.25"))
const MAX_LOSS  = 1e3

# ── NODE baselines ─────────────────────────────────────────────────────────────
const N_SUB = 4   # RK4 sub-steps per training interval

function make_node(hidden)
    Chain(Dense(N_GRID+1 => hidden, tanh),
          Dense(hidden => hidden, tanh),
          Dense(hidden => N_GRID)) |> f64
end

node_lg = make_node(64)   # ~120K params
node_sm = make_node(16)   # ~29K params

node_lg_opt = Flux.setup(Adam(LR_S1), node_lg)
node_sm_opt = Flux.setup(Adam(LR_S1), node_sm)

function rk4_step(model, p, c, dt)
    inp(q) = vcat(q, [c])
    k1 = model(inp(p))
    k2 = model(inp(p .+ 0.5dt.*k1))
    k3 = model(inp(p .+ 0.5dt.*k2))
    k4 = model(inp(p .+    dt.*k3))
    p .+ (dt/6).*(k1 .+ 2k2 .+ 2k3 .+ k4)
end

function node_loss(model, ps, c, dts; n_sub=N_SUB)
    loss = 0.0;  p = ps[1]
    for (i, dt) in enumerate(dts)
        dt_s = dt / n_sub
        for _ in 1:n_sub;  p = rk4_step(model, p, c, dt_s);  end
        loss += 0.5 * sum(abs2, p .- ps[i+1])
    end
    loss
end

function node_predict(model, p0, c, dts; n_sub=N_SUB)
    ps = [p0];  p = p0
    for dt in dts
        dt_s = dt / n_sub
        for _ in 1:n_sub;  p = rk4_step(model, p, c, dt_s);  end
        push!(ps, p)
    end
    ps
end

count_params(m) = sum(length(p) for p in Flux.trainables(m))

# ── Model: context encoder → 30 Chebyshev coefficients ────────────────────────
# ~2K params, O(1) in state dimension n=900
enc = Chain(
    Dense(1 => 32, tanh),
    Dense(32 => 32, tanh),
    Dense(32 => M_LIB; init=Flux.zeros32),
) |> f64

opt = Flux.setup(Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipGrad(1.0),
    Adam(LR_S1),
), enc)

@printf "\nModel parameter counts:\n"
@printf "  Semigroup encoder (O(1) in n)  %6d\n" count_params(enc)
@printf "  NODE-small                     %6d\n" count_params(node_sm)
@printf "  NODE-large                     %6d\n" count_params(node_lg)
flush(stdout)

# ── SemigroupLayer for FPE ─────────────────────────────────────────────────────
# A0_LIB holds the diffusion base; DIRS_LIB are the 30 Chebyshev drift directions.
const SL_FPE = SemigroupLayer(A0_LIB, DIRS_LIB; m_krylov=30)

# ── Training ───────────────────────────────────────────────────────────────────
# Primary path: SemigroupLayer rrule — enc(c) → θ → SL_FPE(p_t, θ, dt) → loss.
# Zygote differentiates through the encoder; the Krylov Fréchet VJP handles
# the semigroup step via the registered rrule.
#
# Parallel path (proxy-VJP): when Threads.nthreads() > 1 we precompute
# ∂L/∂α on threads (Zygote is not thread-safe inside a tape) then let Zygote
# differentiate only the cheap dot(∂L/∂α, enc(c)) step.  Mathematically
# identical to the SemigroupLayer path.

function train_step!(enc, opt, data, ctxs)
    n = length(data)
    if Threads.nthreads() > 1
        # ── Parallel proxy-VJP ─────────────────────────────────────────────────
        alphas    = [vec(enc([c])) for c in ctxs]
        sg_grads  = Vector{Vector{Float64}}(undef, n)
        sg_losses = Vector{Float64}(undef, n)
        Threads.@threads for i in 1:n
            (ps, dts) = data[i]
            α = alphas[i]
            A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
            g, l = shared_gradient(A, DIRS_LIB, ps, dts; m_krylov=30)
            sg_grads[i]  = (isfinite(l) && l ≤ MAX_LOSS) ? g : zeros(M_LIB)
            sg_losses[i] = (isfinite(l) && l ≤ MAX_LOSS) ? l : 0.0
        end
        _, grads = Zygote.withgradient(enc) do m
            total = 0.0
            for i in 1:n
                total += dot(sg_grads[i], vec(m([ctxs[i]])))
            end
            total
        end
        Flux.update!(opt, enc, grads[1])
        return sum(sg_losses)
    else
        # ── SemigroupLayer path (single-threaded) ──────────────────────────────
        loss_val = Ref(0.0)
        _, grads = Zygote.withgradient(enc) do m
            total = 0.0
            for i in 1:n
                (ps, dts) = data[i]
                θ = vec(m([ctxs[i]]))
                for t in eachindex(dts)
                    p_pred = SL_FPE(ps[t], θ, dts[t])
                    r = ps[t+1] - p_pred
                    l = 0.5 * sum(abs2, r)
                    isfinite(l) && l ≤ MAX_LOSS && (total += l)
                end
            end
            total
        end
        Flux.update!(opt, enc, grads[1])
        # recompute scalar loss for logging (no grad needed)
        for i in 1:n
            (ps, dts) = data[i]
            α = vec(enc([ctxs[i]]))
            A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
            _, l = shared_gradient(A, DIRS_LIB, ps, dts; m_krylov=30)
            isfinite(l) && l ≤ MAX_LOSS && (loss_val[] += l)
        end
        return loss_val[]
    end
end

function eval_loss(enc, data, ctxs)
    total = 0.0
    for (i, c) in enumerate(ctxs)
        (ps, dts) = data[i]
        α = vec(enc([c]))
        A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
        _, l = shared_gradient(A, DIRS_LIB, ps, dts; m_krylov=30)
        total += (isfinite(l) && l < MAX_LOSS) ? l : 0.0
    end
    total
end

# ── Stage 1: train for support discovery ──────────────────────────────────────
@printf "\nStage 1: %d epochs  LR=%.0e  λ_L1=%.0e  [%d threads]\n" N_STAGE1 LR_S1 λ_L1 Threads.nthreads()
flush(stdout)

for epoch in 1:N_STAGE1
    l_sg = train_step!(enc, opt, train_data, TRAIN_CTXS)

    # NODE-large
    _, lg_g = Zygote.withgradient(node_lg) do m
        total = 0.0
        for (i, c) in enumerate(TRAIN_CTXS)
            (ps, dts) = train_data[i]
            total += node_loss(m, ps, c, dts)
        end
        total
    end
    Flux.update!(node_lg_opt, node_lg, lg_g[1])

    # NODE-small
    _, sm_g = Zygote.withgradient(node_sm) do m
        total = 0.0
        for (i, c) in enumerate(TRAIN_CTXS)
            (ps, dts) = train_data[i]
            total += node_loss(m, ps, c, dts)
        end
        total
    end
    Flux.update!(node_sm_opt, node_sm, sm_g[1])

    if epoch % 100 == 0 || epoch == 1
        te_sg = eval_loss(enc, test_data, TEST_CTXS)
        te_lg = sum(node_loss(node_lg, ps, c, dts) for ((ps,dts),c) in zip(test_data, TEST_CTXS))
        te_sm = sum(node_loss(node_sm, ps, c, dts) for ((ps,dts),c) in zip(test_data, TEST_CTXS))
        @printf "epoch %4d  SG=%.5f  NODE-lg=%.5f  NODE-sm=%.5f\n" epoch te_sg te_lg te_sm
        avg_a = mean(abs.(vec(enc([c]))) for c in vcat(TRAIN_CTXS, TEST_CTXS))
        top5  = sortperm(avg_a; rev=true)[1:5]
        print("  top5 avg|α|:")
        for j in top5
            comp = j ≤ K_BASIS ? "x" : "y"
            jb   = j ≤ K_BASIS ? j : j - K_BASIS
            (ki,li) = BASIS_IDX[jb]
            marker = j ∈ TRUE_SUPPORT ? "✓" : "·"
            @printf "  %s dir%2d(T%dT%d·%s)=%.3f" marker j ki li comp avg_a[j]
        end
        println()
        flush(stdout)
    end
end

# ── Support extraction ─────────────────────────────────────────────────────────
println("\n━━━ Support extraction ━━━")
all_ctxs  = vcat(TRAIN_CTXS, TEST_CTXS)
avg_abs_α = mean(abs.(vec(enc([c]))) for c in all_ctxs)

@printf "\n%-4s  %-16s  %-8s  %-6s\n" "dir" "basis" "avg|α|" "active"
for j in 1:M_LIB
    comp     = j ≤ K_BASIS ? "x" : "y"
    jb       = j ≤ K_BASIS ? j : j - K_BASIS
    (ki, li) = BASIS_IDX[jb]
    label    = "T$(ki)T$(li)·$(comp)"
    active   = avg_abs_α[j] > λ_STLS
    marker   = active ? " ✓" : "  "
    in_true  = j ∈ TRUE_SUPPORT ? " [TRUE]" : ""
    @printf "%-4d  %-16s  %-8.4f%s%s\n" j label avg_abs_α[j] marker in_true
end

disc_support  = [j for j in 1:M_LIB if avg_abs_α[j] > λ_STLS]
support_mask  = zeros(M_LIB)
for j in disc_support;  support_mask[j] = 1.0;  end

println("\nDiscovered: ", disc_support)
println("True:       ", TRUE_SUPPORT)
correct = Set(disc_support) == Set(TRUE_SUPPORT)
println("Support recovery: ", correct ? "CORRECT ✓" : "INCORRECT ✗")
flush(stdout)

# ── Stage 2: masked refit on discovered support ────────────────────────────────
@printf "\nStage 2: masked refit  |support|=%d  LR=%.0e\n" length(disc_support) LR_S2
flush(stdout)

enc2 = deepcopy(enc)
opt2 = Flux.setup(Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipGrad(1.0),
    Adam(LR_S2),
), enc2)

function train_step_masked!(enc, opt, data, ctxs, mask)
    n      = length(data)
    alphas = [vec(enc([c])) .* mask for c in ctxs]

    sg_grads  = Vector{Vector{Float64}}(undef, n)
    sg_losses = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        (ps, dts) = data[i]
        α = alphas[i]
        A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
        g, l = shared_gradient(A, DIRS_LIB, ps, dts; m_krylov=30)
        if !isfinite(l) || l > MAX_LOSS
            sg_grads[i]  = zeros(M_LIB)
            sg_losses[i] = 0.0
        else
            sg_grads[i]  = g .* mask
            sg_losses[i] = l
        end
    end

    _, grads = Zygote.withgradient(enc) do m
        total = 0.0
        for i in 1:n
            total += dot(sg_grads[i], vec(m([ctxs[i]])) .* mask)
        end
        total
    end
    Flux.update!(opt, enc, grads[1])
    return sum(sg_losses)
end

for epoch in 1:N_STAGE2
    l = train_step_masked!(enc2, opt2, train_data, TRAIN_CTXS, support_mask)
    if epoch % 100 == 0 || epoch == 1
        te = eval_loss(enc2, test_data, TEST_CTXS)
        @printf "  epoch %3d  train=%.5f  test=%.5f\n" epoch l te
        flush(stdout)
    end
end

# ── Evaluation ─────────────────────────────────────────────────────────────────
println("\n━━━ Coefficient recovery ━━━")
println("\nRaw Chebyshev coefficients at active directions:")
@printf "%-6s  %10s %10s  %10s %10s  %10s %10s  %10s %10s\n" "c" "α[J10,x]_t" "α[J10,x]_p" "α[J01,x]_t" "α[J01,x]_p" "α[J01,y]_t" "α[J01,y]_p" "α[J10,y]_t" "α[J10,y]_p"

abs_errs = zeros(4)
rel_errs = zeros(4);  rel_n = zeros(4)
θ_abs_errs = zeros(2)
θ_rel_errs = zeros(2);  θ_rel_n = zeros(2)
for c in TEST_CTXS
    α_hat  = vec(enc2([c])) .* support_mask
    α_true = true_alpha(c)
    js = [J_10, J_01, K_BASIS+J_01, K_BASIS+J_10]
    @printf "%-6.2f  %10.4f %10.4f  %10.4f %10.4f  %10.4f %10.4f  %10.4f %10.4f\n" c α_true[js[1]] α_hat[js[1]] α_true[js[2]] α_hat[js[2]] α_true[js[3]] α_hat[js[3]] α_true[js[4]] α_hat[js[4]]
    for (k, j) in enumerate(js)
        ae = abs(α_hat[j] - α_true[j])
        abs_errs[k] += ae
        if abs(α_true[j]) > 0.1   # skip near-zero (zero-crossing contexts)
            rel_errs[k] += ae / abs(α_true[j])
            rel_n[k] += 1
        end
    end
    # Recover physical θ₁, θ₂ from Chebyshev coeffs (two estimates each, average)
    θ₁_hat  = (α_hat[J_10]  + α_hat[K_BASIS+J_01]) / 4
    θ₂_hat  = (α_hat[J_01]  - α_hat[K_BASIS+J_10]) / 4
    θ₁_true = -(0.5 + 0.3*sin(π*c))
    θ₂_true =  1.5*cos(π*c)
    θ_abs_errs[1] += abs(θ₁_hat - θ₁_true)
    θ_abs_errs[2] += abs(θ₂_hat - θ₂_true)
    if abs(θ₁_true) > 0.1
        θ_rel_errs[1] += abs(θ₁_hat - θ₁_true) / abs(θ₁_true);  θ_rel_n[1] += 1
    end
    if abs(θ₂_true) > 0.1
        θ_rel_errs[2] += abs(θ₂_hat - θ₂_true) / abs(θ₂_true);  θ_rel_n[2] += 1
    end
end
abs_errs  ./= length(TEST_CTXS)
rel_errs  ./= max.(rel_n, 1)
θ_abs_errs ./= length(TEST_CTXS)
θ_rel_errs ./= max.(θ_rel_n, 1)

@printf "\nMean abs err (Chebyshev coeffs):\n"
@printf "  α[J10,x]=%.4f  α[J01,x]=%.4f  α[J01,y]=%.4f  α[J10,y]=%.4f\n" abs_errs[1] abs_errs[2] abs_errs[3] abs_errs[4]
@printf "Mean rel err (non-zero contexts only):\n"
@printf "  α[J10,x]=%.1f%%  α[J01,x]=%.1f%%  α[J01,y]=%.1f%%  α[J10,y]=%.1f%%\n" 100rel_errs[1] 100rel_errs[2] 100rel_errs[3] 100rel_errs[4]
@printf "\nPhysical params (mean abs err / mean rel err):\n"
@printf "  θ₁(c) = -(0.5+0.3sin(πc)):  abs=%.4f  rel=%.1f%%\n" θ_abs_errs[1] 100θ_rel_errs[1]
@printf "  θ₂(c) =  1.5cos(πc):        abs=%.4f  rel=%.1f%%\n" θ_abs_errs[2] 100θ_rel_errs[2]

# ── Extrapolation comparison ───────────────────────────────────────────────────
println("\n━━━ Extrapolation (t=$(DT_EXTRA), $(DT_EXTRA/maximum(DTS_TRAIN))× training) ━━━")

function sg_predict_extra(enc2, p0, c, mask)
    α = vec(enc2([c])) .* mask
    A = A0_LIB + sum(α[j]*DIRS_LIB[j] for j in 1:M_LIB)
    expv(DT_EXTRA, A, p0; m=50)
end

traj_loss(p_pred, p_true) = 0.5*sum(abs2, p_pred .- p_true)
pos_viol(p)  = sum(p .< -1e-6) / length(p)
mass_err(p)  = abs(sum(p) - 1.0)

extrap_sg = Float64[];  extrap_lg = Float64[];  extrap_sm = Float64[]
pv_sg = Float64[];   pv_lg = Float64[];   pv_sm = Float64[]
me_sg = Float64[];   me_lg = Float64[];   me_sm = Float64[]

for (i, c) in enumerate(TEST_CTXS)
    ps_ex, dts_ex = extra_data[i]
    p0_ex = ps_ex[1];  p_true_ex = ps_ex[2]

    p_sg  = sg_predict_extra(enc2, p0_ex, c, support_mask)
    p_lgv = node_predict(node_lg, p0_ex, c, dts_ex; n_sub=N_SUB*4)[2]
    p_smv = node_predict(node_sm, p0_ex, c, dts_ex; n_sub=N_SUB*4)[2]

    push!(extrap_sg, traj_loss(p_sg,  p_true_ex))
    push!(extrap_lg, traj_loss(p_lgv, p_true_ex))
    push!(extrap_sm, traj_loss(p_smv, p_true_ex))
    push!(pv_sg, pos_viol(p_sg));   push!(me_sg, mass_err(p_sg))
    push!(pv_lg, pos_viol(p_lgv));  push!(me_lg, mass_err(p_lgv))
    push!(pv_sm, pos_viol(p_smv));  push!(me_sm, mass_err(p_smv))
end

@printf "%-26s  %10s  %12s  %12s\n" "Model" "Extrap loss" "Pos.viol.%" "Mass err"
@printf "%-26s  %10.5f  %12.4f  %12.6f\n" "Semigroup (~$(count_params(enc))p)" mean(extrap_sg) 100mean(pv_sg) mean(me_sg)
@printf "%-26s  %10.5f  %12.4f  %12.6f\n" "NODE-small (~$(count_params(node_sm))p)" mean(extrap_sm) 100mean(pv_sm) mean(me_sm)
@printf "%-26s  %10.5f  %12.4f  %12.6f\n" "NODE-large (~$(count_params(node_lg))p)" mean(extrap_lg) 100mean(pv_lg) mean(me_lg)

# ── Figure ─────────────────────────────────────────────────────────────────────
println("\nGenerating figure…")

# Demo density at extrapolation horizon, c=0.5
c_demo    = 0.5
p0_demo   = gaussian_ic()
ps_demo   = extra_data[findfirst(==(c_demo), TEST_CTXS)]
p_true_d  = ps_demo[1][2]
p_sg_d    = sg_predict_extra(enc2, p0_demo, c_demo, support_mask)
p_lg_d    = node_predict(node_lg, p0_demo, c_demo, [DT_EXTRA]; n_sub=N_SUB*4)[2]
to2d(p)   = reshape(p, NX, NY)

# Dense context range for smooth curves
θ_true_c(c) = (-(0.5+0.3sin(π*c)), 1.5cos(π*c))
ctx_dense = collect(range(0.0, 1.0; length=200))
θ1_true_d = [θ_true_c(c)[1] for c in ctx_dense]
θ2_true_d = [θ_true_c(c)[2] for c in ctx_dense]
θ1_sg_d   = [(vec(enc2([c])) .* support_mask)[J_10] / 2 for c in ctx_dense]
θ2_sg_d   = [(vec(enc2([c])) .* support_mask)[J_01] / 2 for c in ctx_dense]


# avg|α| for support panel
all_ctxs_fig = vcat(TRAIN_CTXS, TEST_CTXS)
avg_abs_fig  = mean(abs.(vec(enc([c]))) for c in all_ctxs_fig)

fig = Figure(size=(1200, 330), fontsize=11)

# Panel A: support discovery
ax_a = Axis(fig[1,1];
    xlabel="direction j", ylabel="avg |α(c)|",
    title="(A) Support discovery (M=30)",
    xticks=(TRUE_SUPPORT, ["T₁T₀·x", "T₀T₁·x", "T₀T₁·y", "T₁T₀·y"]))
bar_colors = [j ∈ TRUE_SUPPORT ? :steelblue : :lightgray for j in 1:M_LIB]
barplot!(ax_a, 1:M_LIB, avg_abs_fig; color=bar_colors)
hlines!(ax_a, [λ_STLS]; color=:red, linestyle=:dash, linewidth=1.5, label="threshold")
axislegend(ax_a; position=:rt, labelsize=9, framevisible=false)

# Panel B: learned parameter curves
ax_b = Axis(fig[1,2]; xlabel="context c", ylabel="θ(c)",
            title="(B) Recovered parameters", xticks=0:0.25:1)
lines!(ax_b, ctx_dense, θ1_true_d; color=:steelblue, linewidth=2, label=L"\theta_1^\star")
lines!(ax_b, ctx_dense, θ2_true_d; color=:firebrick, linewidth=2, label=L"\theta_2^\star")
lines!(ax_b, ctx_dense, θ1_sg_d; color=:steelblue, linewidth=1.5, linestyle=:dash)
lines!(ax_b, ctx_dense, θ2_sg_d; color=:firebrick, linewidth=1.5, linestyle=:dash, label="SG (dashed)")
scatter!(ax_b, TEST_CTXS, [vec(enc2([c]))[J_10]/2 for c in TEST_CTXS];
         color=:steelblue, marker=:diamond, markersize=8)
scatter!(ax_b, TEST_CTXS, [vec(enc2([c]))[J_01]/2 for c in TEST_CTXS];
         color=:firebrick, marker=:diamond, markersize=8, label="test ◇")
axislegend(ax_b; position=:rb, labelsize=9, framevisible=false)

# Panels C-E: 2D density at extrapolation time
clim = maximum(p_true_d) * 1.05
for (col, (p2d, ttl)) in enumerate([
        (p_true_d, "True"),
        (p_sg_d,   "Semigroup\n(~$(count_params(enc))p)"),
        (p_lg_d,   "NODE-large\n(~$(count_params(node_lg))p)")])
    ax = Axis(fig[1, col+2]; xlabel="x", ylabel=(col==1 ? "y" : ""),
              title="($(('B'+col))) $ttl  t=$(DT_EXTRA)", aspect=1)
    hm = heatmap!(ax, XS, YS, to2d(p2d); colormap=:viridis, colorrange=(0.0, clim))
    col == 3 && Colorbar(fig[1,6], hm; label="p(x,y)", width=12)
end

# Panel F: extrapolation loss bar chart
ax_f = Axis(fig[1,7];
    xlabel="test context c", ylabel="extrapolation loss",
    title="(F) t=$(DT_EXTRA) ($(DT_EXTRA/maximum(DTS_TRAIN))× training)",
    yscale=log10, xticks=(1:5, [@sprintf("%.1f",c) for c in TEST_CTXS]))
xs_b = 1:length(TEST_CTXS);  w = 0.25
barplot!(ax_f, xs_b .- w, extrap_sg; width=w, color=:steelblue, label="Semigroup")
barplot!(ax_f, xs_b,      extrap_sm; width=w, color=:orange,    label="NODE-sm")
barplot!(ax_f, xs_b .+ w, extrap_lg; width=w, color=:tomato,    label="NODE-lg")
axislegend(ax_f; position=:lt, labelsize=9, framevisible=false)

# ── Save results for figure generation ────────────────────────────────────────
using CSV, DataFrames

# Support discovery: avg|α| per direction
support_df = DataFrame(
    dir           = 1:M_LIB,
    avg_abs_alpha = avg_abs_fig,
    is_true       = [j ∈ TRUE_SUPPORT for j in 1:M_LIB],
)
CSV.write("paper/figures/sparse2d_support.csv", support_df)

# Extrapolation losses per test context
extrap_df = DataFrame(
    context  = TEST_CTXS,
    sg_loss  = extrap_sg,
    node_sm  = extrap_sm,
    node_lg  = extrap_lg,
)
CSV.write("paper/figures/sparse2d_extrap.csv", extrap_df)

# Coefficient recovery: dense context grid for smooth curves
ctx_save = collect(range(0.0, 1.0; length=100))
θ1_true_s = [θ_true_c(c)[1] for c in ctx_save]
θ2_true_s = [θ_true_c(c)[2] for c in ctx_save]
θ1_sg_s   = [(vec(enc2([c])) .* support_mask)[J_10] / 2 for c in ctx_save]
θ2_sg_s   = [(vec(enc2([c])) .* support_mask)[J_01] / 2 for c in ctx_save]
coeff_df = DataFrame(
    context  = ctx_save,
    theta1_true = θ1_true_s,
    theta2_true = θ2_true_s,
    theta1_sg   = θ1_sg_s,
    theta2_sg   = θ2_sg_s,
)
CSV.write("paper/figures/sparse2d_coeffs.csv", coeff_df)
println("Saved sparse2d_support.csv, sparse2d_extrap.csv, sparse2d_coeffs.csv")

mkpath("paper/figures")
save("paper/figures/fig_sparse2d_cheby.pdf", fig)
save("paper/figures/fig_sparse2d_cheby.png", fig)
println("Saved paper/figures/fig_sparse2d_cheby.{pdf,png}")
