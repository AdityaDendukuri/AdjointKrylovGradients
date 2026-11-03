"""
Stuart-Landau oscillator: sparse nonlinear FPE identification.

System:  dX = b(X; μ)dt + √(2D) dW
  b_x(x,y; μ) = μx - ωy - x³ - xy²       (contraction + rotation + cubic saturation)
  b_y(x,y; μ) = ωx + μy - x²y - y³
  D = 0.15,  ω = 1.0

Dynamics:
  μ < 0 → stable spiral to origin
  μ > 0 → limit cycle at radius √μ, frequency ω/(2π)
  μ = 0 → Hopf bifurcation

Library: monomials degree ≤ 3, two components → M = 20 directions.
  {1, x, y, x², xy, y², x³, x²y, xy², y³} × {b_x, b_y}

True support: 8/20 active directions
  b_x: x (coeff μ, context-dep), y (coeff -ω), x³ (coeff -1), xy² (coeff -1)
  b_y: x (coeff +ω), y (coeff μ, context-dep), x²y (coeff -1), y³ (coeff -1)

Key comparison:
  SemigroupLayer-MSE at Δt=1.0  → finds all 8 active directions (exact propagator)
  RK4-MSE at Δt=1.0, 1 step     → deliberately under-resolved stiffness diagnostic
  RK4-MSE at Δt=1.0, 32 steps   → stability-resolved explicit baseline
  Euler-MSE at Δt=1.0           → 1st-order Taylor; weak baseline
  SemigroupLayer-MSE at Δt=0.1  → reference (all methods work at fine Δt)

The one-step RK4 result is not a fair stiff-solver comparison because
|λ_fast|Δt≈79 is far outside its stability region. The stability-resolved
variant uses substeps small enough that |λ_fast|h is inside that region.
"""

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Flux, Zygote, ChainRulesCore
using ExponentialUtilities: expv
using CSV, DataFrames

include("../src/shared_krylov.jl")
include("../src/fpe2d.jl")

# ── Grid ───────────────────────────────────────────────────────────────────────
const NX_SL = 30;  const NY_SL = 30
const XS_SL = collect(range(-2.0, 2.0; length=NX_SL))
const YS_SL = collect(range(-2.0, 2.0; length=NY_SL))
const D_SL  = 0.15
const ω_SL  = 1.0
const N_SL  = NX_SL * NY_SL   # 900

# Column-major linearization (matches fpe2d.jl)
lin_sl(i, j) = i + (j-1)*NX_SL
function snap_sl(x, y)
    ix = clamp(round(Int, (x-XS_SL[1])/(XS_SL[2]-XS_SL[1]))+1, 1, NX_SL)
    iy = clamp(round(Int, (y-YS_SL[1])/(YS_SL[2]-YS_SL[1]))+1, 1, NY_SL)
    lin_sl(ix, iy)
end

# ── Monomial library (degree ≤ 3) ─────────────────────────────────────────────
# {1, x, y, x², xy, y², x³, x²y, xy², y³} → K=10 per component, M=20 total
const MONO_LABELS = ["1","x","y","x²","xy","y²","x³","x²y","xy²","y³"]
const K_SL = 10;  const M_SL = 2 * K_SL

bx_fns_sl = [
    (x,y) -> 1.0,   (x,y) -> x,    (x,y) -> y,
    (x,y) -> x^2,   (x,y) -> x*y,  (x,y) -> y^2,
    (x,y) -> x^3,   (x,y) -> x^2*y,(x,y) -> x*y^2,(x,y) -> y^3,
]
by_fns_sl = fill((x,y)->0.0, M_SL)
bx_fns_sl_full = [bx_fns_sl; fill((x,y)->0.0, K_SL)]
by_fns_sl_full = [fill((x,y)->0.0, K_SL); bx_fns_sl]

println("Building monomial FPE library  (degree≤3, K=$K_SL, M=$M_SL)…")
const A0_SL, DIRS_SL = fpe_library_2d(XS_SL, YS_SL, bx_fns_sl_full, by_fns_sl_full; D=D_SL)
const SL_SL = SemigroupLayer(A0_SL, DIRS_SL; m_krylov=40)

# Monomial index lookup
mono_idx(label) = findfirst(==(label), MONO_LABELS)
const M_x  = mono_idx("x");   const M_y  = mono_idx("y")
const M_x3 = mono_idx("x³");  const M_xy2 = mono_idx("xy²")
const M_x2y= mono_idx("x²y"); const M_y3  = mono_idx("y³")

# True support: b_x has x,y,x³,xy²; b_y has x,y,x²y,y³
const TRUE_SUPPORT_SL = sort([M_x, M_y, M_x3, M_xy2,
                               K_SL+M_x, K_SL+M_y, K_SL+M_x2y, K_SL+M_y3])
println("  True support: $TRUE_SUPPORT_SL ($(length(TRUE_SUPPORT_SL))/$(M_SL) active)")

# ── True parameter map ─────────────────────────────────────────────────────────
# b_x = μx - ωy - x³ - xy²  →  coeff of x=μ, y=-ω, x³=-1, xy²=-1
# b_y = ωx + μy - x²y - y³  →  coeff of x=+ω, y=μ, x²y=-1, y³=-1
function true_alpha_sl(c; mu=Float64(c), omega=ω_SL)
    α = zeros(M_SL)
    α[M_x]         =  mu     # b_x · x   (context-dep)
    α[M_y]         = -omega  # b_x · y   (context-indep)
    α[M_x3]        = -1.0    # b_x · x³  (context-indep)
    α[M_xy2]       = -1.0    # b_x · xy² (context-indep)
    α[K_SL+M_x]    = +omega  # b_y · x   (context-indep)
    α[K_SL+M_y]    =  mu     # b_y · y   (context-dep)
    α[K_SL+M_x2y]  = -1.0    # b_y · x²y (context-indep)
    α[K_SL+M_y3]   = -1.0    # b_y · y³  (context-indep)
    α
end

# Sanity: column sums of true generator should be ≈ 0 (Markov)
let c = 0.5
    A = A0_SL + sum(true_alpha_sl(c)[j]*DIRS_SL[j] for j in 1:M_SL)
    @printf "Generator sanity (c=0.5): col-sum err=%.2e\n" maximum(abs.(vec(sum(A; dims=1))))
end

# ── Data: exact FPE transitions from START_CELLS ──────────────────────────────
# Sample from exp(A_true·dt)·e_i — exactly model-consistent, no discretisation mismatch.
const START_XS_SL = [-1.5, -0.75, 0.0, 0.75, 1.5]
const START_YS_SL = [-1.5, -0.75, 0.0, 0.75, 1.5]
const START_CELLS_SL = [snap_sl(x,y) for x in START_XS_SL for y in START_YS_SL]

function gen_data_sl(c, dt; rng=Random.default_rng())
    A = A0_SL + sum(true_alpha_sl(c)[j]*DIRS_SL[j] for j in 1:M_SL)
    # Off-centre Gaussian — breaks radial symmetry so rotation is identifiable.
    # A symmetric IC centred at origin leaves (-ωy∂_x + ωx∂_y)p = 0: rotation invisible.
    p0 = let σ=0.3, cx=0.8, cy=0.3
        p = [exp(-((XS_SL[i]-cx)^2+(YS_SL[j]-cy)^2)/(2σ^2))
             for i in 1:NX_SL, j in 1:NY_SL]
        p = vec(p); p ./= sum(p); p
    end
    ps = [p0]
    for _ in 1:3
        p = expv(dt, A, ps[end]; m=50)
        p = max.(real.(p), 0.0); p ./= sum(p)
        push!(ps, p)
    end
    ps, fill(dt, 3)
end

# ── Contexts ───────────────────────────────────────────────────────────────────
const TRAIN_CTXS_SL = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
const TEST_CTXS_SL  = [-0.2, 0.2, 0.6, 1.0]

const DT_COARSE = 1.0   # coarse: first-order Euler fails (~60° rotation per step)
const DT_FINE   = 0.1   # fine: all methods work (reference)

Random.seed!(42)
println("\nGenerating exact FPE data…")
train_c = Dict(c => gen_data_sl(c, DT_COARSE) for c in TRAIN_CTXS_SL)
test_c  = Dict(c => gen_data_sl(c, DT_COARSE) for c in TEST_CTXS_SL)
train_f = Dict(c => gen_data_sl(c, DT_FINE)   for c in TRAIN_CTXS_SL)
test_f  = Dict(c => gen_data_sl(c, DT_FINE)   for c in TEST_CTXS_SL)

@printf "  Coarse Δt=%.1f  Fine Δt=%.1f  n_train=%d  n_test=%d\n" DT_COARSE DT_FINE length(TRAIN_CTXS_SL) length(TEST_CTXS_SL)
println("  Contexts span Hopf bifurcation: μ ∈ [$(minimum(TRAIN_CTXS_SL)), $(maximum(TRAIN_CTXS_SL))]")

# ── Losses ─────────────────────────────────────────────────────────────────────
function sg_mse_loss(model, data_dict, ctxs)
    sum(begin
        ps, dts = data_dict[c]
        θ = vec(model([c]))
        sum(0.5*sum(abs2, ps[t+1] - SL_SL(ps[t], θ, dts[t])) for t in eachindex(dts))
    end for c in ctxs)
end

# First-order Euler: replace exp(A·dt) with (I + A·dt)
function euler_mse_loss(model, data_dict, ctxs)
    sum(begin
        ps, dts = data_dict[c]
        θ = vec(model([c]))
        A = A0_SL + sum(θ[j]*DIRS_SL[j] for j in 1:M_SL)
        sum(0.5*sum(abs2, ps[t+1] - (ps[t] + A*(ps[t]*dts[t]))) for t in eachindex(dts))
    end for c in ctxs)
end

# RK4 with configurable substeps. n_sub=1 reproduces the submitted coarse
# Taylor baseline; n_sub≈32 resolves the fastest diffusion mode
# (|λ_fast|≈79, so |λ_fast|*(1/32)≈2.47 is inside RK4's stability interval).
function rk4_mse_loss(model, data_dict, ctxs; n_sub=1)
    sum(begin
        ps, dts = data_dict[c]
        θ = vec(model([c]))
        A = A0_SL + sum(θ[j]*DIRS_SL[j] for j in 1:M_SL)
        sum(begin
            p, dt = ps[t], dts[t]
            h = dt / n_sub
            p_rk4 = p
            for _ in 1:n_sub
                k1 = A*p_rk4
                k2 = A*(p_rk4 .+ 0.5h.*k1)
                k3 = A*(p_rk4 .+ 0.5h.*k2)
                k4 = A*(p_rk4 .+    h.*k3)
                p_rk4 = p_rk4 .+ (h/6).*(k1 .+ 2 .*k2 .+ 2 .*k3 .+ k4)
            end
            0.5*sum(abs2, ps[t+1] - p_rk4)
        end for t in eachindex(dts))
    end for c in ctxs)
end

# ── Encoder ────────────────────────────────────────────────────────────────────
make_enc_sl() = Chain(Dense(1=>32, tanh), Dense(32=>32, tanh),
                      Dense(32=>M_SL; init=Flux.zeros32)) |> f64
count_p(m) = sum(length(p) for p in Flux.trainables(m))

# ── Training ───────────────────────────────────────────────────────────────────
const N_S1_SL = parse(Int,     get(ENV, "S1", "800"))
const N_S2_SL = parse(Int,     get(ENV, "S2", "600"))
const LR1_SL  = parse(Float64, get(ENV, "LR1","3e-3"))
const LR2_SL  = parse(Float64, get(ENV, "LR2","1e-3"))
const λ_SL    = parse(Float64, get(ENV, "STLS","0.4"))
const RK4_STABLE_SUBSTEPS = parse(Int, get(ENV, "RK4_STABLE_SUBSTEPS", "32"))
const RK4_STABLE_EPOCHS = parse(Int, get(ENV, "RK4_STABLE_EPOCHS", string(N_S1_SL)))
const MAX_L   = 1e3

function train_stage1!(enc, opt, loss_fn, train_dict, test_dict, label; epochs=N_S1_SL)
    @printf "\nStage 1 [%s | %d params]: %d epochs  LR=%.0e\n" label count_p(enc) epochs LR1_SL
    @printf "%-6s  %-14s  %-14s  %-20s\n" "epoch" "train" "test" "top-6 avg|α|"
    for epoch in 1:epochs
        l, g = Zygote.withgradient(enc) do m
            loss_fn(m, train_dict, TRAIN_CTXS_SL)
        end
        (isfinite(l) && l < MAX_L) && Flux.update!(opt, enc, g[1])
        if epoch % 100 == 0 || epoch == 1
            tr = loss_fn(enc, train_dict, TRAIN_CTXS_SL)
            te = loss_fn(enc, test_dict,  TEST_CTXS_SL)
            # Diagnostics must use training contexts only. Test contexts are
            # reserved for the final held-out evaluation.
            avg = mean(abs.(vec(enc([c]))) for c in TRAIN_CTXS_SL)
            top6 = sortperm(avg; rev=true)[1:6]
            marks = [j ∈ TRUE_SUPPORT_SL ? "✓$j" : "·$j" for j in top6]
            @printf "%-6d  %-14.5f  %-14.5f  %s\n" epoch tr te join(marks," ")
            flush(stdout)
        end
    end
end

function extract_support(enc, label)
    # Model selection is based exclusively on training contexts.
    avg = mean(abs.(vec(enc([c]))) for c in TRAIN_CTXS_SL)
    disc = sort([j for j in 1:M_SL if avg[j] > λ_SL])
    correct = Set(disc) == Set(TRUE_SUPPORT_SL)
    @printf "[%s] support: %s  (%s)\n" label repr(disc) (correct ? "CORRECT ✓" : "INCORRECT ✗")
    avg, disc
end

function stage2!(enc_base, support_mask, data_dict, label)
    enc2 = deepcopy(enc_base)
    opt2 = Flux.setup(Flux.Optimisers.OptimiserChain(ClipGrad(1.0), Adam(LR2_SL)), enc2)
    @printf "\nStage 2 [%s masked, |supp|=%d]: %d epochs\n" label sum(support_mask.>0) N_S2_SL
    for epoch in 1:N_S2_SL
        l, g = Zygote.withgradient(enc2) do m
            sum(begin
                ps, dts = data_dict[c]
                θ = vec(m([c])) .* support_mask
                sum(0.5*sum(abs2, ps[t+1] - SL_SL(ps[t], θ, dts[t])) for t in eachindex(dts))
            end for c in TRAIN_CTXS_SL)
        end
        (isfinite(l) && l < MAX_L) && Flux.update!(opt2, enc2, g[1])
        if epoch % 100 == 0 || epoch == 1
            tr = sum(begin ps,dts=data_dict[c]; θ=vec(enc2([c])).*support_mask
                    sum(0.5*sum(abs2, ps[t+1]-SL_SL(ps[t],θ,dts[t])) for t in eachindex(dts)) end
                    for c in TRAIN_CTXS_SL)
            @printf "  epoch %3d  train=%.5f\n" epoch tr; flush(stdout)
        end
    end
    enc2
end

# ── Run three variants ─────────────────────────────────────────────────────────
make_opt(enc) = Flux.setup(Flux.Optimisers.OptimiserChain(ClipGrad(1.0), Adam(LR1_SL)), enc)
function fresh_enc_sl(seed=42)
    Random.seed!(seed)
    make_enc_sl()
end

const RUN_PRIMARY_EXPERIMENT = get(ENV, "RUN_PRIMARY_EXPERIMENT", "1") == "1"

if RUN_PRIMARY_EXPERIMENT

# 1. SemigroupLayer-MSE at coarse Δt (primary result)
enc_sg_c = fresh_enc_sl();  opt_sg_c = make_opt(enc_sg_c)
time_sg_c = @elapsed train_stage1!(enc_sg_c, opt_sg_c, sg_mse_loss, train_c, test_c,
                                   "SG-MSE Δt=$(DT_COARSE)")

# 2. First-order Euler at coarse Δt
enc_euler_c = fresh_enc_sl();  opt_euler_c = make_opt(enc_euler_c)
time_euler_c = @elapsed train_stage1!(enc_euler_c, opt_euler_c, euler_mse_loss,
                                      train_c, test_c, "Euler Δt=$(DT_COARSE)")

# 3. One deliberately under-resolved RK4 step at coarse Δt (submitted diagnostic)
enc_rk4_c = fresh_enc_sl();  opt_rk4_c = make_opt(enc_rk4_c)
time_rk4_c = @elapsed train_stage1!(enc_rk4_c, opt_rk4_c, rk4_mse_loss,
                                    train_c, test_c, "RK4  Δt=$(DT_COARSE)")

# 4. Stability-resolved RK4 at the same observation interval
rk4_stable_loss(model, data_dict, ctxs) =
    rk4_mse_loss(model, data_dict, ctxs; n_sub=RK4_STABLE_SUBSTEPS)
enc_rk4_stable = fresh_enc_sl();  opt_rk4_stable = make_opt(enc_rk4_stable)
time_rk4_stable = @elapsed train_stage1!(
    enc_rk4_stable, opt_rk4_stable, rk4_stable_loss, train_c, test_c,
    "RK4×$(RK4_STABLE_SUBSTEPS) Δt=$(DT_COARSE)"; epochs=RK4_STABLE_EPOCHS)

# 5. SemigroupLayer-MSE at fine Δt (reference — all methods should work here)
enc_sg_f = fresh_enc_sl();  opt_sg_f = make_opt(enc_sg_f)
time_sg_f = @elapsed train_stage1!(enc_sg_f, opt_sg_f, sg_mse_loss, train_f, test_f,
                                   "SG-MSE Δt=$(DT_FINE)")

# ── Support extraction ─────────────────────────────────────────────────────────
println("\n━━━ Support extraction ━━━")
avg_sg_c,    disc_sg_c    = extract_support(enc_sg_c,    "SG    Δt=$(DT_COARSE)")
avg_euler_c, disc_euler_c = extract_support(enc_euler_c, "Euler Δt=$(DT_COARSE)")
avg_rk4_c,   disc_rk4_c   = extract_support(enc_rk4_c,   "RK4   Δt=$(DT_COARSE)")
avg_rk4_stable, disc_rk4_stable =
    extract_support(enc_rk4_stable, "RK4×$(RK4_STABLE_SUBSTEPS) Δt=$(DT_COARSE)")
avg_sg_f,    disc_sg_f    = extract_support(enc_sg_f,    "SG    Δt=$(DT_FINE)  ")
println("True support:   $TRUE_SUPPORT_SL")

# Stage 2 for the SG coarse model (if support is correct)
mask_sg_c = zeros(M_SL)
for j in disc_sg_c; mask_sg_c[j] = 1.0; end
enc2_sg_c = stage2!(enc_sg_c, mask_sg_c, train_c, "SG Δt=$(DT_COARSE)")

# ── Evaluation ─────────────────────────────────────────────────────────────────
println("\n━━━ Rate recovery (SG coarse, test contexts) ━━━")
@printf "%-5s  %-8s %-8s  %-8s %-8s  %-8s\n" "μ" "μ_true" "μ_pred" "ω_true" "ω_pred" "cubic_pred"

rows = NamedTuple[]
for c in sort(vcat(TRAIN_CTXS_SL, TEST_CTXS_SL))
    α_hat  = vec(enc2_sg_c([c])) .* mask_sg_c
    α_true = true_alpha_sl(c)
    μ_hat  = (α_hat[M_x] + α_hat[K_SL+M_y]) / 2   # average of two μ estimates
    ω_hat  = (-α_hat[M_y] + α_hat[K_SL+M_x]) / 2
    cub    = mean(abs, [α_hat[M_x3], α_hat[M_xy2], α_hat[K_SL+M_x2y], α_hat[K_SL+M_y3]])
    marker = c ∈ TEST_CTXS_SL ? "←" : "  "
    @printf "%-5.2f%s %-8.3f %-8.3f  %-8.3f %-8.3f  %-8.3f\n" c marker c μ_hat ω_SL ω_hat cub
    push!(rows, (mu_ctx=c, mu_true=Float64(c), mu_pred=μ_hat, omega_true=ω_SL, omega_pred=ω_hat,
                 cubic_pred=cub, is_test=(c ∈ TEST_CTXS_SL)))
end

test_rows = [r for r in rows if r.is_test]
valid = [r for r in test_rows if abs(r.mu_true) > 0.05]
mu_err = mean(abs(r.mu_pred - r.mu_true)/abs(r.mu_true) for r in valid) * 100
om_err = mean(abs(r.omega_pred - r.omega_true)/r.omega_true for r in test_rows) * 100
println("\n━━━ Summary ━━━")
@printf "SG Δt=%.1f  train-selected support=%s  held-out μ_err=%.1f%%  held-out ω_err=%.1f%%\n" DT_COARSE (disc_sg_c==sort(TRUE_SUPPORT_SL) ? "✓" : "✗") mu_err om_err

# ── Save results ───────────────────────────────────────────────────────────────
# Support weights for figure
all_avgs = DataFrame(
    dir       = 1:M_SL,
    component = [(j <= K_SL ? "bx" : "by") for j in 1:M_SL],
    monomial  = [MONO_LABELS[(j-1) % K_SL + 1] for j in 1:M_SL],
    degree    = [count(c->c in "23", MONO_LABELS[(j-1)%K_SL+1]) for j in 1:M_SL],
    avg_sg_c  = avg_sg_c,
    avg_euler_c = avg_euler_c,
    avg_rk4_c = avg_rk4_c,
    avg_rk4_stable = avg_rk4_stable,
    avg_sg_f  = avg_sg_f,
    is_true   = [j ∈ TRUE_SUPPORT_SL for j in 1:M_SL],
)
CSV.write("paper/figures/sl_support.csv", all_avgs)

# Rate recovery
CSV.write("paper/figures/sl_recovery.csv", DataFrame(rows))

# Dense context range for smooth curves
ctx_dense = collect(range(-0.4, 1.3; length=120))
coeff_df = DataFrame(
    mu          = ctx_dense,
    mu_pred     = [(vec(enc2_sg_c([c])) .* mask_sg_c)[M_x]   for c in ctx_dense],
    mu_true     = collect(ctx_dense),
    omega_pred  = [(-vec(enc2_sg_c([c])).*mask_sg_c)[M_y]     for c in ctx_dense],
    omega_true  = fill(ω_SL, length(ctx_dense)),
)
CSV.write("paper/figures/sl_coeffs.csv", coeff_df)

# Density evolution for figure: IC placed ON the limit cycle at μ=1.1 (r=√1.1).
# At Δt=1.0, ω=1 rad/s → blob travels ~57° per step. SG (matrix exponential)
# tracks the full orbit; EM (Euler step I+A·Δt) cannot rotate probability mass
# at coarse Δt and ends up at the wrong angular position by t=Δt.
let c_demo = 1.1, dt = DT_COARSE
    α_true = true_alpha_sl(c_demo)
    α_sg   = vec(enc2_sg_c([c_demo])) .* mask_sg_c
    α_euler = vec(enc_euler_c([c_demo]))

    A_true = A0_SL + sum(α_true[j]*DIRS_SL[j] for j in 1:M_SL)
    A_sg   = A0_SL + sum(α_sg[j]*DIRS_SL[j]   for j in 1:M_SL)
    A_euler = A0_SL + sum(α_euler[j]*DIRS_SL[j] for j in 1:M_SL)

    # IC placed ON the limit cycle — blob orbits at ω=1 rad/s, reaching angle ≈57°
    # per Δt=1.0 step. SG (matrix exponential) tracks the orbit exactly; EM (Euler
    # step I+A·Δt) cannot rotate the probability mass and ends up at the wrong angle.
    # The step=0 panel (saved below) shows the IC; clim is calibrated to step>0 so
    # the IC panel deliberately saturates — a bright dot that visually marks the start.
    p0 = let σ=0.18, cx=sqrt(1.1), cy=0.0
        p = [exp(-((XS_SL[i]-cx)^2+(YS_SL[j]-cy)^2)/(2σ^2))
             for i in 1:NX_SL, j in 1:NY_SL]
        p = vec(p); p ./= sum(p); p
    end

    # Propagate true, SG, and EM for 3 steps
    function propagate(A, p0, n_steps, dt)
        ps = [p0]
        for _ in 1:n_steps
            p = expv(dt, A, ps[end]; m=50)
            p = max.(real.(p), 0.0); p ./= sum(p)
            push!(ps, p)
        end
        ps
    end
    # EM propagation: p_{t+1} ≈ p_t + A·Δt·p_t  (Euler on density)
    function propagate_em(A, p0, n_steps, dt)
        ps = [p0]
        for _ in 1:n_steps
            p = ps[end] + A * ps[end] * dt
            p = max.(p, 0.0); p ./= sum(p)
            push!(ps, p)
        end
        ps
    end

    ps_true  = propagate(A_true,  p0, 3, dt)
    ps_sg    = propagate(A_sg,    p0, 3, dt)
    ps_euler = propagate_em(A_euler, p0, 3, dt)   # first-order Euler

    # Flatten: one row per (x, y, step, model)
    dens_rows = NamedTuple[]
    for (model_tag, ps) in [("true", ps_true), ("sg", ps_sg), ("em", ps_euler)]
        for (step, p) in enumerate(ps)
            pmat = reshape(p, NX_SL, NY_SL)
            for (i, x) in enumerate(XS_SL), (j, y) in enumerate(YS_SL)
                push!(dens_rows, (model=model_tag, step=step-1, x=x, y=y, prob=pmat[i,j]))
            end
        end
    end
    CSV.write("paper/figures/sl_density.csv", DataFrame(dens_rows))
    println("Saved sl_density.csv  ($(length(dens_rows)) rows)")
end

# Loss history: save final losses for comparison table
loss_df = DataFrame(
    method  = ["SG Δt=$(DT_COARSE)", "Euler Δt=$(DT_COARSE)",
               "RK4×1 Δt=$(DT_COARSE)", "RK4×$(RK4_STABLE_SUBSTEPS) Δt=$(DT_COARSE)",
               "SG Δt=$(DT_FINE)"],
    support = [repr(disc_sg_c), repr(disc_euler_c), repr(disc_rk4_c),
               repr(disc_rk4_stable), repr(disc_sg_f)],
    correct = [Set(disc_sg_c)==Set(TRUE_SUPPORT_SL),
               Set(disc_euler_c)==Set(TRUE_SUPPORT_SL),
               Set(disc_rk4_c)==Set(TRUE_SUPPORT_SL),
               Set(disc_rk4_stable)==Set(TRUE_SUPPORT_SL),
               Set(disc_sg_f)==Set(TRUE_SUPPORT_SL)],
    epochs = [N_S1_SL, N_S1_SL, N_S1_SL, RK4_STABLE_EPOCHS, N_S1_SL],
    train_seconds = [time_sg_c, time_euler_c, time_rk4_c, time_rk4_stable, time_sg_f],
)
CSV.write("paper/figures/sl_comparison.csv", loss_df)
println("\nSaved sl_support.csv, sl_recovery.csv, sl_coeffs.csv, sl_density.csv, sl_comparison.csv")

end # RUN_PRIMARY_EXPERIMENT

# ── Same-system black-box Neural ODE baselines ─────────────────────────────────
#
# The submitted NODE architecture was first exercised in sparse2d_cheby.jl.
# Rebuttal results must instead train and evaluate it on the Stuart--Landau
# data above. Set RUN_NODE_BASELINES=1 to run this section. The default of
# 32 RK4 substeps per unit time resolves the fastest discretized diffusion mode
# much more fairly than the submitted 4-substep setting.

const RUN_NODE_BASELINES = get(ENV, "RUN_NODE_BASELINES", "0") == "1"

function make_node_sl(hidden)
    Chain(
        Dense(N_SL + 1 => hidden, tanh),
        Dense(hidden => hidden, tanh),
        Dense(hidden => N_SL),
    ) |> f64
end

function node_rk4_step_sl(model, p, c, h)
    inp(q) = vcat(q, [c])
    k1 = model(inp(p))
    k2 = model(inp(p .+ 0.5h .* k1))
    k3 = model(inp(p .+ 0.5h .* k2))
    k4 = model(inp(p .+ h .* k3))
    p .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function node_propagate_sl(model, p0, c, dt; substeps_per_unit)
    # The integration grid is a fixed numerical choice, not a differentiable
    # model quantity. Keeping ceil/Int conversion off the AD tape also avoids
    # undefined integer cotangents in Zygote.
    n_sub = Zygote.@ignore max(1, ceil(Int, abs(dt) * substeps_per_unit))
    h = dt / n_sub
    p = p0
    for _ in 1:n_sub
        p = node_rk4_step_sl(model, p, c, h)
    end
    p
end

function node_teacher_forced_loss_sl(model, data_dict, ctxs; substeps_per_unit)
    sum(begin
        ps, dts = data_dict[c]
        sum(begin
            p_pred = node_propagate_sl(
                model, ps[t], c, dts[t]; substeps_per_unit=substeps_per_unit
            )
            0.5 * sum(abs2, ps[t + 1] - p_pred)
        end for t in eachindex(dts))
    end for c in ctxs)
end

function train_node_sl(hidden, seed; n_epochs, substeps_per_unit)
    Random.seed!(seed)
    model = make_node_sl(hidden)
    opt = Flux.setup(
        Flux.Optimisers.OptimiserChain(ClipGrad(1.0), Adam(LR1_SL)),
        model,
    )

    @printf "\nNODE Stuart--Landau: hidden=%d params=%d seed=%d epochs=%d RK4 substeps/unit=%d\n" hidden count_p(model) seed n_epochs substeps_per_unit
    for epoch in 1:n_epochs
        l, g = Zygote.withgradient(model) do m
            node_teacher_forced_loss_sl(
                m, train_c, TRAIN_CTXS_SL; substeps_per_unit=substeps_per_unit
            )
        end
        (isfinite(l) && l < MAX_L) && Flux.update!(opt, model, g[1])
        if epoch == 1 || epoch % 100 == 0
            te = node_teacher_forced_loss_sl(
                model, test_c, TEST_CTXS_SL; substeps_per_unit=substeps_per_unit
            )
            @printf "  epoch %4d train=%.6e test=%.6e\n" epoch l te
            flush(stdout)
        end
    end
    model
end

function evaluate_node_sl(model; substeps_per_unit, extrap_time=6.0)
    transition_losses = Float64[]
    extrap_losses = Float64[]
    positivity_violations = Float64[]
    mass_errors = Float64[]

    for c in TEST_CTXS_SL
        ps, dts = test_c[c]
        push!(
            transition_losses,
            node_teacher_forced_loss_sl(
                model, Dict(c => (ps, dts)), [c];
                substeps_per_unit=substeps_per_unit,
            ),
        )

        A_true = A0_SL + sum(
            true_alpha_sl(c)[j] * DIRS_SL[j] for j in 1:M_SL
        )
        p_true = expv(extrap_time, A_true, ps[1]; m=60)
        p_pred = node_propagate_sl(
            model, ps[1], c, extrap_time;
            substeps_per_unit=substeps_per_unit,
        )
        push!(extrap_losses, 0.5 * sum(abs2, p_true - p_pred))
        push!(positivity_violations, count(<(-1e-6), p_pred) / length(p_pred))
        push!(mass_errors, abs(sum(p_pred) - 1.0))
    end

    (
        transition_loss=mean(transition_losses),
        extrap_loss=mean(extrap_losses),
        positivity_violation_pct=100 * mean(positivity_violations),
        mass_error=mean(mass_errors),
    )
end

if RUN_NODE_BASELINES
    node_epochs = parse(Int, get(ENV, "NODE_EPOCHS", string(N_S1_SL)))
    node_substeps = parse(Int, get(ENV, "NODE_SUBSTEPS_PER_UNIT", "32"))
    node_widths = parse.(Int, split(get(ENV, "NODE_WIDTHS", "16,64"), ","))
    node_seeds = parse.(Int, split(get(ENV, "NODE_SEEDS", "11,22,33"), ","))

    node_rows = NamedTuple[]
    for hidden in node_widths, seed in node_seeds
        node = train_node_sl(
            hidden, seed;
            n_epochs=node_epochs,
            substeps_per_unit=node_substeps,
        )
        metrics = evaluate_node_sl(node; substeps_per_unit=node_substeps)
        row = (
            hidden=hidden,
            params=count_p(node),
            seed=seed,
            epochs=node_epochs,
            substeps_per_unit=node_substeps,
            transition_loss=metrics.transition_loss,
            extrap_time=6.0,
            extrap_loss=metrics.extrap_loss,
            positivity_violation_pct=metrics.positivity_violation_pct,
            mass_error=metrics.mass_error,
        )
        push!(node_rows, row)
        @printf "NODE hidden=%d seed=%d extrap=%.4e pos=%.2f%% mass=%.3e\n" hidden seed row.extrap_loss row.positivity_violation_pct row.mass_error
    end
    CSV.write("paper/figures/sl_node_results.csv", DataFrame(node_rows))
    println("Saved paper/figures/sl_node_results.csv")
end
