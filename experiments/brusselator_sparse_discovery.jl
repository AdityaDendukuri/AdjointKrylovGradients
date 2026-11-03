"""
Sparse Brusselator CME: global-simplex encoder discovery.

Step 2 architecture: θ_{r,k}(c) = THETA_MAX · σ(a_r(c)) · softmax(β_r)[k]
  - β_r (basis_logits) is GLOBAL per channel — basis selection shared across contexts
  - encoder outputs only 4 amplitude logits (vs. 44 previously)
  - entropy penalty H(π_r) replaces gate L1 — single sparsity source

Library: 4 reaction channels × 10 cubic polynomial propensities = 40 directions.
  Channels: (+1,0) X-prod, (-1,0) X-deg, (-1,+1) X→Y, (+1,-1) 2X+Y→3X
  Propensities: all C(x,a)·C(y,b) with a+b ≤ 3

True Brusselator uses 4 of the 40:
  dir 1  : (+1, 0) × 1          ∅→X         rate = A
  dir 13 : (-1, 0) × x          X→∅         rate = 1
  dir 23 : (-1,+1) × x          X→Y         rate = B
  dir 39 : (+1,-1) × C(x,2)y    2X+Y→3X    rate = 1

Stage 1: GlobalSimplexEncoder discovers union support via τ annealing.
Stage 2: MaskedGlobalSimplexEncoder refits amplitudes on discovered support.
"""

include("brusselator_setup.jl")
# brusselator_setup.jl generates train_data_bs / test_data_bs with y_init=2.
# Generate a complementary y_init=0 dataset for Channel 1 discrimination.
# y=0 kills all y-dependent propensities (C(y,k)=0 for k≥1), making dir 1
# (constant production) the only viable Channel 1 direction.
println("\nGenerating complementary training data (y_init=0)...")
train_data_y0_bs = [(d = ssa_trajectory_bs(c; y_init=0); @printf "  train ctx (%.2f,%.2f) done\n" c[1] c[2]; flush(stdout); d) for c in train_ctxs_bs]
println("Generating complementary test data (y_init=0)...")
test_data_y0_bs  = [(d = ssa_trajectory_bs(c; y_init=0); @printf "  test  ctx (%.2f,%.2f) done\n" c[1] c[2]; flush(stdout); d) for c in test_ctxs_bs]

# ── Hyperparameters ────────────────────────────────────────────────────────────
const N_EPOCHS_BS1  = parse(Int,     get(ENV, "BRUSS_SP_STAGE1",   "600"))
const N_EPOCHS_BS2  = parse(Int,     get(ENV, "BRUSS_SP_STAGE2",   "400"))
const N_WARMUP_BS   = parse(Int,     get(ENV, "BRUSS_SP_WARMUP",   "200"))
const λ_THETA_BS    = parse(Float64, get(ENV, "BRUSS_SP_THETA_L2", "1e-6"))
const τ_INIT_BS     = parse(Float64, get(ENV, "BRUSS_SP_TAU_INIT", "2.0"))
const τ_MIN_BS      = parse(Float64, get(ENV, "BRUSS_SP_TAU_MIN",  "0.05"))

# Exponential anneal: τ_INIT → τ_MIN over N_EPOCHS_BS1 epochs, starting from epoch 1.
τ_schedule(ep) = max(τ_MIN_BS, τ_INIT_BS * exp(-log(τ_INIT_BS / τ_MIN_BS) * (ep - 1) / (N_EPOCHS_BS1 - 1)))

function train_loss_bs(model, data, ctxs; τ=1.0)
    # Encoder outputs α̃ (Fisher-scaled). SL_BS_S uses scaled dirs so the
    # generator is identical: Σ α̃ⱼ D̃ⱼ = Σ θⱼ Dⱼ. Regularisation on α̃.
    loss = 0.0
    for ((ps, dts), c) in zip(data, ctxs)
        α̃ = model(ctx_vec_bs(c), τ)
        for t in eachindex(dts)
            p_pred = SL_BS_S(ps[t], α̃, dts[t])
            loss += 0.5 * sum(abs2, ps[t+1] - p_pred)
        end
        loss += λ_THETA_BS * sum(abs2, α̃)
    end
    loss
end

# ── Stage 1: global-simplex discovery ─────────────────────────────────────────
# basis_logits initialised near-zero → uniform softmax (pure exploration at start)
global_model_bs = GlobalSimplexEncoder(
    base_encoder_cme_slim(),
    0.01 .* randn(N_CH_BS, K_BS),
    THETA_MAX_BS_S,
)
global_opt_bs = Flux.setup(Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipGrad(0.5),
    Flux.Optimisers.WeightDecay(1e-5),
    Adam(1e-3),
), global_model_bs)

@printf "\nStage 1: global-simplex discovery with temperature annealing\n"
@printf "  library=%d directions  epochs=%d  τ: %.2f → %.2f  ICs: y=0 + y=2\n" M_BS N_EPOCHS_BS1 τ_INIT_BS τ_MIN_BS

for epoch in 1:N_EPOCHS_BS1
    τ = τ_schedule(epoch)
    l, grads = Zygote.withgradient(global_model_bs) do model
        train_loss_bs(model, train_data_bs,    train_ctxs_bs; τ) +
        train_loss_bs(model, train_data_y0_bs, train_ctxs_bs; τ)
    end
    isfinite(l) || (epoch % 20 == 0 && @printf "  epoch %3d  SKIPPED (NaN/Inf)\n" epoch; continue)
    Flux.update!(global_opt_bs, global_model_bs, grads[1])

    if epoch % 100 == 0 || epoch == 1
        τ_diag = τ_schedule(epoch)
        tr = train_loss_bs(global_model_bs, train_data_bs,    train_ctxs_bs; τ=τ_diag) +
             train_loss_bs(global_model_bs, train_data_y0_bs, train_ctxs_bs; τ=τ_diag)
        te = train_loss_bs(global_model_bs, test_data_bs,     test_ctxs_bs;  τ=τ_diag) +
             train_loss_bs(global_model_bs, test_data_y0_bs,  test_ctxs_bs;  τ=τ_diag)

        # basis_logits argmax at current τ
        correct_basis = count(1:N_CH_BS) do r
            argmax(global_model_bs.basis_logits[r, :] ./ τ_diag) + (r-1)*K_BS ∈ TRUE_SUPPORT_BS
        end

        # E_c[|θ(c)|] at current τ
        θ_mean = mean(abs.(global_model_bs(ctx_vec_bs(c), τ_diag)) for c in train_ctxs_bs)
        correct_theta = count(1:N_CH_BS) do r
            idxs = (r-1)*K_BS+1:r*K_BS
            argmax(θ_mean[idxs]) + (r-1)*K_BS ∈ TRUE_SUPPORT_BS
        end

        # Entropy of softmax(β/τ) — should drop as τ decreases
        entropies = [let π_r = softmax(global_model_bs.basis_logits[r, :] ./ τ_diag)
                         -sum(π_r .* log.(π_r .+ 1e-8))
                     end for r in 1:N_CH_BS]

        @printf "  epoch %3d  τ=%.3f  train=%.5f  test=%.5f  basis=%d/%d  θ_mean=%d/%d  H=[%.2f %.2f %.2f %.2f]\n" epoch τ_diag tr te correct_basis N_CH_BS correct_theta N_CH_BS entropies...
    end
end

# ── Support extraction ─────────────────────────────────────────────────────────
# Evaluate at τ_MIN (committed state) for both readouts.

basis_pi   = [softmax(global_model_bs.basis_logits[r, :] ./ τ_MIN_BS) for r in 1:N_CH_BS]
theta_vals = mean(abs.(global_model_bs(ctx_vec_bs(c), τ_MIN_BS)) for c in train_ctxs_bs)

disc_support_basis = sort!([
    argmax(basis_pi[r]) + (r-1)*K_BS for r in 1:N_CH_BS
])
disc_support_theta = sort!([
    let idxs = (r-1)*K_BS+1:r*K_BS; argmax(theta_vals[idxs]) + (r-1)*K_BS end
    for r in 1:N_CH_BS
])

println("\nSupport recovery:")
@printf "  basis argmax:   %s  correct=%d/%d\n" repr(disc_support_basis) count(∈(TRUE_SUPPORT_BS), disc_support_basis) N_CH_BS
@printf "  θ_mean argmax:  %s  correct=%d/%d\n" repr(disc_support_theta) count(∈(TRUE_SUPPORT_BS), disc_support_theta) N_CH_BS
println("  true support:   ", TRUE_SUPPORT_BS)
println("  agreement:      ", disc_support_basis == disc_support_theta ? "yes" : "NO — readouts disagree")

disc_support = disc_support_basis   # primary
support_mask = zeros(M_BS)
for j in disc_support; support_mask[j] = 1.0; end

# Per-channel breakdown
println("\nPer-channel breakdown (π_r value | θ_mean), ★ = selected, ✓ = true direction:")
for r in 1:N_CH_BS
    idxs     = (r-1)*K_BS+1:r*K_BS
    ch_pi    = basis_pi[r]
    ch_theta = theta_vals[idxs]
    H_r      = -sum(ch_pi .* log.(ch_pi .+ 1e-8))
    top5     = sort(collect(enumerate(ch_theta)); by=x->-x[2])[1:5]
    @printf "  Channel %d (%+d,%+d)  H=%.3f:\n" r CHANNELS_BS[r][1] CHANNELS_BS[r][2] H_r
    for (rank, (k, g)) in enumerate(top5)
        j         = (r-1)*K_BS + k
        true_mark = j ∈ TRUE_SUPPORT_BS ? " ✓" : ""
        sel_mark  = rank == 1 ? " ★" : ""
        @printf "    dir %2d  π=%.4f  θ_mean=%.4f  %s%s%s\n" j ch_pi[k] g LABELS_BS[j] true_mark sel_mark
    end
end

# ── Stage 2: amplitude refit ───────────────────────────────────────────────────
masked_model_bs = MaskedGlobalSimplexEncoder(
    deepcopy(global_model_bs.encoder),
    support_mask,
    THETA_MAX_BS_S,
)
masked_opt_bs = Flux.setup(Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipGrad(0.5),
    Adam(1e-3),
), masked_model_bs)

@printf "\nStage 2: amplitude refit on |support|=%d\n" length(disc_support)

for epoch in 1:N_EPOCHS_BS2
    l, grads = Zygote.withgradient(masked_model_bs) do model
        train_loss_bs(model, train_data_bs, train_ctxs_bs)
    end
    isfinite(l) || continue
    Flux.update!(masked_opt_bs, masked_model_bs, grads[1])

    if epoch % 100 == 0 || epoch == 1
        tr = train_loss_bs(masked_model_bs, train_data_bs, train_ctxs_bs)
        te = train_loss_bs(masked_model_bs, test_data_bs,  test_ctxs_bs)
        @printf "  epoch %3d  train=%.5f  test=%.5f\n" epoch tr te
    end
end

# ── Reactive SINDy (Hoffmann, Fröhner & Noé 2019) — trajectory level ──────────
# For each SSA trajectory and time window, evaluate propensity basis functions
# at the instantaneous state and regress the observed finite-difference
# increment. We evaluate both the original shared-coefficient fit and a matched
# context-aware fit θ_j(A,B)=β_j0+β_jA A+β_jB B. The latter contains the true
# context dependence and is the fair baseline for support recovery.

const N_SINDY_TRAJ = parse(Int, get(ENV, "SINDY_N_TRAJ", "500"))
const SINDY_SEED = parse(Int, get(ENV, "SINDY_SEED", "2026"))
const BRUSS_OUTPUT_DIR = get(ENV, "BRUSS_OUTPUT_DIR", "paper/figures")

function reactive_sindy_dataset(
    ctxs;
    n_traj=N_SINDY_TRAJ,
    y_inits=(0, 2),
    seed=SINDY_SEED,
)
    Random.seed!(seed)
    nu_X = Float64[c[1] for c in CHANNELS_BS]
    nu_Y = Float64[c[2] for c in CHANNELS_BS]

    Φ_X = Vector{Float64}[]
    Φ_Y = Vector{Float64}[]
    Z_ctx = Vector{Float64}[]
    dX_vec = Float64[]; dY_vec = Float64[]

    t_eval = [0.0; collect(T_SAVE_BS)]   # [0, 0.05, 0.25, 0.75]

    for c in ctxs, y_init in y_inits
        A_ctx, B_ctx = c
        x0 = clamp(max(3, round(Int, A_ctx * OMEGA_BS)), 0, X_MAX_BS)
        u0_s = [:X => x0, :Y => y_init]
        ps_s = [:k1A => Float64(A_ctx)*OMEGA_BS, :k2 => 1.0/OMEGA_BS^2,
                :k3B => Float64(B_ctx), :k4 => 1.0]

        for _ in 1:n_traj
            sol = solve(JumpProblem(BRUS_RN_BS, u0_s, (0.0, maximum(T_SAVE_BS)), ps_s),
                        SSAStepper())
            for t_idx in 1:length(DTS_BS)
                x_t   = clamp(Int(round(sol(t_eval[t_idx  ])[1])), 0, X_MAX_BS)
                y_t   = clamp(Int(round(sol(t_eval[t_idx  ])[2])), 0, Y_MAX_BS)
                x_tp1 = clamp(Int(round(sol(t_eval[t_idx+1])[1])), 0, X_MAX_BS)
                y_tp1 = clamp(Int(round(sol(t_eval[t_idx+1])[2])), 0, Y_MAX_BS)
                # Instantaneous propensity basis at (x_t, y_t)
                phi_t = [comb_bs(x_t, a)*comb_bs(y_t, b) for (a,b) in MONOMIALS_BS]
                push!(Φ_X, vcat([nu_X[r] .* phi_t for r in 1:N_CH_BS]...))
                push!(Φ_Y, vcat([nu_Y[r] .* phi_t for r in 1:N_CH_BS]...))
                push!(Z_ctx, [1.0, Float64(A_ctx), Float64(B_ctx)])
                push!(dX_vec, Float64(x_tp1 - x_t) / DTS_BS[t_idx])
                push!(dY_vec, Float64(y_tp1 - y_t) / DTS_BS[t_idx])
            end
        end
    end

    Φ = vcat(hcat(Φ_X...)', hcat(Φ_Y...)')   # (2·n_traj·n_ctx·n_steps) × M_BS
    y = vcat(dX_vec, dY_vec)
    Z = vcat(hcat(Z_ctx...)', hcat(Z_ctx...)')
    Matrix(Φ), y, Matrix(Z)
end

function context_design(Φ, Z)
    Ψ = zeros(size(Φ, 1), 3M_BS)
    for j in 1:M_BS
        Ψ[:, 3j-2:3j] .= reshape(Φ[:, j], :, 1) .* Z
    end
    Ψ
end

context_scores(β, Z) = vec(maximum(abs.(Z * β); dims=1))

include("../src/reactive_sindy.jl")

@printf "\n━━━ Reactive SINDy — trajectory level (%d traj/ctx/IC × %d ctxs × 2 ICs) ━━━\n" N_SINDY_TRAJ length(train_ctxs_bs)
Φ_sindy, y_sindy, Z_sindy = reactive_sindy_dataset(train_ctxs_bs)
Ψ_sindy = context_design(Φ_sindy, Z_sindy)
stats_shared = reactive_sindy_regression_stats(Φ_sindy, y_sindy)
stats_context = reactive_sindy_regression_stats(Ψ_sindy, y_sindy)
Z_train = reduce(vcat, (
    reshape([1.0, Float64(c[1]), Float64(c[2])], 1, :)
    for c in train_ctxs_bs
))

const SINDY_L1_RATIOS = (0.5, 0.9, 1.0)
const SINDY_ALPHA_MULTS = 10.0 .^ range(-5.0, -0.2; length=12)
const SINDY_CUTOFFS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.75, 1.00)

function better_sindy(candidate, best)
    candidate.n_correct > best.n_correct ||
    (candidate.n_correct == best.n_correct &&
     candidate.n_spurious < best.n_spurious) ||
    (candidate.n_correct == best.n_correct &&
     candidate.n_spurious == best.n_spurious &&
     candidate.coeff_error < best.coeff_error)
end

true_mean = mean(true_theta_bs(c) for c in train_ctxs_bs)
shared_alpha_scale = max(maximum(stats_shared.cross), 1e-12)
context_alpha_scale = max(maximum(stats_context.cross), 1e-12)

# This oracle sweep gives Reactive SINDy its best possible support result over
# the reported elastic-net/cutoff grid. A validation-selected result can only
# be weaker, so this is conservative when comparing support recovery.
sindy_result = let best = (
        n_correct=-1, n_spurious=M_BS, coeff_error=Inf,
        alpha=NaN, l1_ratio=NaN, cutoff=NaN,
        disc=Int[], theta=zeros(M_BS),
    )
    for l1_ratio in SINDY_L1_RATIOS, mult in SINDY_ALPHA_MULTS
        alpha = shared_alpha_scale * mult
        θ_fit = nonnegative_elastic_net(
            stats_shared; alpha=alpha, l1_ratio=l1_ratio,
        )
        for cutoff in SINDY_CUTOFFS
            θ = copy(θ_fit)
            θ[θ .< cutoff] .= 0.0
            disc = findall(>(0.0), θ)
            n_c = length(intersect(Set(disc), Set(TRUE_SUPPORT_BS)))
            n_s = length(setdiff(Set(disc), Set(TRUE_SUPPORT_BS)))
            candidate = (
                n_correct=n_c, n_spurious=n_s,
                coeff_error=norm(θ - true_mean),
                alpha=alpha, l1_ratio=l1_ratio, cutoff=cutoff,
                disc=disc, theta=θ,
            )
            better_sindy(candidate, best) && (best = candidate)
        end
    end
    best
end

true_context = reduce(vcat, (
    reshape(true_theta_bs(c), 1, :) for c in train_ctxs_bs
))
sindy_context_result = let best = (
        n_correct=-1, n_spurious=M_BS, coeff_error=Inf,
        alpha=NaN, l1_ratio=NaN, cutoff=NaN,
        disc=Int[], beta=zeros(3, M_BS),
    )
    for l1_ratio in SINDY_L1_RATIOS, mult in SINDY_ALPHA_MULTS
        alpha = context_alpha_scale * mult
        β_fit = reshape(
            nonnegative_elastic_net(
                stats_context; alpha=alpha, l1_ratio=l1_ratio,
            ),
            3, M_BS,
        )
        for cutoff in SINDY_CUTOFFS
            scores = context_scores(β_fit, Z_train)
            active = scores .>= cutoff
            β = copy(β_fit)
            β[:, .!active] .= 0.0
            disc = findall(active)
            n_c = length(intersect(Set(disc), Set(TRUE_SUPPORT_BS)))
            n_s = length(setdiff(Set(disc), Set(TRUE_SUPPORT_BS)))
            candidate = (
                n_correct=n_c, n_spurious=n_s,
                coeff_error=norm(Z_train * β - true_context),
                alpha=alpha, l1_ratio=l1_ratio, cutoff=cutoff,
                disc=disc, beta=β,
            )
            better_sindy(candidate, best) && (best = candidate)
        end
    end
    best
end

θ_sindy = sindy_result.theta
β_sindy_context = sindy_context_result.beta
scores_sindy_context = context_scores(β_sindy_context, Z_sindy)
function one_per_channel_support(scores)
    sort([
        let idxs=(r-1)*K_BS+1:r*K_BS
            first(idxs) + argmax(scores[idxs]) - 1
        end
        for r in 1:N_CH_BS
    ])
end
shared_channel_support = one_per_channel_support(abs.(θ_sindy))
context_channel_support = one_per_channel_support(scores_sindy_context)

@printf "\nOracle-best nonnegative elastic-net Reactive SINDy:\n"
@printf "  shared:  α=%.3e l1_ratio=%.1f cutoff=%.2f support=%s (%d/4 true, %d spurious)\n" sindy_result.alpha sindy_result.l1_ratio sindy_result.cutoff repr(sindy_result.disc) sindy_result.n_correct sindy_result.n_spurious
@printf "  context: α=%.3e l1_ratio=%.1f cutoff=%.2f support=%s (%d/4 true, %d spurious)\n" sindy_context_result.alpha sindy_context_result.l1_ratio sindy_context_result.cutoff repr(sindy_context_result.disc) sindy_context_result.n_correct sindy_context_result.n_spurious
@printf "  one-per-channel readout: shared=%s (%d/4), context=%s (%d/4)\n" repr(shared_channel_support) count(∈(TRUE_SUPPORT_BS), shared_channel_support) repr(context_channel_support) count(∈(TRUE_SUPPORT_BS), context_channel_support)

# ── Evaluation ─────────────────────────────────────────────────────────────────
println("\n━━━ Union support recovery ━━━")
true_union = union_support_bs(vcat(train_ctxs_bs, test_ctxs_bs))
@printf "true union support      = %s\n" repr(true_union)
@printf "discovered support      = %s\n" repr(disc_support)
@printf "exact union recovered   = %s\n" (disc_support == true_union ? "yes" : "no")

println("\n━━━ Rate recovery on test contexts ━━━")
@printf "%-14s  %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n" "c=(A,B)" "A_true" "A_hat" "B_true" "B_hat" "k2_true" "k2_hat" "k4_true" "k4_hat"

let
    errs_k1 = Float64[]; errs_k2 = Float64[]
    errs_k3 = Float64[]; errs_k4 = Float64[]

    for c in test_ctxs_bs
        θ_true = true_theta_bs(c)
        θ_hat  = masked_model_bs(ctx_vec_bs(c)) ./ FISHER_S_BS   # α̃ → physical θ

        push!(errs_k1, abs(θ_hat[1]  - θ_true[1])  / max(abs(θ_true[1]),  1e-8))
        push!(errs_k4, abs(θ_hat[13] - θ_true[13]) / max(abs(θ_true[13]), 1e-8))
        push!(errs_k3, abs(θ_hat[23] - θ_true[23]) / max(abs(θ_true[23]), 1e-8))
        push!(errs_k2, abs(θ_hat[39] - θ_true[39]) / max(abs(θ_true[39]), 1e-8))

        cstr = @sprintf "(%.2f,%.2f)" c[1] c[2]
        @printf "%-14s  %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n" cstr θ_true[1] θ_hat[1] θ_true[23] θ_hat[23] θ_true[39] θ_hat[39] θ_true[13] θ_hat[13]
    end

    println("\n━━━ Mean relative errors ━━━")
    @printf "  k₁ (=A,   varies):  %.1f%%\n" 100 * mean(errs_k1)
    @printf "  k₃ (=B,   varies):  %.1f%%\n" 100 * mean(errs_k3)
    @printf "  k₂ (=1,   fixed ):  %.1f%%\n" 100 * mean(errs_k2)
    @printf "  k₄ (=1,   fixed ):  %.1f%%\n" 100 * mean(errs_k4)

    println("\n━━━ Spurious / missed directions ━━━")
    spurious = setdiff(disc_support, TRUE_SUPPORT_BS)
    missed   = setdiff(TRUE_SUPPORT_BS, disc_support)
    isempty(spurious) ? println("  spurious: none") : println("  spurious: ", [LABELS_BS[j] for j in spurious])
    isempty(missed)   ? println("  missed:   none") : println("  missed:   ", [LABELS_BS[j] for j in missed])

    # ── Save results ──────────────────────────────────────────────────────────
    using CSV, DataFrames
    mkpath(BRUSS_OUTPUT_DIR)

    # Use θ_mean (normalised per channel) as filter weights for the figure.
    # This now reflects actual model contribution rather than a disconnected gate.
    # Convert α̃ → physical θ for the figure weights
    theta_phys = theta_vals ./ FISHER_S_BS
    norm_w = similar(theta_phys)
    for r in 1:N_CH_BS
        idxs   = (r-1)*K_BS+1:r*K_BS
        ch_max = maximum(theta_phys[idxs])
        norm_w[idxs] .= theta_phys[idxs] ./ max(ch_max, 1e-8)
    end
    pi_vals = vcat([basis_pi[r] for r in 1:N_CH_BS]...)
    gate_df = DataFrame(
        dir           = 1:M_BS,
        filter_w      = theta_phys,
        filter_w_norm = norm_w,
        pi_r          = pi_vals,
        channel       = [(j-1) ÷ K_BS + 1 for j in 1:M_BS],
        monomial      = [(j-1) % K_BS + 1  for j in 1:M_BS],
        is_true       = [j ∈ TRUE_SUPPORT_BS for j in 1:M_BS],
    )
    CSV.write(joinpath(BRUSS_OUTPUT_DIR, "brusselator_gates.csv"), gate_df)

    rate_rows = NamedTuple[]
    for c in test_ctxs_bs
        θt = true_theta_bs(c)
        θh = masked_model_bs(ctx_vec_bs(c)) ./ FISHER_S_BS   # α̃ → physical θ
        push!(rate_rows, (
            A_true  = θt[1],   A_pred  = θh[1],
            B_true  = θt[23],  B_pred  = θh[23],
            k2_true = θt[39],  k2_pred = θh[39],
            k4_true = θt[13],  k4_pred = θh[13],
        ))
    end
    CSV.write(
        joinpath(BRUSS_OUTPUT_DIR, "brusselator_rates.csv"),
        DataFrame(rate_rows),
    )

    # Reactive SINDy coefficients for figure Panel D
    sindy_df = DataFrame(
        dir           = 1:M_BS,
        coeff_abs     = abs.(θ_sindy),
        coeff_norm    = abs.(θ_sindy) ./ max(maximum(abs.(θ_sindy)), 1e-8),
        channel       = [(j-1) ÷ K_BS + 1 for j in 1:M_BS],
        monomial      = [(j-1) % K_BS + 1  for j in 1:M_BS],
        is_true       = [j ∈ TRUE_SUPPORT_BS for j in 1:M_BS],
    )
    CSV.write(
        joinpath(BRUSS_OUTPUT_DIR, "brusselator_sindy.csv"),
        sindy_df,
    )

    # Matched context-aware SINDy: group magnitudes and coefficient predictions
    # on held-out contexts. Support selection above uses training contexts only.
    sindy_context_df = DataFrame(
        dir        = 1:M_BS,
        coeff_abs  = scores_sindy_context,
        coeff_norm = scores_sindy_context ./
                     max(maximum(scores_sindy_context), 1e-8),
        channel    = [(j-1) ÷ K_BS + 1 for j in 1:M_BS],
        monomial   = [(j-1) % K_BS + 1 for j in 1:M_BS],
        is_true    = [j ∈ TRUE_SUPPORT_BS for j in 1:M_BS],
    )
    CSV.write(
        joinpath(BRUSS_OUTPUT_DIR, "brusselator_sindy_context.csv"),
        sindy_context_df,
    )

    context_rate_rows = NamedTuple[]
    for c in test_ctxs_bs
        z = [1.0, Float64(c[1]), Float64(c[2])]
        θh = vec(z' * β_sindy_context)
        θt = true_theta_bs(c)
        push!(context_rate_rows, (
            seed=SINDY_SEED,
            A_ctx=Float64(c[1]), B_ctx=Float64(c[2]),
            A_true=θt[1], A_pred=θh[1],
            B_true=θt[23], B_pred=θh[23],
            k2_true=θt[39], k2_pred=θh[39],
            k4_true=θt[13], k4_pred=θh[13],
        ))
    end
    CSV.write(
        joinpath(BRUSS_OUTPUT_DIR, "brusselator_sindy_context_rates.csv"),
        DataFrame(context_rate_rows),
    )
    hyper_df = DataFrame(
        model=["shared", "context-aware"],
        seed=fill(SINDY_SEED, 2),
        trajectories_per_context_ic=fill(N_SINDY_TRAJ, 2),
        alpha=[sindy_result.alpha, sindy_context_result.alpha],
        l1_ratio=[sindy_result.l1_ratio, sindy_context_result.l1_ratio],
        cutoff=[sindy_result.cutoff, sindy_context_result.cutoff],
        true_hits=[sindy_result.n_correct, sindy_context_result.n_correct],
        spurious=[sindy_result.n_spurious, sindy_context_result.n_spurious],
        support=[repr(sindy_result.disc), repr(sindy_context_result.disc)],
        channel_support=[repr(shared_channel_support), repr(context_channel_support)],
        channel_true_hits=[
            count(∈(TRUE_SUPPORT_BS), shared_channel_support),
            count(∈(TRUE_SUPPORT_BS), context_channel_support),
        ],
        selection=fill("oracle support grid", 2),
    )
    CSV.write(
        joinpath(BRUSS_OUTPUT_DIR, "brusselator_sindy_hyperparameters.csv"),
        hyper_df,
    )

    println("\nSaved brusselator_gates.csv, brusselator_rates.csv, " *
            "brusselator_sindy.csv, brusselator_sindy_context.csv, and " *
            "Brusselator SINDy hyperparameter/rate outputs")
end
