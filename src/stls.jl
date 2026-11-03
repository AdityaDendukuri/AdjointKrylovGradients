# =============================================================================
# Sequential Thresholded Least Squares (STLS) Optimizer
# =============================================================================

using Optim

"""
    STLSResult

Result container for STLS optimization.

# Fields
- `θ_opt`: Optimized parameter dictionary
- `objective`: Final objective value
- `history`: Dict with iteration history (objective, nnz, threshold)
- `converged`: Whether the support converged
- `n_iterations`: Total iterations performed
"""
struct STLSResult{T}
    θ_opt::Dict{T,Vector{Float64}}
    objective::Float64
    history::Dict{String,Vector}
    converged::Bool
    n_iterations::Int
end

"""
    optimize_stls(P, R, X, dt, θ_init;
                  τ=0.01, inner_iter=200, inner_rtol=1e-6, outer_iter=20, polish_iter=50,
                  verbose=true, feature_map=quad_features) -> STLSResult

Sequential Thresholded Least Squares for sparse CME parameter inference.

Alternates between:
1. **Optimization**: L-BFGS on the current parameter support, to a relative
   gradient tolerance
2. **Thresholding**: Hard threshold to promote sparsity

This is inspired by SINDy (Sparse Identification of Nonlinear Dynamics)
and is particularly effective for discovering sparse reaction kinetics.

# Arguments
- `P`: Vector of probability distributions at observation times
- `R`: Reaction stoichiometries
- `X`: State space
- `dt`: Time step between observations
- `θ_init`: Initial parameter dictionary

# Keyword Arguments
- `τ`: Threshold fraction relative to max |θ| (default: 0.01)
- `inner_iter`: cap on L-BFGS iterations per thresholding cycle (default: 200)
- `inner_rtol`: each inner solve stops once the gradient has fallen by this
  factor from its value at the start of the solve (default: 1e-6). Thresholding
  a partially converged iterate makes the support depend on the optimizer's
  path; `nothing` restores a fixed budget of `inner_iter` steps
- `outer_iter`: Number of threshold-optimize cycles (default: 20)
- `polish_iter`: Final optimization iterations on discovered support (default: 50)
- `verbose`: Print progress (default: true)
- `feature_map`: Feature function (default: quad_features, can use cubic_features)

# Returns
- `STLSResult` containing optimized parameters and convergence info

# Algorithm
```
for iter in 1:outer_iter
    θ = L-BFGS(θ, inner_rtol)    # Optimize on current support to tolerance
    τ_val = τ * max(|θ|)          # Adaptive threshold
    θ[|θ| < τ_val] = 0            # Hard threshold
end
θ = L-BFGS(θ, polish_iter)        # Final polish on support
```

# Example
```julia
result = optimize_stls(P, R, X, dt, θ_init;
                       τ=0.05, outer_iter=20, verbose=true)
θ_learned = result.θ_opt

# For trimolecular reactions, use cubic features:
result = optimize_stls(P, R, X, dt, θ_init;
                       τ=0.05, feature_map=cubic_features)
```
"""
function optimize_stls(P, R, X, dt, θ_init; kw...)
    optimize_stls([(P, X)], R, dt, θ_init; kw...)
end

"""
    optimize_stls(windows, R, dt, θ_init; ...) -> STLSResult

Multi-window variant. `windows` is a `Vector` of `(P_w, X_w)` tuples, one per
time window. The objective and gradient are averaged across windows; each window
can have its own state space. Useful when the full trajectory spans a much longer
horizon than any single window (e.g. Brusselator: five 10-unit windows covering
[0,50] instead of one 101-step solve).
"""
function optimize_stls(windows::AbstractVector{<:Tuple}, R, dt, θ_init;
                       τ=0.01,
                       λ=0.0,
                       inner_iter=200,
                       inner_rtol=1e-6,
                       optimizer=LBFGS(),
                       outer_iter=20,
                       polish_iter=50,
                       verbose=true,
                       enforce_positive=true,
                       feature_map=quad_features,
                       verbose_history=false,
                       gradient_fn=gradient_adjoint_krylov!,
                       fisher_precond=false,
                       per_reaction_threshold=false,
                       weighted_loss=false,
                       weight_eps=1e-6,
                       m_krylov=25,
                       n_quadrature=16,
                       elimination_patience=1,
                       threshold_mode=:relative,
                       significance_level=3.0,
                       warmup_iter=200,
                       reject_damaging_threshold=true,
                       damage_tolerance=0.25)

    nw = length(windows)
    N  = length(first(first(windows)[2]))
    K  = length(feature_map(first(first(windows)[2])))
    # Build per-window derivative operators (each window has its own state space)
    dA_list = [(P_w, X_w, build_derivatives(X_w, R; feature_map=feature_map, K=K))
               for (P_w, X_w) in windows]
    sorted_R = sort(collect(R))

    # Precompute per-window inverse-variance weights for weighted loss.
    # w_k[x] = 1 / (P[k+1][x] + weight_eps) — fixed from data, not recomputed.
    W_vecs_list = weighted_loss ? [compute_weights(P_w; eps=weight_eps) for (P_w, _) in windows] : nothing

    # Aliases for single-window fisher_precond (uses first window; unused for nw>1)
    P_first, X_first, dA_first = dA_list[1]

    # Flatten parameters
    θ = flatten_params(θ_init)

    # Propensity normalization scale (data-based, θ-independent):
    # σ[j] = sqrt(Σ_k ‖dA[ζ][f] P[k]‖²)  — RMS dynamical influence of parameter j
    # Threshold in θ̃=θ×σ space so all parameters compete by dynamical contribution,
    # not raw rate magnitude. Computed once from the trajectory, no circularity.
    fisher_scale = if fisher_precond
        T_steps = length(P_first) - 1
        scale = zeros(length(θ))
        idx = 1
        for ζ in sorted_R, f in 1:K
            s = 0.0
            for k in 1:T_steps
                v = dA_first[ζ][f] * P_first[k]
                s += dot(v, v)
            end
            scale[idx] = sqrt(s)
            idx += 1
        end
        max.(scale, 1e-30)
    else
        nothing
    end

    # Track progress
    history = Dict(
        "iteration" => Int[],
        "objective" => Float64[],
        "nnz" => Int[],
        "threshold" => Float64[]
    )
    if verbose_history
        history["θ_snapshots"] = Vector{Float64}[]
    end

    g_w_buf = zeros(length(θ))   # reusable buffer for per-window gradient accumulation
    # consecutive outer iterations each coefficient has spent below threshold
    below_count = zeros(Int, length(θ))

    if verbose
        println("="^60)
        println("SEQUENTIAL THRESHOLDED LEAST SQUARES")
        println("="^60)
        @printf("%-8s %-12s %-8s %-10s\n", "Iter", "Objective", "NNZ", "Threshold")
        println("-"^60)
    end

    # MAX_LOG_RATE: rates above e^10 ≈ 22026/min are unphysical for any
    # biochemical system. Guards prevent optimizer step explosion (large LBFGS
    # steps in log-rate space → exp(φ) = Inf → generator has Inf entries →
    # Arnoldi produces NaN basis vectors).
    MAX_LOG_RATE = 10.0

    iterations_done = 0
    prev_support = nothing   # track actual support vector for true convergence check

    for outer in 1:outer_iter
        iterations_done = outer

        # 1. Optimize on current support in linear k-space.
        # Uses the full parameter vector θ so that the LBFGS can move any
        # component (including currently-zero ones), which is essential for
        # STLS: the thresholding creates zeros that must be allowed to grow
        # back if the sparsity pattern hasn't converged yet.
        MAX_RATE = exp(MAX_LOG_RATE)  # ≈ 22026/min
        f_inner = θ_flat -> begin
            # Penalty for negative or astronomically large rates:
            # keeps A finite so expv never sees Inf in the generator.
            if any(θ_flat .< -1e-6) || any(θ_flat .> MAX_RATE)
                return 1e10 + sum(max.(θ_flat .- MAX_RATE, 0.0)) +
                              sum(max.(-θ_flat .- 1e-6, 0.0))
            end
            θ_dict = unflatten_params(θ_flat, R, K)
            tot = 0.0
            for (iw, (P_w, X_w, _)) in enumerate(dA_list)
                W_w = W_vecs_list !== nothing ? W_vecs_list[iw] : nothing
                v = try objective(P_w, R, θ_dict, X_w, dt, λ; feature_map=feature_map, W_vecs=W_w, m_krylov=m_krylov) catch; return 1e10 end
                tot += v
            end
            tot / nw
        end

        g_inner! = (g_flat, θ_flat) -> begin
            fill!(g_flat, 0.0)
            if any(θ_flat .< -1e-6) || any(θ_flat .> MAX_RATE)
                g_flat .= max.(θ_flat .- MAX_RATE, 0.0) .- max.(-θ_flat .- 1e-6, 0.0)
                return
            end
            for (iw, (P_w, X_w, dA_w)) in enumerate(dA_list)
                W_w = W_vecs_list !== nothing ? W_vecs_list[iw] : nothing
                gradient_fn(g_w_buf, θ_flat, P_w, R, X_w, dt, dA_w, λ; feature_map=feature_map, K=K, W_vecs=W_w,
                            m_krylov=m_krylov, n_quadrature=n_quadrature)
                if any(!isfinite, g_w_buf)
                    @warn "Non-finite gradient in window; falling back to finite differences"
                    gradient!(g_w_buf, θ_flat, P_w, R, X_w, dt, dA_w, λ; feature_map=feature_map, K=K)
                    any(!isfinite, g_w_buf) && fill!(g_w_buf, 0.0)
                end
                g_flat .+= g_w_buf
            end
            g_flat ./= nw
        end

        # Every thresholding decision is made on the iterate the inner solve
        # reaches. A fixed step budget leaves that iterate partially converged,
        # so which of two correlated directions survives depends on the line
        # search and the starting point. Each pass therefore runs to a relative
        # gradient tolerance, as linear STLS solves exactly before thresholding;
        # `inner_iter` (and `warmup_iter` on the first pass) only caps it.
        iters_this_pass = (outer == 1 ? max(warmup_iter, inner_iter) : inner_iter)
        inner_opts = if inner_rtol === nothing
            Optim.Options(iterations=(outer == 1 ? warmup_iter : inner_iter),
                          show_trace=false)
        else
            g_start = similar(θ)
            g_inner!(g_start, θ)
            Optim.Options(iterations=iters_this_pass,
                          g_abstol=inner_rtol * maximum(abs, g_start),
                          show_trace=false)
        end

        result = try
            optimize(f_inner, g_inner!, θ, optimizer, inner_opts)
        catch
            nothing
        end
        result === nothing && (verbose && println("  [inner LBFGS failed, keeping current θ]"); nothing)

        θ = result !== nothing ? Optim.minimizer(result) : θ
        # Project to non-negative domain (rates must be ≥ 0)
        if enforce_positive
            θ = max.(θ, 0.0)
        end
        obj = result !== nothing ? Optim.minimum(result) : f_inner(θ)

        # 2. Hard threshold (adaptive)
        # With fisher_precond: rescale θ̃[j] = θ[j]×√D[j] so each parameter is
        # weighted by its dynamical influence before thresholding. D is fixed
        # at initialization to avoid misleading rescaling at wrong iterates.
        # With per_reaction_threshold: threshold each reaction's feature block
        # against its own max rather than the global max. This handles rate-scale
        # separation: small rates within a reaction compete only against that
        # reaction's dominant rate, not against rates of other reactions.
        # Combining both: per-reaction Fisher preconditioning — threshold each
        # direction's Fisher-scaled parameters against that direction's own max.
        # This is the most robust choice when both inter-reaction scale separation
        # (handled by per_reaction_threshold) and intra-reaction scale separation
        # (handled by fisher_precond) are present simultaneously.
        # Each branch marks which coefficients are currently below threshold; the
        # actual elimination happens afterwards, subject to hysteresis.
        below = falses(length(θ))
        threshold = 0.0
        if threshold_mode === :significance
            # Keep a coefficient only if it is statistically distinguishable from
            # zero, rather than large relative to the block leader.
            #
            # For the Gauss-Newton approximation H ≈ JᵀJ, the diagonal
            # D_j = (JᵀJ)_jj gives Var(θ̂_j) ≈ σ²/D_j, so the t-statistic is
            #
            #     t_j = |θ_j| √D_j / σ,
            #
            # where σ² is the residual variance. Note |θ_j|√D_j is exactly the
            # Fisher-scaled coordinate the relative rule already uses — the
            # difference is that it is compared against an absolute noise scale σ
            # instead of τ·max φ. A strong term therefore no longer sets the bar
            # for weak-but-real ones, and a correlated distractor that explains no
            # variance beyond noise is dropped on its own merits.
            #
            # D is taken at the current iterate and shares the Krylov paths with
            # the gradient, so it costs one extra norm per direction per node.
            D_acc = zeros(length(θ))
            g_tmp = zeros(length(θ))
            D_tmp = zeros(length(θ))
            n_resid = 0
            for (iw, (P_w, X_w, dA_w)) in enumerate(dA_list)
                W_w = W_vecs_list !== nothing ? W_vecs_list[iw] : nothing
                fill!(g_tmp, 0.0); fill!(D_tmp, 0.0)
                gradient_and_fisher!(g_tmp, D_tmp, θ, P_w, R, X_w, dt, dA_w, 0.0;
                                     feature_map=feature_map, K=K, W_vecs=W_w,
                                     m_krylov=m_krylov, n_quadrature=n_quadrature)
                D_acc .+= D_tmp
                n_resid += length(P_w[1]) * (length(P_w) - 1)
            end
            # obj is 0.5·Σ‖r‖² summed over windows, so Σ‖r‖² = 2·obj.
            dof    = max(n_resid - count(!iszero, θ), 1)
            sigma  = sqrt(max(2 * obj, 0.0) / dof)
            tstat  = abs.(θ) .* sqrt.(max.(D_acc, 0.0)) ./ max(sigma, 1e-300)
            below .= tstat .< significance_level
            threshold = significance_level * sigma
        elseif fisher_precond && fisher_scale !== nothing && per_reaction_threshold
            # Per-reaction Fisher preconditioning
            θ_dict_pr = unflatten_params(θ, R, K)
            for (i, ζ) in enumerate(sorted_R)
                base_idx = (i-1)*K
                fs_ζ = fisher_scale[base_idx+1:base_idx+K]
                θ_scaled_ζ = abs.(θ_dict_pr[ζ]) .* fs_ζ
                ζ_max = maximum(θ_scaled_ζ)
                if ζ_max > 1e-30
                    thresh_ζ = τ * ζ_max
                    threshold = max(threshold, thresh_ζ)
                    below[base_idx+1:base_idx+K] .= θ_scaled_ζ .< thresh_ζ
                end
            end
        elseif fisher_precond && fisher_scale !== nothing
            θ_scaled = θ .* fisher_scale
            threshold = τ * maximum(abs, θ_scaled)
            below .= abs.(θ_scaled) .< threshold
        elseif per_reaction_threshold
            # Per-reaction thresholding: for each reaction ζ, apply
            # threshold_ζ = τ × max(|θ_ζ|) independently.
            θ_dict_pr = unflatten_params(θ, R, K)
            for (i, ζ) in enumerate(sorted_R)
                base_idx = (i-1)*K
                θ_ζ = θ_dict_pr[ζ]
                ζ_max = maximum(abs.(θ_ζ))
                if ζ_max > 1e-30
                    thresh_ζ = τ * ζ_max
                    threshold = max(threshold, thresh_ζ)
                    below[base_idx+1:base_idx+K] .= abs.(θ_ζ) .< thresh_ζ
                end
            end
        else
            θ_max = maximum(abs.(θ))
            threshold = τ * θ_max
            below .= abs.(θ) .< threshold
        end

        # Elimination hysteresis.  Thresholding acts on the current iterate, so a
        # true coefficient that is transiently underestimated is otherwise removed
        # on the spot.  That is not hypothetical: under Fisher scaling the term
        # left closest to the threshold is the one with the weakest dynamical
        # action, which on Schlogl is k3 — the *largest* raw rate — and a jittered
        # start that undershoots it drops it permanently in practice.
        # Requiring a coefficient to sit below threshold for `elimination_patience`
        # consecutive outer iterations lets L-BFGS pull such a term back first.
        # patience = 1 reproduces the original immediate-elimination behaviour.
        below_count .= ifelse.(below, below_count .+ 1, 0)
        θ_sparse = copy(θ)
        θ_sparse[below_count .>= elimination_patience] .= 0.0

        # Objective guard. A removal that damages the fit was a mistake about the
        # support, not a discovery of sparsity, so we test it before accepting it.
        # Removing a true direction can raise the objective by an order of
        # magnitude while leaving the support size stable, at which point the
        # nnz-based convergence test below exits and reports success.
        # `reject_damaging_threshold` reinstates any removal that increases the
        # objective by more than `damage_tolerance` in relative terms.
        if reject_damaging_threshold && any(θ_sparse .!= θ)
            obj_thresholded = f_inner(θ_sparse)
            if isfinite(obj_thresholded) && isfinite(obj) &&
               obj_thresholded > (1 + damage_tolerance) * max(obj, eps())
                verbose && @printf("  [threshold rejected: objective %.3e -> %.3e]\n",
                                   obj, obj_thresholded)
                θ_sparse = copy(θ)
                # Do not let the same coefficients be removed again next pass on
                # the strength of the counter they accumulated this pass.
                below_count .= 0
            end
        end

        nnz_count = count(x -> abs(x) > 1e-10, θ_sparse)

        # Store history
        push!(history["iteration"], outer)
        push!(history["objective"], obj)
        push!(history["nnz"], nnz_count)
        push!(history["threshold"], threshold)
        if verbose_history
            push!(history["θ_snapshots"], copy(θ))  # pre-threshold, for smooth trajectory logging
        end

        if verbose
            @printf("%-8d %-12.6e %-8d %-10.6e\n",
                    outer, obj, nnz_count, threshold)
            flush(stdout)
        end

        # Update for next iteration
        θ = θ_sparse
        curr_support = abs.(θ) .> 1e-10

        # Check convergence: support identity (not just cardinality) must be unchanged
        if prev_support !== nothing && curr_support == prev_support
            verbose && println("\nSupport converged at iteration $outer")
            break
        end
        prev_support = curr_support
    end

    # 3. Polish: Final optimization on discovered support
    verbose && println("-"^60)
    verbose && println("Polishing solution on discovered support...")

    support = abs.(θ) .> 1e-10
    θ_support = θ[support]

    if length(θ_support) > 0
        # Polish in log-rate space (recommended by optimization theory):
        #   k_r = exp(φ_r)  →  unconstrained, naturally positive, better conditioned.
        # The optimizer sees O(1) variables regardless of rate magnitude,
        # which is critical when rates span orders of magnitude (e.g. k3/k4 ≈ 30×).
        # Chain rule: ∂L/∂φ_r = ∂L/∂k_r · k_r
        log_θ_support = log.(max.(θ_support, 1e-12))

        f_polish = φ_s -> begin
            if any(φ_s .> MAX_LOG_RATE)
                return 1e10 + sum(max.(φ_s .- MAX_LOG_RATE, 0.0))
            end
            θ_full = zeros(length(θ))
            θ_full[support] = exp.(φ_s)
            θ_dict = unflatten_params(θ_full, R, K)
            tot = 0.0
            for (iw, (P_w, X_w, _)) in enumerate(dA_list)
                W_w = W_vecs_list !== nothing ? W_vecs_list[iw] : nothing
                v = try objective(P_w, R, θ_dict, X_w, dt, 0.0; feature_map=feature_map, W_vecs=W_w, m_krylov=m_krylov) catch; return 1e10 end
                tot += v
            end
            tot / nw
        end

        g_polish! = (g_s, φ_s) -> begin
            if any(φ_s .> MAX_LOG_RATE)
                g_s .= max.(φ_s .- MAX_LOG_RATE, 0.0)
                return
            end
            k_s = exp.(φ_s)
            θ_full = zeros(length(θ))
            θ_full[support] = k_s
            g_full = zeros(length(θ))
            for (iw, (P_w, X_w, dA_w)) in enumerate(dA_list)
                W_w = W_vecs_list !== nothing ? W_vecs_list[iw] : nothing
                gradient_fn(g_w_buf, θ_full, P_w, R, X_w, dt, dA_w, 0.0; feature_map=feature_map, K=K, W_vecs=W_w,
                            m_krylov=m_krylov, n_quadrature=n_quadrature)
                if any(!isfinite, g_w_buf)
                    @warn "Non-finite gradient in polish phase; falling back to finite differences"
                    gradient!(g_w_buf, θ_full, P_w, R, X_w, dt, dA_w, 0.0; feature_map=feature_map, K=K)
                    any(!isfinite, g_w_buf) && fill!(g_w_buf, 0.0)
                end
                g_full .+= g_w_buf
            end
            g_full ./= nw
            g_s .= g_full[support] .* k_s
        end

        result_polish = optimize(f_polish, g_polish!, log_θ_support, LBFGS(),
                                 Optim.Options(iterations=polish_iter, show_trace=false,
                                               store_trace=true, extended_trace=verbose_history))

        # Reconstruct full parameter vector (map back from log-space)
        θ_final = zeros(length(θ))
        θ_final[support] = exp.(Optim.minimizer(result_polish))
        obj_final = Optim.minimum(result_polish)
        converged = Optim.converged(result_polish)

        # Record polish history
        if verbose_history
            polish_objs = Float64[]
            polish_θ = Vector{Float64}[]
            for state in Optim.trace(result_polish)
                push!(polish_objs, state.value)
                if haskey(state.metadata, "x")
                    θ_full_snap = zeros(length(θ))
                    θ_full_snap[support] = exp.(state.metadata["x"])  # log-rate → rate
                    push!(polish_θ, θ_full_snap)
                end
            end
            history["polish_objectives"] = polish_objs
            history["polish_θ_snapshots"] = polish_θ
        end
    else
        θ_final = zeros(length(θ))
        θ_dict0 = unflatten_params(θ_final, R, K)
        obj_final = sum(objective(P_w, R, θ_dict0, X_w, dt, 0.0; feature_map=feature_map, m_krylov=m_krylov)
                        for (P_w, X_w, _) in dA_list) / nw
        converged = true
    end

    if verbose
        @printf("Final objective: %.6e\n", obj_final)
        @printf("Final NNZ: %d / %d\n", count(abs.(θ_final) .> 1e-10), length(θ_final))
        println("="^60)
    end

    θ_opt = unflatten_params(θ_final, R, K)

    return STLSResult(θ_opt, obj_final, history, converged, iterations_done)
end

"""
    optimize_stls_twoscale(P, R, X, dt, θ_init;
                           τ_fast=0.1, τ_slow=0.01,
                           seed_slow=0.01, inner_iter=200,
                           outer_iter=40, polish_iter=100,
                           verbose=true, kwargs...) -> STLSResult

Two-scale STLS for systems with rate-scale separation.

Phase 1 (fast): run STLS with large τ_fast to identify dominant rates.
  Parameters below the threshold are zeroed — slow rates are suppressed.

Phase 2 (slow): re-initialize the zeroed features to `seed_slow` (a small
  positive value above the Phase-2 threshold) and run STLS with τ_slow using
  per-reaction thresholding so slow rates compete only within their own reaction.

This handles cases where rates span orders of magnitude within a single
reaction (e.g. propensity = k_fast * X + k_slow * XY).
"""
function optimize_stls_twoscale(P, R, X, dt, θ_init;
                                 τ_fast=0.1,
                                 τ_slow=0.01,
                                 seed_slow=0.01,
                                 inner_iter=200,
                                 outer_iter=40,
                                 polish_iter=100,
                                 verbose=true,
                                 feature_map=quad_features,
                                 gradient_fn=gradient_adjoint_krylov!,
                                 kwargs...)

    N = length(first(X))
    K = feature_map === cubic_features ? cubic_feature_dimension(N) : feature_dimension(N)

    # ── Phase 1: identify dominant (fast) rates ───────────────────
    verbose && begin
        println("="^60)
        println("TWOSCALE STLS — Phase 1: fast rates  (τ=$τ_fast)")
        println("="^60)
    end

    result1 = optimize_stls(P, R, X, dt, θ_init;
                            τ=τ_fast,
                            outer_iter=outer_iter,
                            inner_iter=inner_iter,
                            polish_iter=0,           # skip polish; Phase 2 will polish
                            verbose=verbose,
                            feature_map=feature_map,
                            gradient_fn=gradient_fn,
                            per_reaction_threshold=false,
                            kwargs...)

    # ── Build Phase 2 init: keep fast rates, seed zeros with seed_slow ──
    θ_p2 = Dict(ζ => copy(result1.θ_opt[ζ]) for ζ in R)
    n_seeded = 0
    for ζ in R
        for f in 1:K
            if abs(θ_p2[ζ][f]) < 1e-10
                θ_p2[ζ][f] = seed_slow
                n_seeded += 1
            end
        end
    end
    verbose && @printf("Phase 1 done. Seeded %d zeroed features with %.4f for Phase 2.\n\n",
                       n_seeded, seed_slow)

    # ── Phase 2: find slow rates using per-reaction thresholding ─────
    verbose && begin
        println("="^60)
        println("TWOSCALE STLS — Phase 2: slow rates  (τ=$τ_slow, per-reaction)")
        println("="^60)
    end

    result2 = optimize_stls(P, R, X, dt, θ_p2;
                            τ=τ_slow,
                            outer_iter=outer_iter,
                            inner_iter=inner_iter,
                            polish_iter=polish_iter,
                            verbose=verbose,
                            feature_map=feature_map,
                            gradient_fn=gradient_fn,
                            per_reaction_threshold=true,
                            kwargs...)

    return result2
end

"""
    print_parameter_recovery(θ_learned, θ_true, R; species_names=String[])

Pretty-print parameter recovery comparison.

# Arguments
- `θ_learned`: Learned parameters
- `θ_true`: True parameters (for comparison)
- `R`: Reaction stoichiometries
- `species_names`: Optional names for species
"""
function print_parameter_recovery(θ_learned, θ_true, R;
                                  species_names::Vector{String}=String[])
    N = length(first(R))
    K = feature_dimension(N)
    names = feature_names(N; species_names=species_names)

    println("\n" * "="^70)
    println("PARAMETER RECOVERY")
    println("="^70)

    for ζ in sort(collect(R))
        θ_l = θ_learned[ζ]
        θ_t = get(θ_true, ζ, zeros(K))

        println("\nReaction $ζ:")

        # Find non-zero entries in either
        active = findall(i -> abs(θ_l[i]) > 1e-6 || abs(θ_t[i]) > 1e-6, 1:K)

        if isempty(active)
            println("  (all parameters near zero)")
            continue
        end

        @printf("  %-15s %-12s %-12s %-12s\n",
                "Feature", "Learned", "True", "Error")
        println("  " * "-"^55)

        for i in active
            error = abs(θ_l[i] - θ_t[i])
            @printf("  %-15s %-12.6f %-12.6f %-12.6f\n",
                    names[i], θ_l[i], θ_t[i], error)
        end
    end

    println("="^70)
end
