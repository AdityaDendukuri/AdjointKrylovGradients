"""
    compute_weights(P; eps=1e-6) -> Vector{Vector{Float64}}

Compute diagonal inverse-variance weights for the statistically weighted CME loss.

For each time step k, returns weight vector

    w_k[x] = 1 / (P[k+1][x] + eps)

so that the weighted objective

    L_W(θ) = (1/2) Σ_k r_k' * diag(w_k) * r_k

down-weights residuals at low-probability (high-variance) states and up-weights
residuals at high-probability (low-variance) states, approaching a multinomial
chi-square loss. The eps floor prevents division by zero.

# Arguments
- `P`: Vector of probability distributions (length T+1)
- `eps`: Floor added to each probability before inverting (default 1e-6)

# Returns
- Vector of length T, each element a weight vector of the same length as P[1]
"""
function compute_weights(P; eps=1e-6)
    T = length(P) - 1
    [1.0 ./ (P[k+1] .+ eps) for k in 1:T]
end

# Gauss-Legendre nodes and weights on [0,1] come from gauss_legendre_01 in
# shared_krylov.jl, which supports any order at full double precision.  This file
# previously carried its own copies of 5- and 8-point tables whose literals were
# mistranscribed (two nodes of the 8-point rule off by 1.5e-8), and selected
# between them with `n_quadrature == 5 ? ... : ...`, which silently downgraded any
# request above 8 to an 8-point rule.

# ---------------------------------------------------------------------------
# _arnoldi_apply! — kept for backwards compatibility; no longer used internally.
# ---------------------------------------------------------------------------
@inline function _arnoldi_apply!(out, ks, t)
    m  = ks.m
    Vm = view(ks.V, :, 1:m)
    Hm = Matrix(view(ks.H, 1:m, 1:m))
    e1 = zeros(m); e1[1] = 1.0
    y  = exp(t .* Hm) * e1
    mul!(out, Vm, y)
    out .*= ks.beta
    return out
end

# ---------------------------------------------------------------------------
# _basis_gl_matrix!  —  batch all n_q GL columns for one Arnoldi basis.
#
# Fills out (n × n_q) with:
#   out[:, q] = β · Vm · exp(t_q · Hm) · e₁
#
# Key optimisations vs calling _arnoldi_apply! n_q times:
#   1. Hm, Vm, β are extracted once.
#   2. The n_q small (m × m) exponentials are computed via a shared
#      eigendecomposition:  exp(t_q Hm) e₁ = VE (exp.(t_q d) .* c),
#      where VE, d = eigen(Hm),  c = VE\e₁  — all precomputed once.
#   3. All n_q columns are assembled with a single GEMM (BLAS level-3).
# ---------------------------------------------------------------------------
@inline function _basis_gl_matrix!(out::Matrix{Float64}, ks, ts::AbstractVector{Float64}, β_scale::Float64)
    m  = ks.m
    Vm = view(ks.V, :, 1:m)          # n × m  (view, no copy)
    Hm = Matrix(view(ks.H, 1:m, 1:m))  # m × m  (one copy, reused below)
    e1 = zeros(m); e1[1] = 1.0

    # Eigendecompose Hm once; handle complex spectrum safely.
    VE, d = let F = eigen(Hm); F.vectors, F.values end
    # Use pinv fallback when VE is (nearly) singular — happens when the
    # Arnoldi basis is numerically degenerate (e.g. residual near zero).
    c = try VE \ e1 catch; pinv(VE) * e1 end   # m-vector, possibly complex

    n_q   = length(ts)
    Y     = Matrix{ComplexF64}(undef, m, n_q)
    for q in 1:n_q
        Y[:, q] = VE * (exp.(ts[q] .* d) .* c)
    end
    Y_r = real.(Y)   # drop imaginary rounding errors (result is real)

    # out = β_scale · Vm · Y_r    (one level-3 BLAS call)
    mul!(out, Vm, Y_r)
    out .*= β_scale
    return out
end

"""
    gradient_adjoint_krylov!(g, θ_flat, P, R, X, dt, dA, λ=0.0; ...)

Adjoint-Krylov gradient using shared Arnoldi bases per time step.

**Algorithm (per time step k):**

1. Forward Arnoldi:  `ks_f = arnoldi(A, P[k])`.
2. p̂ₖ = exp(ΔtA) P[k] from `ks_f` (no extra Arnoldi).
3. Adjoint Arnoldi:  `ks_a = arnoldi(Aᵀ, rₖ)`,  rₖ = P[k+1] - p̂ₖ.
4. Batch GL columns via eigendecomposition of the two m×m Hessenbergs:
   - `F_k[:,q] = β_f Vf exp(s_q Δt Hf) e₁`  (all q in one GEMM)
   - `Λ_k[:,q] = β_a Va exp((1-s_q) Δt Ha) e₁`
5. Fused, threaded direction phase: accumulate `g[j] += …` immediately
   without storing F_k / Λ_k across time steps.

**Cost per step:** 2 Arnoldi runs + 2 eigen(m×m) + 2 GEMMs (n×m×n_q)
  + MK sparse inner products, with the GEMM and inner products threaded.

Default `n_quadrature=5` (5-point GL), accurate to O(h¹⁰) on the analytic
integrand; use `n_quadrature=8` for very stiff problems.
"""
function gradient_adjoint_krylov!(g, θ_flat, P, R, X, dt, dA, λ=0.0;
                                   feature_map=quad_features, K=nothing,
                                   m_krylov=25, n_quadrature=5, tol=1e-8,
                                   W_vecs=nothing)
    N_species = length(first(X))
    if K === nothing
        K = feature_map === cubic_features ? cubic_feature_dimension(N_species) : feature_dimension(N_species)
    end
    θ  = unflatten_params(θ_flat, R, K)
    A  = build_generator(X, R, θ; feature_map=feature_map)
    At = A'

    fill!(g, 0.0)

    T_steps = length(P) - 1
    n       = size(A, 1)
    m_K     = min(m_krylov, n - 1)
    dts     = dt isa AbstractVector ? dt : fill(Float64(dt), T_steps)

    s_nodes, w_nodes = gauss_legendre_01(n_quadrature)
    n_q = length(s_nodes)

    # Flat parameter list in canonical sort order (matches flatten_params).
    R_sorted = sort(collect(R))
    N_mats   = [dA[ζ][f] for ζ in R_sorted for f in 1:K]
    MK       = length(N_mats)

    # Reusable per-step path matrices (allocated once, overwritten each step).
    F_k = zeros(n, n_q)
    Λ_k = zeros(n, n_q)
    buf = zeros(n)

    n_skipped = 0

    for k in 1:T_steps
        dt_k = dts[k]

        # ── Forward Arnoldi ────────────────────────────────────────────────
        ks_f = try arnoldi(A, P[k]; m=m_K, tol=tol) catch; nothing end
        if ks_f === nothing; n_skipped += 1; continue; end

        # p_pred reuses the forward basis (no additional Arnoldi).
        p_pred = _arnoldi_apply!(zeros(n), ks_f, dt_k)
        if any(!isfinite, p_pred); n_skipped += 1; continue; end

        r_k  = P[k+1] .- p_pred
        ar_k = W_vecs !== nothing ? W_vecs[k] .* r_k : r_k
        norm(ar_k) > 1e-15 || continue

        # ── Adjoint Arnoldi ────────────────────────────────────────────────
        ks_a = try arnoldi(At, ar_k; m=m_K, tol=tol) catch; nothing end
        if ks_a === nothing; n_skipped += 1; continue; end

        # ── Batch GL matrices via eigendecomposition + single GEMM ─────────
        s_fwd  = s_nodes .* dt_k           # forward  GL times: s_q · Δt
        s_adj  = (1 .- s_nodes) .* dt_k   # adjoint GL times: (1-s_q) · Δt
        _basis_gl_matrix!(F_k, ks_f, s_fwd, ks_f.beta)
        _basis_gl_matrix!(Λ_k, ks_a, s_adj, ks_a.beta)

        if any(!isfinite, F_k) || any(!isfinite, Λ_k)
            n_skipped += 1; continue
        end

        # ── Direction phase ────────────────────────────────────────────────
        wdt = dts[k] .* w_nodes   # precompute dt·w_q for this step
        @inbounds for idx in 1:MK
            val = 0.0
            for q in 1:n_q
                mul!(buf, N_mats[idx], view(F_k, :, q))
                val -= wdt[q] * dot(view(Λ_k, :, q), buf)
            end
            g[idx] += val
        end
    end

    n_skipped > 0 && @warn "gradient_adjoint_krylov!: $n_skipped / $T_steps time steps skipped"

    λ > 0 && (g .+= λ * sign.(θ_flat))
    return g
end

"""
    gradient_approx!(g, θ_flat, P, R, X, dt, dA, λ=0.0; ...)

First-order approximate gradient using the naive Fréchet approximation
of Didier et al. (2024):

    L_exp(A Δt, E) ≈ Δt · exp(A Δt) · E

This replaces the Gauss-Legendre quadrature over the path-vector integral
with the endpoint value, equivalent to assuming J and A commute.

The gradient becomes:
    ∂L/∂θ_{mk} ≈ -Δt Σ_t ⟨exp(Aᵀ Δt) r_t, A_{mk} p_t⟩

where r_t = P[t+1] - exp(A Δt) P[t].

Cost: 2T expv calls (same Krylov budget as the exact method) +
      MK·T sparse inner products (no quadrature overhead).

Note: this approximation is exact when A_{mk} commutes with A, i.e.
when the perturbation direction commutes with the generator. For CME
generators with random-like structure, Didier et al. show the error
is O(√(log n)/n) in the operator norm; for sparse structured CME
generators the error depends on the commutator ‖[A_{mk}, A]‖.
"""
function gradient_approx!(g, θ_flat, P, R, X, dt, dA, λ=0.0;
                           feature_map=quad_features, K=nothing,
                           m_krylov=25, tol=1e-8, W_vecs=nothing)
    N_species = length(first(X))
    if K === nothing
        K = feature_map === cubic_features ? cubic_feature_dimension(N_species) : feature_dimension(N_species)
    end
    θ = unflatten_params(θ_flat, R, K)
    A = build_generator(X, R, θ; feature_map=feature_map)
    At = A'

    fill!(g, 0.0)
    T_steps = length(P) - 1
    n       = size(A, 1)
    m_K     = min(m_krylov, n - 1)
    dts     = dt isa AbstractVector ? dt : fill(Float64(dt), T_steps)

    # Precompute u_t = exp(Aᵀ Δt) r_t for each time step.
    # r_t = P[t+1] - exp(A Δt) P[t]  (forward residual).
    U      = [zeros(n) for _ in 1:T_steps]
    active = falses(T_steps)

    for k in 1:T_steps
        dt_k   = dts[k]
        p_pred = try expv(dt_k, A,  P[k];  m=m_K, tol=tol) catch; nothing end
        (p_pred === nothing || any(!isfinite, p_pred)) && continue

        r_k = P[k+1] .- p_pred
        norm(r_k) < 1e-15 && continue

        u_k = try expv(dt_k, At, r_k; m=m_K, tol=tol) catch; nothing end
        (u_k === nothing || any(!isfinite, u_k)) && continue

        U[k]      = u_k
        active[k] = true
    end

    # Direction phase: g[j] = -Δt Σ_t ⟨u_t, dA[j] P[t]⟩
    buf = zeros(n)
    idx = 1
    for ζ in sort(collect(R))
        for f in 1:K
            N_mat = dA[ζ][f]
            val   = 0.0
            for k in 1:T_steps
                active[k] || continue
                mul!(buf, N_mat, P[k])
                val -= dts[k] * dot(U[k], buf)
            end
            g[idx] = val
            idx   += 1
        end
    end

    λ > 0 && (g .+= λ * sign.(θ_flat))
    return g
end

"""
    gradient_and_fisher!(g, D, θ_flat, P, R, X, dt, dA, λ=0.0; ...)

Compute gradient g and Gauss-Newton Fisher diagonal D simultaneously.
Shares all Krylov path vectors with the gradient computation — zero extra cost.

D[j] = dt² Σₖ Σᵢ wᵢ ‖dA[ζ][f] fᵢᵏ‖²  (j = linearized index of (ζ,f))

D is the diagonal of the Gauss-Newton Hessian: H ≈ JᵀJ where J is the
Jacobian of the CME residuals w.r.t. θ. Used for Fisher preconditioning:
rescale θ̃[j] = θ[j] × √D[j] so that all parameters have unit influence
on the dynamics, making the STLS threshold scale-invariant.
"""
function gradient_and_fisher!(g, D, θ_flat, P, R, X, dt, dA, λ=0.0;
                               feature_map=quad_features, K=nothing,
                               m_krylov=25, n_quadrature=8, tol=1e-8,
                               W_vecs=nothing)
    N_species = length(first(X))
    if K === nothing
        K = feature_map === cubic_features ? cubic_feature_dimension(N_species) : feature_dimension(N_species)
    end
    θ  = unflatten_params(θ_flat, R, K)
    A  = build_generator(X, R, θ; feature_map=feature_map)
    At = A'

    fill!(g, 0.0); fill!(D, 0.0)

    T_steps = length(P) - 1
    n       = size(A, 1)
    m_K     = min(m_krylov, n - 1)
    dts     = dt isa AbstractVector ? dt : fill(Float64(dt), T_steps)

    s_nodes, w_nodes = gauss_legendre_01(n_quadrature)
    n_q = length(s_nodes)

    F      = [zeros(n, n_q) for _ in 1:T_steps]
    Λ      = [zeros(n, n_q) for _ in 1:T_steps]
    active = falses(T_steps)

    for k in 1:T_steps
        dt_k = dts[k]

        ks_f = try arnoldi(A, P[k]; m=m_K, tol=tol) catch; nothing end
        ks_f === nothing && continue

        p_pred = _arnoldi_apply!(zeros(n), ks_f, dt_k)
        any(!isfinite, p_pred) && continue

        r_k  = P[k+1] .- p_pred
        ar_k = W_vecs !== nothing ? W_vecs[k] .* r_k : r_k
        norm(ar_k) > 1e-15 || continue

        ks_a = try arnoldi(At, ar_k; m=m_K, tol=tol) catch; nothing end
        ks_a === nothing && continue

        skip_step = false
        for q in 1:n_q
            fq = _arnoldi_apply!(zeros(n), ks_f, s_nodes[q] * dt_k)
            if any(!isfinite, fq); skip_step = true; break; end
            F[k][:, q] = fq

            lq = _arnoldi_apply!(zeros(n), ks_a, (1 - s_nodes[q]) * dt_k)
            if any(!isfinite, lq); skip_step = true; break; end
            Λ[k][:, q] = lq
        end
        skip_step && continue
        active[k] = true
    end

    buf = zeros(n)
    idx = 1
    for ζ in sort(collect(R))
        for f in 1:K
            N_mat = dA[ζ][f]
            gval  = 0.0; dval = 0.0
            for k in 1:T_steps
                active[k] || continue
                dt_k = dts[k]
                for q in 1:n_q
                    mul!(buf, N_mat, view(F[k], :, q))
                    w = w_nodes[q]
                    gval -= dt_k * w * dot(view(Λ[k], :, q), buf)
                    dval += dt_k^2 * w * dot(buf, buf)
                end
            end
            g[idx] = gval
            D[idx] = dval
            idx   += 1
        end
    end

    λ > 0 && (g .+= λ * sign.(θ_flat))
    return g, D
end

function gradient_adjoint_krylov(θ_flat, P, R, X, dt, dA, λ=0.0;
                                  feature_map=quad_features, K=nothing,
                                  m_krylov=25, n_quadrature=8, tol=1e-8,
                                  W_vecs=nothing)
    N_species = length(first(X))
    if K === nothing
        K = feature_map === cubic_features ? cubic_feature_dimension(N_species) : feature_dimension(N_species)
    end
    g = zeros(length(R) * K)
    gradient_adjoint_krylov!(g, θ_flat, P, R, X, dt, dA, λ;
                              feature_map=feature_map, K=K,
                              m_krylov=m_krylov, n_quadrature=n_quadrature, tol=tol,
                              W_vecs=W_vecs)
    return g
end
