# =============================================================================
# Gradient computation via Fréchet derivatives
# =============================================================================

"""
    BlockOp{TM,TN}

Block matrix operator for computing Fréchet derivatives of matrix exponentials.

Represents the 2n × 2n block matrix:
```
[M  N]
[0  M]
```

Used internally by `frechet_action` for efficient computation without
explicitly forming the full block matrix.
"""
struct BlockOp{TM,TN}
    M::TM
    N::TN
    n::Int
end

Base.size(B::BlockOp) = (2B.n, 2B.n)
Base.size(B::BlockOp, d::Int) = 2B.n
Base.eltype(::BlockOp) = Float64
LinearAlgebra.ishermitian(::BlockOp) = false

function LinearAlgebra.mul!(out, B::BlockOp, v)
    x = view(v, 1:B.n)
    y = view(v, B.n+1:2B.n)
    ox = view(out, 1:B.n)
    oy = view(out, B.n+1:2B.n)

    mul!(ox, B.M, x)
    mul!(oy, B.M, y)
    mul!(ox, B.N, y, 1.0, 1.0)

    return out
end

"""
    frechet_action(M, N, v) -> Vector{Float64}

Compute the Fréchet derivative action: `L_exp(M, N) v`.

This computes the directional derivative of exp(M) in direction N,
applied to vector v. Uses Krylov-based matrix exponential for efficiency.

# Mathematical Definition
```
L_exp(M, N) v = [exp(B)]₁₁ where B = [M N; 0 M]
```
evaluated on [0; v].

# Arguments
- `M`: Base matrix
- `N`: Direction matrix
- `v`: Vector to apply the derivative to

# Returns
- Result vector of same length as v
"""
function frechet_action(M, N, v)
    B = BlockOp(M, N, length(v))
    w = expv(1.0, B, vcat(zeros(length(v)), v); m=15, tol=1e-8)
    return w[1:length(v)]
end

"""
    build_derivatives(X, R; feature_map=quad_features, K=nothing) -> Dict

Precompute derivative matrices ∂A/∂θ_{ζ,f} for all reactions and features.

This avoids recomputing sparse structure during optimization by caching
all |R| × K derivative matrices.

# Arguments
- `X`: State space (vector of tuples)
- `R`: Reaction stoichiometries (vector of tuples)
- `feature_map`: Feature function (default: quad_features, can use cubic_features)
- `K`: Feature dimension (auto-computed if not provided)

# Returns
- Dict mapping each ζ ∈ R to a vector of K sparse matrices,
  where `dA[ζ][f]` is ∂A/∂θ_{ζ,f}

# Mathematical Form
For reaction ζ and feature f:
```
∂A/∂θ_{ζ,f}[i,j] = φ_f(x_j)  if x_j + ζ = x_i
∂A/∂θ_{ζ,f}[j,j] = -φ_f(x_j)  (conservation)
```
"""
function build_derivatives(X, R; feature_map=quad_features, K=nothing)
    N = length(first(X))
    n = length(X)
    if K === nothing
        K = length(feature_map(first(X)))
    end
    idx = Dict(x => i for (i, x) in enumerate(X))

    dA = Dict{eltype(R),Vector{SparseMatrixCSC{Float64,Int}}}()

    for ζ in R
        I_list = [Int[] for _ in 1:K]
        J_list = [Int[] for _ in 1:K]
        V_list = [Float64[] for _ in 1:K]

        for (j, x_j) in enumerate(X)
            φ_j = feature_map(x_j)
            x_i = ntuple(k -> x_j[k] + ζ[k], N)
            i = get(idx, x_i, 0)

            if i > 0
                for f in 1:K
                    if abs(φ_j[f]) > 1e-15
                        # Off-diagonal: transition rate
                        push!(I_list[f], i)
                        push!(J_list[f], j)
                        push!(V_list[f], φ_j[f])
                        # Diagonal: conservation
                        push!(I_list[f], j)
                        push!(J_list[f], j)
                        push!(V_list[f], -φ_j[f])
                    end
                end
            end
        end

        dA[ζ] = [sparse(I_list[f], J_list[f], V_list[f], n, n) for f in 1:K]
    end

    return dA
end

"""
    flatten_params(θ::Dict) -> Vector{Float64}

Flatten parameter dictionary to a vector for optimization.

Parameters are ordered by sorted reaction stoichiometries.

# Example
```julia
θ = Dict((1,0) => [0.1, 0.2], (-1,0) => [0.3, 0.4])
v = flatten_params(θ)  # [0.3, 0.4, 0.1, 0.2] (sorted by reaction)
```
"""
function flatten_params(θ::Dict)
    v = Float64[]
    for ζ in sort(collect(keys(θ)))
        append!(v, θ[ζ])
    end
    return v
end

"""
    unflatten_params(v, R, K) -> Dict

Reconstruct parameter dictionary from flat vector.

# Arguments
- `v`: Flat parameter vector
- `R`: Reaction stoichiometries (for keys)
- `K`: Feature dimension

# Example
```julia
v = [0.3, 0.4, 0.1, 0.2]
R = [(-1,0), (1,0)]
θ = unflatten_params(v, R, 2)
```
"""
function unflatten_params(v, R, K)
    θ = Dict{eltype(R),Vector{Float64}}()
    i = 1
    for ζ in sort(collect(R))
        θ[ζ] = v[i:i+K-1]
        i += K
    end
    return θ
end

"""
    gradient!(g, θ_flat, P, R, X, dt, dA, λ=0.0; feature_map=quad_features, K=nothing)

Compute gradient of the objective function in-place.

Uses Fréchet derivatives of matrix exponentials for efficient backpropagation
through the CME forward model.

# Arguments
- `g`: Pre-allocated gradient vector (modified in-place)
- `θ_flat`: Flat parameter vector
- `P`: Vector of probability distributions
- `R`: Reaction stoichiometries
- `X`: State space
- `dt`: Time step
- `dA`: Precomputed derivative matrices from `build_derivatives`
- `λ`: L1 regularization strength (default: 0.0)
- `feature_map`: Feature function (default: quad_features)
- `K`: Feature dimension (auto-computed if not provided)

# Mathematical Form
```
∂L/∂θ = Σₖ 2(P̂ₖ₊₁ - Pₖ₊₁)ᵀ L_exp(A dt, ∂A/∂θ dt) Pₖ + λ sign(θ)
```
where P̂ₖ₊₁ = exp(A dt) Pₖ is the prediction.
"""
function gradient!(g, θ_flat, P, R, X, dt, dA, λ=0.0; feature_map=quad_features, K=nothing)
    N = length(first(X))
    if K === nothing
        K = length(feature_map(first(X)))
    end
    θ = unflatten_params(θ_flat, R, K)
    A = build_generator(X, R, θ; feature_map=feature_map)

    fill!(g, 0.0)

    # Chain rule through each time step
    for k in 1:length(P)-1
        # Forward prediction
        p_pred = expv(dt, A, P[k]; m=15, tol=1e-8)

        # Residual gradient direction
        u = p_pred - P[k+1]

        # Backpropagate through Fréchet derivative
        idx = 1
        for ζ in sort(collect(R))
            for f in 1:K
                s = frechet_action(A * dt, dA[ζ][f] * dt, P[k])
                g[idx] += dot(u, s)
                idx += 1
            end
        end
    end

    # L1 regularization subgradient
    if λ > 0
        g .+= λ * sign.(θ_flat)
    end

    return g
end

"""
    objective(P, R, θ, X, dt, λ=0.0; feature_map=quad_features) -> Float64

Compute the objective function value.

# Mathematical Form
```
L(θ) = (1/2) Σₖ ‖Pₖ₊₁ - exp(A(θ) dt) Pₖ‖² + λ ‖θ‖₁
```

# Arguments
- `P`: Vector of probability distributions
- `R`: Reaction stoichiometries
- `θ`: Parameter dictionary
- `X`: State space
- `dt`: Time step
- `λ`: L1 regularization strength (default: 0.0)
- `feature_map`: Feature function (default: quad_features)

# Returns
- Scalar objective value
"""
function objective(P, R, θ, X, dt, λ=0.0; feature_map=quad_features, W_vecs=nothing,
                   m_krylov=25, tol=1e-10)
    A = build_generator(X, R, θ; feature_map=feature_map)
    T_steps = length(P) - 1
    dts = dt isa AbstractVector ? dt : fill(Float64(dt), T_steps)

    obj = 0.0
    for k in 1:T_steps
        # The forward map here must be at least as accurate as the one implied by
        # the gradient, or L-BFGS minimizes a surrogate its gradients do not
        # describe.  This used to be fixed at m=15, tol=1e-8 while gradients ran
        # at m=25, leaving the objective roughly 30x less accurate than them.
        p_pred = try expv(dts[k], A, P[k]; m=m_krylov, tol=tol) catch; nothing end
        (p_pred === nothing || any(!isfinite, p_pred)) && return 1e10
        r = P[k+1] - p_pred
        obj += W_vecs !== nothing ? 0.5 * dot(r, W_vecs[k] .* r) : 0.5 * dot(r, r)
    end

    # L1 regularization
    if λ > 0
        θ_flat = flatten_params(θ)
        obj += λ * sum(abs, θ_flat)
    end

    return obj
end
