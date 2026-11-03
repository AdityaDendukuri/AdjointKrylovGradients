# =============================================================================
# Core mathematical primitives for CME inference
# =============================================================================

"""
    quad_features(x::NTuple{N,Int}) -> Vector{Float64}

Compute quadratic feature vector for mass-action kinetics.

For a state `x = (x₁, x₂, ..., xₙ)`, returns features:
- `1` (constant)
- `xᵢ` for i = 1:N (linear terms - unimolecular)
- `xᵢ(xᵢ-1)/2` for i = 1:N (self-interaction)
- `xᵢxⱼ` for i < j (bimolecular cross-terms)

# Arguments
- `x`: N-tuple of integer molecule counts

# Returns
- Feature vector of length `K = 1 + 2N + N(N-1)/2`

# Example
```julia
x = (50, 10, 0, 0)  # S=50, E=10, SE=0, P=0
φ = quad_features(x)  # 15-dimensional feature vector
```
"""
function quad_features(x::NTuple{N,Int}) where {N}
    K = feature_dimension(N)
    φ = zeros(K)
    k = 1

    # Constant term
    φ[k] = 1.0
    k += 1

    # Linear terms (species counts)
    for i in 1:N
        φ[k] = Float64(x[i])
        k += 1
    end

    # Quadratic self-interaction: xᵢ(xᵢ-1)/2
    for i in 1:N
        φ[k] = Float64(x[i] * (x[i] - 1)) / 2.0
        k += 1
    end

    # Bimolecular cross-terms: xᵢxⱼ
    for i in 1:N-1, j in i+1:N
        φ[k] = Float64(x[i] * x[j])
        k += 1
    end

    return φ
end

"""
    feature_dimension(N::Int) -> Int

Compute the dimension of the quadratic feature space for N species.

Returns `K = 1 + 2N + N(N-1)/2`.
"""
function feature_dimension(N::Int)
    return 1 + 2N + (N * (N - 1)) ÷ 2
end

"""
    feature_names(N::Int) -> Vector{String}

Generate human-readable names for quadratic features with N species.

# Example
```julia
names = feature_names(4)  # For S, E, SE, P system
# Returns: ["1", "x₁", "x₂", "x₃", "x₄", "x₁²/2", ..., "x₁×x₂", ...]
```
"""
function feature_names(N::Int; species_names::Vector{String}=String[])
    if isempty(species_names)
        species_names = ["x$i" for i in 1:N]
    end

    names = String[]

    # Constant
    push!(names, "1")

    # Linear
    for i in 1:N
        push!(names, species_names[i])
    end

    # Quadratic self
    for i in 1:N
        push!(names, "$(species_names[i])²/2")
    end

    # Bimolecular
    for i in 1:N-1, j in i+1:N
        push!(names, "$(species_names[i])×$(species_names[j])")
    end

    return names
end

# =============================================================================
# Cubic features for trimolecular reactions
# =============================================================================

"""
    cubic_feature_dimension(N::Int) -> Int

Compute the dimension of the cubic feature space for N species.

Includes:
- 1 (constant)
- N (linear)
- N (quadratic self: xᵢ(xᵢ-1)/2)
- N(N-1)/2 (quadratic cross: xᵢxⱼ)
- N (cubic self: xᵢ(xᵢ-1)(xᵢ-2)/6)
- N(N-1) (cubic mixed: xᵢ(xᵢ-1)/2 * xⱼ for i≠j)
- C(N,3) (cubic triple: xᵢxⱼxₖ for i<j<k)
"""
function cubic_feature_dimension(N::Int)
    quad_dim = 1 + 2N + (N * (N - 1)) ÷ 2
    cubic_self = N  # xᵢ(xᵢ-1)(xᵢ-2)/6
    cubic_mixed = N * (N - 1)  # xᵢ(xᵢ-1)/2 * xⱼ
    cubic_triple = (N * (N - 1) * (N - 2)) ÷ 6  # xᵢxⱼxₖ
    return quad_dim + cubic_self + cubic_mixed + cubic_triple
end

"""
    cubic_features(x::NTuple{N,Int}) -> Vector{Float64}

Compute cubic feature vector for mass-action kinetics with trimolecular reactions.

For a state `x = (x₁, x₂, ..., xₙ)`, returns features:
- `1` (constant)
- `xᵢ` for i = 1:N (linear - unimolecular)
- `xᵢ(xᵢ-1)/2` for i = 1:N (dimerization)
- `xᵢxⱼ` for i < j (bimolecular cross-terms)
- `xᵢ(xᵢ-1)(xᵢ-2)/6` for i = 1:N (trimerization)
- `xᵢ(xᵢ-1)/2 * xⱼ` for i ≠ j (2+1 trimolecular, e.g., 2X + Y → products)
- `xᵢxⱼxₖ` for i < j < k (1+1+1 trimolecular)

# Arguments
- `x`: N-tuple of integer molecule counts

# Returns
- Feature vector of length `cubic_feature_dimension(N)`

# Example
```julia
x = (5, 3)  # X=5, Y=3
φ = cubic_features(x)
# Includes X²Y/2 term needed for 2X + Y → 3X reaction
```
"""
function cubic_features(x::NTuple{N,Int}) where {N}
    K = cubic_feature_dimension(N)
    φ = zeros(K)
    k = 1

    # Constant term
    φ[k] = 1.0
    k += 1

    # Linear terms (species counts)
    for i in 1:N
        φ[k] = Float64(x[i])
        k += 1
    end

    # Quadratic self-interaction: xᵢ(xᵢ-1)/2
    for i in 1:N
        φ[k] = Float64(x[i] * (x[i] - 1)) / 2.0
        k += 1
    end

    # Bimolecular cross-terms: xᵢxⱼ
    for i in 1:N-1, j in i+1:N
        φ[k] = Float64(x[i] * x[j])
        k += 1
    end

    # Cubic self-interaction: xᵢ(xᵢ-1)(xᵢ-2)/6
    for i in 1:N
        φ[k] = Float64(x[i] * (x[i] - 1) * (x[i] - 2)) / 6.0
        k += 1
    end

    # Cubic mixed (2+1): xᵢ(xᵢ-1)/2 * xⱼ for all i, j where i ≠ j
    # This gives us X²Y/2 type terms needed for 2X + Y → 3X
    for i in 1:N
        for j in 1:N
            if i != j
                φ[k] = Float64(x[i] * (x[i] - 1) * x[j]) / 2.0
                k += 1
            end
        end
    end

    # Cubic triple: xᵢxⱼxₖ for i < j < k
    for i in 1:N-2
        for j in i+1:N-1
            for l in j+1:N
                φ[k] = Float64(x[i] * x[j] * x[l])
                k += 1
            end
        end
    end

    return φ
end

"""
    cubic_feature_names(N::Int; species_names::Vector{String}=String[]) -> Vector{String}

Generate human-readable names for cubic features with N species.
"""
function cubic_feature_names(N::Int; species_names::Vector{String}=String[])
    if isempty(species_names)
        species_names = ["x$i" for i in 1:N]
    end

    names = String[]

    # Constant
    push!(names, "1")

    # Linear
    for i in 1:N
        push!(names, species_names[i])
    end

    # Quadratic self
    for i in 1:N
        push!(names, "$(species_names[i])²/2")
    end

    # Bimolecular cross
    for i in 1:N-1, j in i+1:N
        push!(names, "$(species_names[i])×$(species_names[j])")
    end

    # Cubic self
    for i in 1:N
        push!(names, "$(species_names[i])³/6")
    end

    # Cubic mixed (2+1)
    for i in 1:N
        for j in 1:N
            if i != j
                push!(names, "$(species_names[i])²$(species_names[j])/2")
            end
        end
    end

    # Cubic triple
    for i in 1:N-2
        for j in i+1:N-1
            for l in j+1:N
                push!(names, "$(species_names[i])×$(species_names[j])×$(species_names[l])")
            end
        end
    end

    return names
end

"""
    build_generator(X, R, θ; feature_map=quad_features) -> SparseMatrixCSC

Build the CME generator matrix A(θ) for given state space and parameters.

The generator matrix satisfies `dp/dt = A p` where `p` is the probability
distribution. Off-diagonal entries `A[i,j]` represent transition rates from
state `j` to state `i`. Diagonal entries are negative sums of outgoing rates.

# Arguments
- `X`: Vector of state tuples defining the state space
- `R`: Vector of stoichiometric vectors (reaction types)
- `θ`: Dict mapping stoichiometry ζ to parameter vector θ[ζ]
- `feature_map`: Feature function (default: quad_features, can use cubic_features)

# Returns
- Sparse generator matrix of size `(|X|, |X|)`

# Mathematical Form
For each reaction ζ ∈ R:
- `A[i,j] = max(0, θ[ζ]ᵀφ(xⱼ))` if `xⱼ + ζ = xᵢ`
- `A[j,j] -= max(0, θ[ζ]ᵀφ(xⱼ))` for conservation

# Example
```julia
X = [(0,0), (1,0), (0,1), (1,1)]
R = [(1,0), (-1,0)]
θ = Dict((1,0) => [0.1, 0.0], (-1,0) => [0.0, 0.05])
A = build_generator(X, R, θ)  # uses quadratic features
A = build_generator(X, R, θ; feature_map=cubic_features)  # uses cubic features
```
"""
function build_generator(X, R, θ; feature_map=quad_features)
    N = length(first(X))
    n = length(X)
    idx = Dict(x => i for (i, x) in enumerate(X))

    I_vec = Int[]
    J_vec = Int[]
    V_vec = Float64[]
    diag = zeros(n)

    for (j, x_j) in enumerate(X)
        φ_j = feature_map(x_j)

        for ζ in R
            x_i = ntuple(k -> x_j[k] + ζ[k], N)
            i = get(idx, x_i, 0)

            if i > 0
                rate = max(0.0, dot(θ[ζ], φ_j))
                push!(I_vec, i)
                push!(J_vec, j)
                push!(V_vec, rate)
                diag[j] -= rate
            end
        end
    end

    return sparse(I_vec, J_vec, V_vec, n, n) + spdiagm(diag)
end

"""
    compute_residuals(P, R, θ, X, dt; feature_map=quad_features) -> Vector{Vector{Float64}}

Compute residuals between observed and predicted distributions.

For each time step k, computes:
    `rₖ = P[k+1] - exp(A(θ) dt) P[k]`

# Arguments
- `P`: Vector of probability distributions at observation times
- `R`: Vector of stoichiometric vectors
- `θ`: Parameter dictionary
- `X`: State space
- `dt`: Time step between observations
- `feature_map`: Feature function (default: quad_features)

# Returns
- Vector of residual vectors for each time step
"""
function compute_residuals(P, R, θ, X, dt; feature_map=quad_features)
    A = build_generator(X, R, θ; feature_map=feature_map)

    P_init = P[1:end-1]
    P_target = P[2:end]

    P_pred = [expv(dt, A, p) for p in P_init]

    return [P_target[i] - P_pred[i] for i in eachindex(P_pred)]
end
