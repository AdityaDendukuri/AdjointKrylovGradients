# =============================================================================
# Mass-action kinetics utilities for structure learning
# =============================================================================

"""
    mass_action_feature_index(ζ::NTuple{N,Int}; cubic=false) -> Int

Map a reaction stoichiometry to its corresponding mass-action feature index.

For mass-action kinetics, each reaction type has exactly one relevant feature:
- Zero-order (∅ → X): constant feature (index 1)
- Unimolecular (Xᵢ → ...): linear feature xᵢ (index 1+i)
- Homodimerization (2Xᵢ → ...): self-interaction xᵢ(xᵢ-1)/2 (index 1+N+i)
- Bimolecular (Xᵢ + Xⱼ → ...): cross-term xᵢxⱼ (index depends on i,j)

With `cubic=true`, also handles:
- Homotrimerization (3Xᵢ → ...): xᵢ(xᵢ-1)(xᵢ-2)/6
- Trimolecular mixed (2Xᵢ + Xⱼ → ...): xᵢ(xᵢ-1)/2 * xⱼ

# Example
```julia
ζ = (-1, -1, 1, 0)  # S + E → SE (bimolecular)
idx = mass_action_feature_index(ζ)  # Returns index for x₁×x₂ feature

# For trimolecular reactions like 2X + Y → 3X (net stoichiometry (1, -1))
ζ = (1, -1)
idx = mass_action_feature_index(ζ; cubic=true, reactants=(2, 1))
```
"""
function mass_action_feature_index(ζ::NTuple{N,Int}; cubic::Bool=false, reactants=nothing) where N
    # If explicit reactant stoichiometry is provided, use it
    # This handles cases where net stoichiometry doesn't determine reaction order
    # (e.g., X + Y → 2Y has net (-1, 1) but propensity ∝ XY, not just X)
    if reactants !== nothing
        return _index_from_reactants(reactants, N; cubic=cubic)
    end

    # Infer reactants from net stoichiometry
    reactant_species = findall(z -> z < 0, collect(ζ))
    counts = [-ζ[i] for i in reactant_species]

    if isempty(reactant_species)
        # Zero-order: constant feature
        return 1
    elseif length(reactant_species) == 1 && counts[1] == 1
        # Unimolecular: xᵢ
        return 1 + reactant_species[1]
    elseif length(reactant_species) == 1 && counts[1] == 2
        # Homodimerization: xᵢ(xᵢ-1)/2
        return 1 + N + reactant_species[1]
    elseif length(reactant_species) == 1 && counts[1] == 3 && cubic
        # Homotrimerization: xᵢ(xᵢ-1)(xᵢ-2)/6
        # In cubic features: after quad features, then N cubic self terms
        quad_dim = 1 + 2N + (N * (N - 1)) ÷ 2
        return quad_dim + reactant_species[1]
    elseif length(reactant_species) == 2 && counts[1] == 1 && counts[2] == 1
        # Bimolecular: xᵢxⱼ (i < j)
        i, j = minmax(reactant_species[1], reactant_species[2])
        # Offset: 1 (const) + N (linear) + N (self) + triangular index
        offset = 1 + 2N
        for a in 1:i-1
            offset += N - a
        end
        return offset + (j - i)
    elseif length(reactant_species) == 2 && cubic
        # Trimolecular mixed: one species with count 2, one with count 1
        if counts[1] == 2 && counts[2] == 1
            # Species i has count 2, species j has count 1
            return _cubic_mixed_index(reactant_species[1], reactant_species[2], N)
        elseif counts[1] == 1 && counts[2] == 2
            return _cubic_mixed_index(reactant_species[2], reactant_species[1], N)
        else
            error("Unsupported trimolecular stoichiometry: $ζ")
        end
    else
        error("Unsupported reaction order for stoichiometry: $ζ")
    end
end

"""
Compute index for xᵢ(xᵢ-1)/2 * xⱼ term in cubic features.
"""
function _cubic_mixed_index(i_double::Int, j_single::Int, N::Int)
    # Cubic features order after quad features:
    # - N cubic self: xᵢ(xᵢ-1)(xᵢ-2)/6 for i = 1..N
    # - N(N-1) cubic mixed: xᵢ(xᵢ-1)/2 * xⱼ for all i ≠ j
    #   Ordered as: (i=1,j=2), (i=1,j=3),..., (i=2,j=1), (i=2,j=3),...
    quad_dim = 1 + 2N + (N * (N - 1)) ÷ 2
    cubic_self_offset = quad_dim + N

    # Index within cubic mixed: (i-1)*(N-1) + (j - (j > i ? 1 : 0))
    idx_in_mixed = (i_double - 1) * (N - 1)
    for jj in 1:N
        if jj != i_double
            idx_in_mixed += 1
            if jj == j_single
                return cubic_self_offset + idx_in_mixed
            end
        end
    end
    error("Should not reach here")
end

"""
Compute feature index from explicit reactant stoichiometry.
Handles orders 0-2 for quadratic features, orders 0-3 for cubic features.
"""
function _index_from_reactants(reactants::NTuple{N,Int}, N_species::Int; cubic::Bool=false) where N
    # reactants[i] = number of molecules of species i consumed
    total_order = sum(reactants)

    if total_order == 0
        return 1  # constant
    elseif total_order == 1
        # Unimolecular
        i = findfirst(r -> r == 1, collect(reactants))
        return 1 + i
    elseif total_order == 2
        nz = findall(r -> r > 0, collect(reactants))
        if length(nz) == 1
            # Homodimerization
            return 1 + N_species + nz[1]
        else
            # Bimolecular
            i, j = minmax(nz[1], nz[2])
            offset = 1 + 2N_species
            for a in 1:i-1
                offset += N_species - a
            end
            return offset + (j - i)
        end
    elseif total_order == 3
        if !cubic
            error("Trimolecular reactions require cubic=true: $reactants")
        end
        nz = findall(r -> r > 0, collect(reactants))
        counts = [reactants[i] for i in nz]

        if length(nz) == 1 && counts[1] == 3
            # Homotrimerization
            quad_dim = 1 + 2N_species + (N_species * (N_species - 1)) ÷ 2
            return quad_dim + nz[1]
        elseif length(nz) == 2
            # Find which has count 2 and which has count 1
            if counts[1] == 2 && counts[2] == 1
                return _cubic_mixed_index(nz[1], nz[2], N_species)
            elseif counts[1] == 1 && counts[2] == 2
                return _cubic_mixed_index(nz[2], nz[1], N_species)
            end
        elseif length(nz) == 3
            # Triple: xᵢxⱼxₖ for i < j < k
            i, j, k = sort(nz)
            quad_dim = 1 + 2N_species + (N_species * (N_species - 1)) ÷ 2
            cubic_self_offset = quad_dim + N_species
            cubic_mixed_offset = cubic_self_offset + N_species * (N_species - 1)
            # Compute triangular index for triple
            idx = 0
            for a in 1:i-1
                for b in a+1:N_species-1
                    idx += N_species - b
                end
            end
            for b in i+1:j-1
                idx += N_species - b
            end
            idx += k - j
            return cubic_mixed_offset + idx
        end
    end
    error("Unsupported reactant stoichiometry: $reactants")
end

"""
    generate_candidate_reactions(N::Int; include_birth=true, include_death=true,
                                  include_conversion=true, include_bimolecular=true,
                                  include_catalytic=true) -> Vector

Generate a pool of candidate reactions for N species.

Returns stoichiometry tuples for:
- Birth: ∅ → Xᵢ
- Death: Xᵢ → ∅
- Conversion: Xᵢ → Xⱼ (unimolecular, one product)
- Catalytic: Xᵢ → Xⱼ + Xₖ (unimolecular, two products - e.g., enzyme catalysis)
- Bimolecular binding: Xᵢ + Xⱼ → Xₖ
- Bimolecular unbinding: Xₖ → Xᵢ + Xⱼ

# Arguments
- `N`: Number of species
- `include_birth`: Include zero-order production reactions
- `include_death`: Include first-order degradation reactions
- `include_conversion`: Include unimolecular conversion reactions
- `include_bimolecular`: Include bimolecular reactions
- `include_catalytic`: Include catalytic reactions (Xᵢ → Xⱼ + Xₖ)

# Example
```julia
R_candidates = generate_candidate_reactions(4)  # For S, E, SE, P system
```
"""
function generate_candidate_reactions(N::Int;
                                       include_birth=true,
                                       include_death=true,
                                       include_conversion=true,
                                       include_bimolecular=true,
                                       include_catalytic=true)
    R = NTuple{N,Int}[]

    # Birth: ∅ → Xᵢ
    if include_birth
        for i in 1:N
            ζ = ntuple(j -> j == i ? 1 : 0, N)
            push!(R, ζ)
        end
    end

    # Death: Xᵢ → ∅
    if include_death
        for i in 1:N
            ζ = ntuple(j -> j == i ? -1 : 0, N)
            push!(R, ζ)
        end
    end

    # Conversion: Xᵢ → Xⱼ (i ≠ j)
    if include_conversion
        for i in 1:N, j in 1:N
            if i != j
                ζ = ntuple(k -> k == i ? -1 : (k == j ? 1 : 0), N)
                push!(R, ζ)
            end
        end
    end

    # Catalytic: Xᵢ → Xⱼ + Xₖ (one reactant, two different products)
    # This covers reactions like SE → P + E
    if include_catalytic
        for i in 1:N, j in 1:N, k in j:N
            if j != i && k != i && (j != k || j != i)
                # Xᵢ → Xⱼ + Xₖ
                ζ = ntuple(l -> begin
                    if l == i
                        -1
                    elseif l == j && l == k
                        2  # Same product twice
                    elseif l == j || l == k
                        1
                    else
                        0
                    end
                end, N)
                if any(z != 0 for z in ζ)
                    push!(R, ζ)
                end
            end
        end
    end

    # Bimolecular: Xᵢ + Xⱼ → Xₖ (binding) and Xₖ → Xᵢ + Xⱼ (unbinding)
    if include_bimolecular
        for i in 1:N, j in i:N, k in 1:N
            # Binding: Xᵢ + Xⱼ → Xₖ
            if i == j
                # Homodimerization: 2Xᵢ → Xₖ
                ζ = ntuple(l -> l == i ? -2 : (l == k ? 1 : 0), N)
            else
                # Heterodimerization: Xᵢ + Xⱼ → Xₖ
                ζ = ntuple(l -> (l == i || l == j) ? -1 : (l == k ? 1 : 0), N)
            end
            # Avoid trivial reactions
            if any(z != 0 for z in ζ)
                push!(R, ζ)
            end

            # Unbinding: Xₖ → Xᵢ + Xⱼ (reverse)
            if i == j
                ζ_rev = ntuple(l -> l == i ? 2 : (l == k ? -1 : 0), N)
            else
                ζ_rev = ntuple(l -> (l == i || l == j) ? 1 : (l == k ? -1 : 0), N)
            end
            if any(z != 0 for z in ζ_rev)
                push!(R, ζ_rev)
            end
        end
    end

    # Remove duplicates and sort
    return sort(unique(R))
end

"""
    mass_action_initialization(R_candidates, K::Int; init_scale=0.01) -> Dict

Initialize parameters for candidate reactions with mass-action constraint.

Each θ[ζ] is a K-dimensional vector with only the mass-action feature nonzero.
This effectively gives one scalar rate per reaction.

# Arguments
- `R_candidates`: Vector of candidate reaction stoichiometries
- `K`: Feature dimension (from feature_dimension(N))
- `init_scale`: Scale for initial nonzero values

# Returns
- Dict mapping stoichiometry → parameter vector
"""
function mass_action_initialization(R_candidates, K::Int; init_scale=0.01)
    N = length(first(R_candidates))
    θ = Dict{eltype(R_candidates), Vector{Float64}}()

    for ζ in R_candidates
        θ_ζ = zeros(K)
        try
            idx = mass_action_feature_index(ζ)
            θ_ζ[idx] = init_scale
        catch
            # Unsupported reaction type - leave as zeros
        end
        θ[ζ] = θ_ζ
    end

    return θ
end

"""
    extract_mass_action_rates(θ::Dict; cubic=false, reactant_stoich=nothing) -> Dict{stoichiometry, Float64}

Extract scalar mass-action rates from parameter dictionary.

For each reaction, returns the value at its mass-action feature index.

# Arguments
- `θ`: Parameter dictionary mapping stoichiometry to feature vectors
- `cubic`: Use cubic feature indexing (for trimolecular reactions)
- `reactant_stoich`: Dict mapping net stoichiometry to reactant stoichiometry.
  Required for reactions where net stoichiometry doesn't determine the reaction order
  (e.g., 2X + Y → 3X has net (1,-1) but uses X²Y/2 feature).

# Example
```julia
# For Brusselator with trimolecular reaction 2X + Y → 3X
reactant_stoich = Dict(
    (1, 0)  => (0, 0),    # ∅ → X
    (1, -1) => (2, 1),    # 2X + Y → 3X (2 of X, 1 of Y consumed)
    (-1, 1) => (1, 0),    # X → Y
    (-1, 0) => (1, 0)     # X → ∅
)
rates = extract_mass_action_rates(θ_opt; cubic=true, reactant_stoich=reactant_stoich)
```
"""
function extract_mass_action_rates(θ::Dict; cubic::Bool=false, reactant_stoich=nothing)
    rates = Dict{keytype(θ), Float64}()
    for (ζ, θ_ζ) in θ
        try
            # Use reactant_stoich when provided, regardless of cubic setting
            # This handles cases like X+Y→2Y where net stoich (-1,1) differs from reactants (1,1)
            if reactant_stoich !== nothing && haskey(reactant_stoich, ζ)
                idx = mass_action_feature_index(ζ; cubic=cubic, reactants=reactant_stoich[ζ])
            else
                idx = mass_action_feature_index(ζ; cubic=cubic)
            end
            rates[ζ] = θ_ζ[idx]
        catch
            rates[ζ] = 0.0
        end
    end
    return rates
end

"""
    reaction_description(ζ::NTuple{N,Int}; species_names=nothing) -> String

Generate human-readable description of a reaction from its stoichiometry.

# Example
```julia
ζ = (-1, -1, 1, 0)
desc = reaction_description(ζ; species_names=["S", "E", "SE", "P"])
# Returns: "S + E → SE"
```
"""
function reaction_description(ζ::NTuple{N,Int}; species_names=nothing) where N
    if species_names === nothing
        species_names = ["X$i" for i in 1:N]
    end

    reactants = String[]
    products = String[]

    for i in 1:N
        if ζ[i] < 0
            count = -ζ[i]
            if count == 1
                push!(reactants, species_names[i])
            else
                push!(reactants, "$(count)$(species_names[i])")
            end
        elseif ζ[i] > 0
            count = ζ[i]
            if count == 1
                push!(products, species_names[i])
            else
                push!(products, "$(count)$(species_names[i])")
            end
        end
    end

    lhs = isempty(reactants) ? "∅" : join(reactants, " + ")
    rhs = isempty(products) ? "∅" : join(products, " + ")

    return "$lhs → $rhs"
end
