# =============================================================================
# Empirical initialization from trajectory data
# =============================================================================

"""
    TransitionData

Container for transition statistics from SSA trajectories.

# Fields
- `counts`: Dict mapping stoichiometry to transition count
- `exposure`: Dict mapping states to total time exposure
"""
struct TransitionData{N}
    counts::Dict{NTuple{N,Int},Int}
    exposure::Dict{NTuple{N,Int},Float64}
end

"""
    count_transitions(trajectories, R; t_start=nothing, t_end=nothing) -> TransitionData

Count reaction transitions in SSA trajectory data.

Iterates directly over jump times in trajectories to count how many times
each reaction type occurs, along with exact exposure time in each state.

# Arguments
- `trajectories`: Vector of SSA trajectory solutions
- `R`: Reaction stoichiometries to count
- `t_start`: Start of time window (default: beginning of trajectories)
- `t_end`: End of time window (default: end of trajectories)

# Returns
- `TransitionData` with counts and exposure statistics

# Example
```julia
R = [(-1,-1,1,0), (1,1,-1,0), (0,1,-1,1)]
data = count_transitions(trajectories, R; t_start=0.0, t_end=1.0)
```
"""
function count_transitions(trajectories, R; t_start=nothing, t_end=nothing, verbose=false)
    N = _infer_species_count(trajectories)

    transition_counts = Dict{NTuple{N,Int},Int}(ζ => 0 for ζ in R)
    state_exposure = Dict{NTuple{N,Int},Float64}()

    verbose && println("Counting transitions in $(length(trajectories)) trajectories...")

    for traj in trajectories
        times = traj.t
        states = traj.u

        # Determine time window
        traj_t_start = t_start === nothing ? times[1] : max(t_start, times[1])
        traj_t_end = t_end === nothing ? times[end] : min(t_end, times[end])

        if traj_t_start >= traj_t_end
            continue
        end

        # Find index range within time window
        i_start = searchsortedfirst(times, traj_t_start)
        i_end = searchsortedlast(times, traj_t_end)

        # Iterate over jumps directly
        for i in i_start:i_end-1
            x_curr = ntuple(j -> Int(states[i][j]), N)
            x_next = ntuple(j -> Int(states[i+1][j]), N)

            # Exposure time is exact time spent in this state
            dt = times[i+1] - times[i]
            state_exposure[x_curr] = get(state_exposure, x_curr, 0.0) + dt

            # Count transition
            ζ = ntuple(j -> x_next[j] - x_curr[j], N)
            if haskey(transition_counts, ζ)
                transition_counts[ζ] += 1
            end
        end

        # Handle exposure for final state up to t_end
        if i_end >= i_start
            x_final = ntuple(j -> Int(states[i_end][j]), N)
            dt_final = traj_t_end - times[i_end]
            if dt_final > 0
                state_exposure[x_final] = get(state_exposure, x_final, 0.0) + dt_final
            end
        end
    end

    if verbose
        println("\nTransition counts:")
        for ζ in sort(collect(R))
            println("  $ζ: $(transition_counts[ζ]) transitions")
        end
        println("\nUnique states visited: $(length(state_exposure))")
    end

    return TransitionData{N}(transition_counts, state_exposure)
end

"""
    empirical_initialization(transition_data::TransitionData, R;
                             feature_map=nothing, verbose=false) -> Dict

Compute initial parameter estimates from transition statistics.

Uses weighted exposure to estimate rate constants. For each reaction,
identifies the relevant feature (based on mass-action kinetics) and
computes:
    rate_estimate = transition_count / weighted_exposure

# Arguments
- `transition_data`: Output from `count_transitions`
- `R`: Reaction stoichiometries
- `feature_map`: Optional custom feature function (default: quad_features)
- `verbose`: Print diagnostic information

# Returns
- Dict mapping stoichiometry to parameter vector

# Note
This function uses heuristics based on standard mass-action kinetics.
For non-standard rate laws, custom initialization may be needed.

# Example
```julia
data = count_transitions(trajectories, R)
θ_init = empirical_initialization(data, R; verbose=true)
```
"""
function empirical_initialization(transition_data::TransitionData{N}, R;
                                  feature_map=nothing, verbose=false,
                                  noise_scale=0.1, reactant_stoich=nothing,
                                  noise_model=:lognormal, rng=Random.default_rng()) where {N}
    if feature_map === nothing
        feature_map = quad_features
    end

    counts = transition_data.counts
    exposure = transition_data.exposure

    K = feature_map === cubic_features ? cubic_feature_dimension(N) : feature_dimension(N)
    θ_emp = Dict{NTuple{N,Int},Vector{Float64}}()

    verbose && println("\n" * "="^70)
    verbose && println("EMPIRICAL INITIALIZATION")
    verbose && println("="^70)

    use_cubic = feature_map === cubic_features

    for ζ in R
        θ = zeros(K)
        total_count = get(counts, ζ, 0)

        # Identify the likely feature for this reaction
        # If explicit reactant stoichiometry is provided, use it
        reactants = reactant_stoich !== nothing ? get(reactant_stoich, ζ, nothing) : nothing
        feature_idx, feature_desc = if reactants !== nothing
            idx = mass_action_feature_index(ζ; cubic=use_cubic, reactants=reactants)
            idx, "explicit reactants"
        else
            _identify_reaction_feature(ζ, N; use_cubic=use_cubic)
        end

        # Compute weighted exposure
        weighted_exposure = 0.0
        for (x, exp_time) in exposure
            φ_x = feature_map(x)
            if feature_idx > 0 && feature_idx <= length(φ_x)
                weighted_exposure += exp_time * φ_x[feature_idx]
            end
        end

        # Estimate rate constant
        if weighted_exposure > 0
            rate_est = total_count / weighted_exposure
            # Perturb the estimate to avoid a trivial warm start.
            #
            # The rate is a positive quantity, so the perturbation must be
            # multiplicative in a way that cannot change its sign. The lognormal
            # factor exp(σ·z) satisfies this for any σ.
            #
            # The previous model, rate·(1 + σ·z) followed by max(·, 1e-8), does
            # not: at σ=5 the factor is negative for z < -0.2, which is 42% of
            # draws, and the clamp then silently replaced the direction by 1e-8.
            # That is deletion, not perturbation — a true reaction removed before
            # the optimizer sees it — and it made each benchmark's success
            # contingent on a roughly 1-in-5 favourable draw. It is retained only
            # as :legacy_additive so the earlier configuration can be reproduced.
            rate_noisy = if noise_model === :lognormal
                rate_est * exp(noise_scale * randn(rng))
            elseif noise_model === :legacy_additive
                max(rate_est * (1.0 + noise_scale * randn(rng)), 1e-8)
            else
                throw(ArgumentError("noise_model must be :lognormal or :legacy_additive, got $noise_model"))
            end
            θ[feature_idx] = rate_noisy

            if verbose
                println("\n$ζ ($feature_desc):")
                println("  Transitions: $total_count")
                println("  Weighted exposure: $(round(weighted_exposure, digits=2))")
                println("  θ[$feature_idx] = $(round(rate_noisy, sigdigits=4)) (base: $(round(rate_est, sigdigits=4)))")
            end
        else
            # Fallback: small random initialization
            θ .= abs.(randn(K)) * 0.001
            if verbose
                println("\n$ζ: No exposure (using random init)")
            end
        end

        θ_emp[ζ] = θ
    end

    verbose && println("="^70)

    return θ_emp
end

"""
Alternative signature for direct trajectory input.
"""
function empirical_initialization(trajectories, R; verbose=false, noise_scale=0.1)
    data = count_transitions(trajectories, R; verbose=verbose)
    return empirical_initialization(data, R; verbose=verbose, noise_scale=noise_scale)
end

# =============================================================================
# Internal helpers
# =============================================================================

"""
Identify the most likely feature for a given reaction stoichiometry.

Returns (feature_index, description) based on mass-action kinetics heuristics.
For cubic features, also handles trimolecular reactions.
"""
function _identify_reaction_feature(ζ::NTuple{N,Int}, num_species::Int; use_cubic=false) where {N}
    # Count reactants (negative entries in stoichiometry)
    reactant_species = findall(x -> x < 0, collect(ζ))
    reactant_counts = [-ζ[i] for i in reactant_species]
    num_reactants = sum(max(0, -z) for z in ζ)

    if num_reactants == 0
        # Zero-order (birth): constant feature
        return 1, "zero-order"

    elseif num_reactants == 1 && length(reactant_species) == 1
        # Unimolecular: linear feature
        species_idx = reactant_species[1]
        feature_idx = 1 + species_idx  # Skip constant
        return feature_idx, "unimolecular (species $species_idx)"

    elseif num_reactants == 2 && length(reactant_species) == 1
        # Dimerization: quadratic self-interaction
        species_idx = reactant_species[1]
        feature_idx = 1 + N + species_idx  # Skip constant + linear
        return feature_idx, "dimerization (species $species_idx)"

    elseif num_reactants == 2 && length(reactant_species) == 2
        # Bimolecular: cross-term
        i, j = sort(reactant_species)
        # Compute cross-term index: 1 + N + N + position in upper triangular
        cross_idx = 0
        for a in 1:N-1
            for b in a+1:N
                cross_idx += 1
                if a == i && b == j
                    feature_idx = 1 + 2N + cross_idx
                    return feature_idx, "bimolecular (species $i × $j)"
                end
            end
        end

    elseif use_cubic && num_reactants == 3
        # Trimolecular reactions (requires cubic features)
        # For 2X + Y → products: need X²Y/2 term
        # For X + Y + Z → products: need XYZ term

        if length(reactant_species) == 2
            # Two species involved: one appears twice (e.g., 2X + Y)
            # Find which species appears twice
            counts = Dict{Int,Int}()
            for s in reactant_species
                counts[s] = get(counts, s, 0) + abs(ζ[s])
            end

            # Find the species with count 2 and count 1
            species_2 = 0
            species_1 = 0
            for (s, c) in counts
                if c == 2
                    species_2 = s
                elseif c == 1
                    species_1 = s
                end
            end

            if species_2 > 0 && species_1 > 0
                # X²Y/2 term: in cubic_features, these come after cubic self terms
                # Order: 1, linear, quad_self, quad_cross, cubic_self, cubic_mixed, cubic_triple
                # Cubic mixed index: for species i with count 2 and species j with count 1
                # The X(X-1)/2 * Y terms are ordered as: i=1,j=2; i=1,j=3; ...; i=2,j=1; i=2,j=3; etc
                quad_dim = 1 + 2N + (N * (N - 1)) ÷ 2
                cubic_self = N

                # Find position in cubic mixed ordering
                mixed_idx = 0
                for i in 1:N
                    for j in 1:N
                        if i != j
                            mixed_idx += 1
                            if i == species_2 && j == species_1
                                feature_idx = quad_dim + cubic_self + mixed_idx
                                return feature_idx, "trimolecular 2×$species_2 + $species_1"
                            end
                        end
                    end
                end
            end
        end
    end

    # Fallback: use constant feature
    return 1, "unknown (using constant)"
end

# Note: _infer_species_count and _extract_state are defined in state_space.jl
# and are shared across the module
