# =============================================================================
# State space construction and distribution building
# =============================================================================

"""
    build_state_space(trajectories, times, R; expansion=1, verbose=false) -> Vector{NTuple}

Build state space from observed trajectory data with optional expansion.

This is the inverse-problem analog of Finite State Projection (FSP).
Instead of arbitrarily truncating the state space, we construct it from
actually visited states plus a configurable expansion for closure.

# Arguments
- `trajectories`: Vector of SSA trajectory solutions (from DifferentialEquations.jl)
- `times`: Observation time points
- `R`: Reaction stoichiometries (for expansion directions)
- `expansion`: Number of expansion steps (default: 1)
- `verbose`: Print diagnostic information (default: false)

# Returns
- Sorted vector of state tuples

# Algorithm
1. Collect all states visited at observation times
2. Optionally expand by adding reachable neighbor states
3. Return sorted state space for consistent indexing

# Example
```julia
times = collect(0.0:0.1:1.0)
R = [(-1,-1,1,0), (1,1,-1,0), (0,1,-1,1)]
X = build_state_space(trajectories, times, R; expansion=3)
```
"""
function build_state_space(trajectories, times, R; expansion=1, verbose=false)
    N = _infer_species_count(trajectories)
    visited = Set{NTuple{N,Int}}()

    # Collect all visited states at observation times
    for traj in trajectories
        for t in times
            state = _extract_state(traj, t, N)
            push!(visited, state)
        end
    end

    verbose && println("States visited at observation times: $(length(visited))")

    # Expand for closure under reactions
    if expansion > 0
        expanded = copy(visited)

        for _ in 1:expansion
            new_states = Set{NTuple{N,Int}}()
            for x in expanded
                for ζ in R
                    # Forward transition
                    x_new = ntuple(i -> x[i] + ζ[i], N)
                    if all(x_new .>= 0)
                        push!(new_states, x_new)
                    end
                    # Backward (for completeness)
                    x_back = ntuple(i -> x[i] - ζ[i], N)
                    if all(x_back .>= 0)
                        push!(new_states, x_back)
                    end
                end
            end
            expanded = union(expanded, new_states)
        end

        verbose && println("After expansion=$expansion: $(length(expanded)) states")
        visited = expanded
    end

    # Sort for consistent ordering
    return sort(collect(visited))
end

"""
    build_distributions(trajectories, times, X) -> Vector{Vector{Float64}}

Build empirical probability distributions from trajectory data.

For each observation time, counts the fraction of trajectories in each
state of the state space X.

# Arguments
- `trajectories`: Vector of SSA trajectory solutions
- `times`: Observation time points
- `X`: State space (vector of tuples)

# Returns
- Vector of probability distributions (one per time point)

# Example
```julia
P = build_distributions(trajectories, times, X)
# P[k][i] = fraction of trajectories in state X[i] at time times[k]
```
"""
function build_distributions(trajectories, times, X)
    N = length(first(X))
    idx_map = Dict(x => i for (i, x) in enumerate(X))
    n = length(X)

    return map(times) do t
        p = zeros(n)
        missed = 0

        for traj in trajectories
            state = _extract_state(traj, t, N)
            i = get(idx_map, state, 0)

            if i > 0
                p[i] += 1.0
            else
                missed += 1
            end
        end

        if missed > 0
            @warn "Time $t: $missed trajectories in states not in X"
        end

        total = sum(p)
        total > 0 ? p / total : p
    end
end

"""
    state_index_map(X) -> Dict

Create a mapping from state tuples to indices.

# Arguments
- `X`: State space vector

# Returns
- Dict mapping state tuple to 1-based index
"""
function state_index_map(X)
    return Dict(x => i for (i, x) in enumerate(X))
end

# =============================================================================
# Internal helpers
# =============================================================================

"""
Infer number of species from trajectory structure.
"""
function _infer_species_count(trajectories)
    traj = first(trajectories)
    return length(traj(0.0))
end

"""
Extract state tuple from trajectory at time t.
"""
function _extract_state(traj, t, N)
    vals = traj(t)
    return ntuple(i -> Int(vals[i]), N)
end
