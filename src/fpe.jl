using SparseArrays

# Builds the generator decomposition A(θ) = A₀ + Σⱼ θⱼ Aⱼ for the 1D FPE
#   ∂_t p = -∂_x(b(x;θ) p) + D ∂_xx p
# with b(x;θ) = Σⱼ θⱼ bⱼ(x) (linear library).
#
# Discretization: uniform grid xs, reflecting BCs.
# Drift uses conservative centered differences:
#   [Aⱼ p]ᵢ ≈ -(bⱼ[i+1] p_{i+1} - bⱼ[i-1] p_{i-1}) / (2h)
# so column j of Aⱼ:  Aⱼ[j+1,j] = +bⱼ[j]/(2h),  Aⱼ[j-1,j] = -bⱼ[j]/(2h).
# Valid generator when diffusion dominates: D/h² ≫ max|bⱼ|/(2h).
function fpe_library_1d(xs::AbstractVector, drift_fns; D=1.0)
    n = length(xs)
    h = (xs[end] - xs[1]) / (n - 1)

    ri = Int[]; ci = Int[]; vi = Float64[]
    for j in 1:n
        j > 1 && (push!(ri, j-1); push!(ci, j); push!(vi,  D/h^2))
        j < n && (push!(ri, j+1); push!(ci, j); push!(vi,  D/h^2))
        d = (j == 1 || j == n) ? -D/h^2 : -2D/h^2
        push!(ri, j); push!(ci, j); push!(vi, d)
    end
    A₀ = sparse(ri, ci, vi, n, n)

    dirs = map(drift_fns) do bfn
        bv = bfn.(xs)
        rd = Int[]; cd = Int[]; vd = Float64[]
        for j in 1:n
            j < n && (push!(rd, j+1); push!(cd, j); push!(vd,  bv[j]/(2h)))
            j > 1 && (push!(rd, j-1); push!(cd, j); push!(vd, -bv[j]/(2h)))
        end
        # reflecting BCs: ghost-cell flux folds back onto the boundary diagonal
        # left wall (j=1): missing outflow -bv[1]/(2h) reflected to diagonal
        push!(rd, 1); push!(cd, 1); push!(vd, -bv[1]/(2h))
        # right wall (j=n): missing outflow +bv[n]/(2h) reflected to diagonal
        push!(rd, n); push!(cd, n); push!(vd,  bv[n]/(2h))
        sparse(rd, cd, vd, n, n)
    end

    return A₀, dirs
end
