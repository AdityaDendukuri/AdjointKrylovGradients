"""Order-16 Chebyshev rational approximation of a real exponential action.

The coefficients are the partial-fraction CRAM16 coefficients of Pusa.  For a
real matrix `B`,

    exp(B) * v ≈ alpha0 * v + 2real(sum(alpha[k] * inv(B-theta[k]I) * v)).

Only one pole from each conjugate pair is stored.
"""
const CRAM16_ALPHA0 = 2.1248537104952237488e-16
const CRAM16_THETA = ComplexF64[
    -10.843917078696988026 + 19.277446167181652284im,
     -5.2649713434426468895 + 16.220221473167927305im,
      5.9481522689511774808 + 3.5874573620183222829im,
      3.5091036084149180974 + 8.4361989858843750826im,
      6.4161776990994341923 + 1.1941223933701386874im,
      1.4193758971856659786 + 10.925363484496722585im,
      4.9931747377179963991 + 5.9968817136039422260im,
     -1.4139284624888862114 + 13.497725698892745389im,
]
const CRAM16_ALPHA = ComplexF64[
    -5.0901521865224915650e-7 - 2.4220017652852287970e-5im,
     2.1151742182466030907e-4 + 4.3892969647380673918e-3im,
     1.1339775178483930527e2 + 1.0194721704215856450e2im,
     1.5059585270023467528e1 - 5.7514052776421819979im,
    -6.4500878025539646595e1 - 2.2459440762652096056e2im,
    -1.4793007113557999718 + 1.7686588323782937906im,
    -6.2518392463207918892e1 - 1.1190391094283228480e1im,
     4.1023136835410021273e-2 - 1.5743466173455468191e-1im,
]

"""Evaluate a real exponential action with CRAM16 and sparse direct solves."""
function sparse_direct_exponential(A, v, dt)
    isreal(A) && isreal(v) ||
        throw(ArgumentError("CRAM16 sparse-direct evaluation currently requires real inputs"))
    B = ComplexF64.(dt .* A)
    result = CRAM16_ALPHA0 .* Float64.(v)
    for (alpha, theta) in zip(CRAM16_ALPHA, CRAM16_THETA)
        factor = lu(B - theta * I)
        result .+= 2 .* real.(alpha .* (factor \ v))
    end
    result
end

"""Compute an exponential action and its reverse-mode parameter derivatives.

`A` and every matrix in `directions` must be real.  For the scalar functional
`upstream' * exp(dt*A) * v`, the routine differentiates the CRAM16 rational
approximation analytically.  Each shifted sparse LU factorization is reused
for the forward solve and its transpose solve, and both solves are shared by
all parameter directions.

The return value contains the propagated state, the parameter VJP, the state
VJP, and the number of numeric factorizations.
"""
function sparse_direct_exponential_vjp(A, directions, v, dt, upstream)
    sparse_direct_exponential_vjp(
        A, SparseDirectionCache(directions), v, dt, upstream,
    )
end

function sparse_direct_exponential_vjp(
    A, direction_cache::SparseDirectionCache, v, dt, upstream,
)
    isreal(A) && isreal(v) && isreal(upstream) ||
        throw(ArgumentError("CRAM16 sparse-direct VJP currently requires real inputs"))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    n = size(A, 1)
    length(v) == n || throw(DimensionMismatch("v has incompatible length"))
    length(upstream) == n ||
        throw(DimensionMismatch("upstream has incompatible length"))
    direction_cache.matrix_size == size(A) ||
        throw(DimensionMismatch("the parameter library has incompatible dimensions"))

    B = ComplexF64.(dt .* A)
    prediction = CRAM16_ALPHA0 .* Float64.(v)
    state_vjp = CRAM16_ALPHA0 .* Float64.(upstream)
    parameter_vjp = zeros(Float64, size(direction_cache.coefficients, 2))
    kernel = zeros(ComplexF64, length(direction_cache.rows))
    projected = similar(kernel, length(parameter_vjp))

    for (alpha, theta) in zip(CRAM16_ALPHA, CRAM16_THETA)
        factor = lu(B - theta * I)
        forward = factor \ v
        # The objective is real-linear.  The analytic derivative therefore
        # uses a nonconjugating transpose, not a Hermitian transpose.
        adjoint = transpose(factor) \ upstream
        prediction .+= 2 .* real.(alpha .* forward)
        state_vjp .+= 2 .* real.(alpha .* adjoint)
        @inbounds for coordinate in eachindex(kernel)
            kernel[coordinate] =
                adjoint[direction_cache.rows[coordinate]] *
                forward[direction_cache.cols[coordinate]]
        end
        mul!(projected, transpose(direction_cache.coefficients), kernel)
        parameter_vjp .-= 2dt .* real.(alpha .* projected)
    end

    (
        prediction=prediction,
        parameter_vjp=parameter_vjp,
        state_vjp=state_vjp,
        factorizations=length(CRAM16_THETA),
    )
end
