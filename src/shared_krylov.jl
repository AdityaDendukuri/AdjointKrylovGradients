using LinearAlgebra, SparseArrays
using ChainRulesCore
using ExponentialUtilities: expv
using FastGaussQuadrature: gausslegendre

const GL_NODES_5   = [0.046910077781, 0.230765345791, 0.500000000000,
                      0.769234654209, 0.953089922219]
const GL_WEIGHTS_5 = [0.118463442528, 0.239314335249, 0.284444444444,
                      0.239314335249, 0.118463442528]
const GL_NODES_8   = [0.019855071751, 0.101666776723, 0.237233795042,
                      0.408282678752, 0.591717321248, 0.762766204958,
                      0.898333223277, 0.980144928249]
const GL_WEIGHTS_8 = [0.050614268145, 0.111190517227, 0.156853322939,
                      0.181341891689, 0.181341891689, 0.156853322939,
                      0.111190517227, 0.050614268145]

# Fixed rules support legacy benchmark paths; production rules are generated
# accurately and cached by `gauss_legendre_01`.

const _GL_CACHE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
const _GL_LOCK = ReentrantLock()

"""
    gauss_legendre_01(n) -> (nodes, weights)

`n`-point Gauss–Legendre rule on `[0,1]`.  Exact for polynomials of degree
`2n-1`; the weights sum to 1.  Results are memoized per order.
"""
function gauss_legendre_01(n::Int)
    n >= 1 || throw(ArgumentError("n_quad must be >= 1, got $n"))
    lock(_GL_LOCK) do
        get!(_GL_CACHE, n) do
            x, w = gausslegendre(n)          # nodes/weights on [-1,1]
            ((x .+ 1) ./ 2, w ./ 2)          # affine map to [0,1]
        end
    end
end

real_scalar_type(::Type{T}) where {T} = typeof(real(zero(T)))

# Stable softplus for unconstrained nonnegative coefficients.
softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    SparseDirectionCache(dirs)

Precompile the union sparsity pattern of a homogeneous sparse generator
library.  If `P` matrix coordinates occur in at least one direction, the cache
stores a sparse `P × M` coefficient matrix.  A low-rank adjoint kernel can
therefore be evaluated only on those `P` coordinates and projected onto every
direction with one sparse transpose multiplication.
"""
struct SparseDirectionCache{T, RT, Ti <: Integer}
    rows :: Vector{Ti}
    cols :: Vector{Ti}
    coefficients :: SparseMatrixCSC{T, Ti}
    direction_norms :: Vector{RT}
    matrix_size :: Tuple{Int, Int}
end

function SparseDirectionCache(
    dirs::AbstractVector{<:SparseMatrixCSC},
)
    isempty(dirs) && throw(ArgumentError(
        "cannot compile an empty direction library",
    ))
    matrix_size = size(first(dirs))
    all(size(direction) == matrix_size for direction in dirs) ||
        throw(DimensionMismatch("all directions must have the same size"))
    T = promote_type(map(eltype, dirs)...)
    RT = real_scalar_type(T)
    Ti = Int
    positions = Dict{Tuple{Int,Int},Int}()
    rows = Ti[]
    cols = Ti[]
    coefficient_rows = Ti[]
    coefficient_cols = Ti[]
    coefficient_values = T[]

    for (direction_index, direction) in pairs(dirs)
        direction_rows = rowvals(direction)
        direction_values = nonzeros(direction)
        for column in axes(direction, 2)
            for pointer in nzrange(direction, column)
                value = direction_values[pointer]
                iszero(value) && continue
                row = direction_rows[pointer]
                coordinate = (row, column)
                union_index = get(positions, coordinate, 0)
                if iszero(union_index)
                    push!(rows, row)
                    push!(cols, column)
                    union_index = length(rows)
                    positions[coordinate] = union_index
                end
                push!(coefficient_rows, union_index)
                push!(coefficient_cols, direction_index)
                push!(coefficient_values, value)
            end
        end
    end

    coefficients = sparse(
        coefficient_rows,
        coefficient_cols,
        coefficient_values,
        length(rows),
        length(dirs),
    )
    direction_norms = RT[norm(direction) for direction in dirs]
    SparseDirectionCache{T, RT, Ti}(
        rows,
        cols,
        coefficients,
        direction_norms,
        matrix_size,
    )
end

struct ArnoldiWorkspace{T}
    V :: Matrix{T}
    H :: Matrix{T}
    w :: Vector{T}
end
ArnoldiWorkspace(T, n, m) = ArnoldiWorkspace{T}(zeros(T,n,m), zeros(T,m,m), zeros(T,n))

function arnoldi_basis!(ws::ArnoldiWorkspace{T}, A, v::AbstractVector{T};
                        m::Int=25, tol=1e-12) where T
    n = length(v)
    β = norm(v)
    m = min(m, n, size(ws.V, 2))
    V, H, w = ws.V, ws.H, ws.w
    @inbounds for j in 1:m; fill!(view(V,:,j), zero(T)); end
    @inbounds for j in 1:m; fill!(view(H,:,j), zero(T)); end
    β < tol && return (V=view(V,:,1:1), H=view(H,1:1,1:1), beta=β, m=1)
    @inbounds V[:, 1] .= v ./ β
    k = m
    @inbounds for j in 1:m
        mul!(w, A, view(V, :, j))
        for i in 1:j
            h = dot(view(V, :, i), w)
            H[i, j] = h
            @simd for l in 1:n; w[l] -= h * V[l, i]; end
        end
        nw = norm(w)
        if j < m
            if nw < tol; k = j; break; end
            H[j+1, j] = nw
            @simd for l in 1:n; V[l, j+1] = w[l] / nw; end
        end
    end
    return (V=view(V,:,1:k), H=view(H,1:k,1:k), beta=β, m=k)
end

function arnoldi_basis(A, v::AbstractVector{T}; m::Int=25, tol=1e-12) where T
    ws = ArnoldiWorkspace(T, length(v), min(m, length(v)))
    arnoldi_basis!(ws, A, v; m=m, tol=tol)
end

function _eval_krylov_batch_dense(ks, ts::AbstractVector, scale)
    m  = ks.m
    Vm = view(ks.V, :, 1:m)
    Hm = Matrix(ks.H)
    any(!isfinite, Hm) && return zeros(eltype(Vm), size(Vm,1), length(ts))
    e1 = zeros(eltype(Hm), m); e1[1] = 1.0
    nts = length(ts)
    Y   = Matrix{eltype(Hm)}(undef, m, nts)
    @inbounds for k in 1:nts
        Y[:, k] = exp(ts[k] * Hm) * e1
    end
    any(!isfinite, Y) && return zeros(eltype(Vm), size(Vm,1), nts)
    Y_out = eltype(Vm) <: Real ? real.(Y) : Y
    scale * Vm * Y_out
end

const LAPACK_SCALAR =
    Union{Float32, Float64, ComplexF32, ComplexF64}

"""
Evaluate `exp(tH)e₁` at many times from one eigendecomposition of the reduced
Hessenberg matrix.

Returns `nothing` when the eigendecomposition is unsafe, allowing the caller
to use a dense-exponential fallback.
"""
function _eval_reduced_exponential_batch(Hm, ts, e1)
    T = eltype(Hm)
    T <: LAPACK_SCALAR || return nothing
    RT = real_scalar_type(T)

    decomposition = try
        eigen(Hm)
    catch error
        if error isa LinearAlgebra.LAPACKException
            return nothing
        end
        rethrow()
    end
    vectors = decomposition.vectors
    vector_norm = opnorm(vectors, 1)
    isfinite(vector_norm) || return nothing

    factorization = lu(vectors; check=false)
    reciprocal_condition = try
        LinearAlgebra.LAPACK.gecon!(
            '1', factorization.factors, vector_norm,
        )
    catch error
        if error isa LinearAlgebra.LAPACKException
            return nothing
        end
        rethrow()
    end
    if !isfinite(reciprocal_condition) ||
       reciprocal_condition < sqrt(eps(RT))
        return nothing
    end

    coefficients = try
        factorization \ e1
    catch error
        if error isa SingularException
            return nothing
        end
        rethrow()
    end
    all(isfinite, coefficients) || return nothing

    nt = length(ts)
    Y = Matrix{eltype(vectors)}(undef, length(e1), nt)
    scaled_coefficients = similar(coefficients)
    @inbounds for k in eachindex(ts)
        scaled_coefficients .=
            exp.(ts[k] .* decomposition.values) .* coefficients
        mul!(view(Y, :, k), vectors, scaled_coefficients)
    end
    all(isfinite, Y) || return nothing

    if T <: Real
        real_norm = sqrt(sum(abs2(real(value)) for value in Y))
        imaginary_norm = sqrt(sum(abs2(imag(value)) for value in Y))
        imaginary_norm <=
            sqrt(eps(RT)) * max(real_norm, eps(RT)) || return nothing
        return real.(Y)
    end
    Y
end

function _eval_krylov_batch(ks, ts::AbstractVector, scale)
    m = ks.m
    # A single dense exponential is faster and maximally robust for endpoint
    # evaluations. The diagonalization path pays off for quadrature batches.
    (length(ts) >= 3 && m >= 12) ||
        return _eval_krylov_batch_dense(ks, ts, scale)

    Vm = view(ks.V, :, 1:m)
    Hm = Matrix(ks.H)
    any(!isfinite, Hm) &&
        return zeros(eltype(Vm), size(Vm, 1), length(ts))
    e1 = zeros(eltype(Hm), m)
    e1[1] = one(eltype(Hm))
    Y = _eval_reduced_exponential_batch(Hm, ts, e1)
    Y === nothing && return _eval_krylov_batch_dense(ks, ts, scale)

    output = scale * Vm * Y
    all(isfinite, output) ||
        return _eval_krylov_batch_dense(ks, ts, scale)
    output
end

# Fréchet VJP shared by `SemigroupLayer` and trajectory-loss wrappers.

function krylov_frechet_vjp(A, dirs, p0, dt, Δ;
                            m_krylov = 25, n_quad = 5,
                            ws_f = nothing, ws_a = nothing)
    M = length(dirs)
    n = length(p0)
    T = eltype(p0)
    RT = real_scalar_type(T)
    raw_nodes, raw_weights = gauss_legendre_01(n_quad)
    nodes = RT.(raw_nodes)
    weights = RT.(raw_weights)

    # r_adj = -Δ:  the adjoint seed sign follows from the derivation —
    # val accumulates with a minus, so passing -Δ gives +∂L/∂θ.
    r_adj = -Δ
    norm(r_adj) < RT(1e-15) && return zeros(RT, M)

    _ws_f = ws_f !== nothing ? ws_f : ArnoldiWorkspace(T, n, m_krylov)
    _ws_a = ws_a !== nothing ? ws_a : ArnoldiWorkspace(T, n, m_krylov)
    buf   = zeros(T, n)

    ks_f = arnoldi_basis!(_ws_f, A,  p0;    m=m_krylov)
    ks_a = arnoldi_basis!(_ws_a, A', r_adj; m=m_krylov)

    F_k = _eval_krylov_batch(ks_f, nodes .* dt,         ks_f.beta)
    Λ_k = _eval_krylov_batch(ks_a, (1 .- nodes) .* dt,  ks_a.beta)

    # grad is always real: θ is real even when A and p0 are complex.
    # real() is a no-op for real types and takes the real part for complex.
    grad = zeros(RT, M)
    for j in 1:M
        val = zero(T)
        @inbounds for q in 1:n_quad
            mul!(buf, dirs[j], view(F_k, :, q))
            val -= weights[q] * dot(view(Λ_k, :, q), buf)
        end
        grad[j] = real(dt * val)
    end
    grad
end

"""
    compressed_core_frechet_vjp(A, dirs, p0, dt, Δ;
        m_krylov=25, core_rtol=1e-8, core_atol=0,
        ws_f=nothing, ws_a=nothing, direction_cache=nothing,
        return_diagnostics=false)

Approximate the exact-semigroup parameter VJP using the same forward and
adjoint Arnoldi spaces as [`krylov_frechet_vjp`](@ref), but integrate the two
projected paths exactly instead of applying a fixed Gauss--Legendre rule.

For projected forward and adjoint paths, define the reduced sensitivity core

```
Cₘ = ∫₀ᵈᵗ exp((dt-t)Hₐ) (βₐe₁)(β_f e₁)' exp(tH_f') dt.
```

It is the upper-right block of one small block-triangular exponential.  If
`Cₘ ≈ U_r Diagonal(σ_r) V_r'`, every library gradient is then

```
gⱼ = real(sum(σ[k] * dot(Vₐ*U[:,k], Aⱼ*(V_f*V[:,k])) for k=1:r)).
```

Thus the expensive reduced integration is shared over every direction.  The
SVD rank is selected so that the discarded Frobenius norm is at most
`max(core_atol, core_rtol*norm(Cₘ))`.  With orthonormal Arnoldi bases, the
resulting projected-gradient compression error obeys

```
|δgⱼ| ≤ norm(Cₘ-Cₘ,r) * norm(Aⱼ, Frobenius).
```

Set `core_rtol=core_atol=0` to retain the full reduced core.  When
`direction_cache=SparseDirectionCache(dirs)`, the library contractions are
fused over the union sparsity pattern.  Construct the cache once and reuse it
across layer calls.  When
`return_diagnostics=true`, return a named tuple containing the gradient,
selected rank, full rank, relative Frobenius tail, and per-direction absolute
error bounds.
"""
function compressed_core_frechet_vjp(
    A, dirs, p0, dt, Δ;
    m_krylov=25,
    core_rtol=1e-8,
    core_atol=0,
    ws_f=nothing,
    ws_a=nothing,
    direction_cache=nothing,
    return_diagnostics=false,
)
    M = length(dirs)
    n = length(p0)
    T = eltype(p0)
    RT = real_scalar_type(T)
    zero_gradient = zeros(RT, M)
    if norm(Δ) < RT(1e-15)
        diagnostics = (
            gradient=zero_gradient,
            core_rank=0,
            core_full_rank=0,
            core_relative_tail=zero(RT),
            direction_error_bounds=zeros(RT, M),
        )
        return return_diagnostics ? diagnostics : zero_gradient
    end

    _ws_f = ws_f !== nothing ? ws_f :
        ArnoldiWorkspace(T, n, m_krylov)
    _ws_a = ws_a !== nothing ? ws_a :
        ArnoldiWorkspace(T, n, m_krylov)
    ks_f = arnoldi_basis!(_ws_f, A, p0; m=m_krylov)
    ks_a = arnoldi_basis!(_ws_a, A', Δ; m=m_krylov)

    _compressed_core_from_bases(
        ks_f,
        ks_a,
        dirs,
        dt;
        core_rtol=core_rtol,
        core_atol=core_atol,
        direction_cache=direction_cache,
        return_diagnostics=return_diagnostics,
    )
end

"""
Internal fused form of [`compressed_core_frechet_vjp`](@ref).  The caller
supplies the forward and adjoint Arnoldi decompositions so a layer that already
computed its prediction and state cotangent does not repeat either sparse
Krylov construction.
"""
function _compressed_core_from_bases(
    ks_f,
    ks_a,
    dirs,
    dt;
    core_rtol=1e-8,
    core_atol=0,
    direction_cache=nothing,
    return_diagnostics=false,
)
    M = length(dirs)
    T = eltype(ks_f.V)
    RT = real_scalar_type(T)
    # The reduced core can have a much wider singular-value range than the
    # state vectors.  Accumulate only this O(m²)-storage/O(m³)-work part in
    # Float64 when the large sparse calculation uses Float32.  This preserves
    # low-rank information without promoting any n-by-n operator.
    WT = T === Float32 ? Float64 :
         T === ComplexF32 ? ComplexF64 : T
    WRT = real_scalar_type(WT)
    n = size(ks_f.V, 1)
    zero_gradient = zeros(RT, M)
    mf, ma = ks_f.m, ks_a.m
    Hf = WT.(ks_f.H)
    Ha = WT.(ks_a.H)
    block_matrix = zeros(WT, ma + mf, ma + mf)
    block_matrix[1:ma, 1:ma] .= Ha
    block_matrix[(ma + 1):end, (ma + 1):end] .= Hf'
    block_matrix[1, ma + 1] =
        convert(WT, ks_a.beta * ks_f.beta)

    block_exponential = exp(WRT(dt) .* block_matrix)
    core = Matrix(block_exponential[1:ma, (ma + 1):end])
    all(isfinite, core) ||
        error("Nonfinite reduced sensitivity core")

    decomposition = svd(core)
    singular_values = decomposition.S
    full_rank = length(singular_values)
    core_norm = norm(singular_values)
    threshold = max(WRT(core_atol), WRT(core_rtol) * core_norm)
    if iszero(core_rtol) && iszero(core_atol)
        selected_rank = full_rank
        tail_squared = zero(WRT)
    else
        tail_squared = sum(abs2, singular_values)
        selected_rank = 0
        while selected_rank < full_rank &&
              sqrt(max(tail_squared, zero(WRT))) > threshold
            selected_rank += 1
            tail_squared -= abs2(singular_values[selected_rank])
        end
    end
    tail_norm = sqrt(max(tail_squared, zero(WRT)))

    if selected_rank == 0
        gradient = zero_gradient
    else
        forward_basis = WT.(view(ks_f.V, :, 1:mf))
        adjoint_basis = WT.(view(ks_a.V, :, 1:ma))
        left_factors =
            adjoint_basis * view(decomposition.U, :, 1:selected_rank)
        right_factors =
            forward_basis * view(decomposition.V, :, 1:selected_rank)
        if direction_cache === nothing
            buffer = zeros(WT, n)
            gradient = zeros(RT, M)
            @inbounds for j in 1:M
                value = zero(WT)
                for index in 1:selected_rank
                    mul!(
                        buffer,
                        dirs[j],
                        view(right_factors, :, index),
                    )
                    value += singular_values[index] *
                             dot(view(left_factors, :, index), buffer)
                end
                gradient[j] = real(value)
            end
        else
            direction_cache.matrix_size == (n, n) ||
                throw(DimensionMismatch(
                    "direction cache matrix size does not match the bases",
                ))
            size(direction_cache.coefficients, 2) == M ||
                throw(DimensionMismatch(
                    "direction cache does not match the direction count",
                ))
            kernel_values =
                zeros(WT, length(direction_cache.rows))
            @inbounds for index in 1:selected_rank
                σ = singular_values[index]
                for coordinate in eachindex(kernel_values)
                    row = direction_cache.rows[coordinate]
                    column = direction_cache.cols[coordinate]
                    kernel_values[coordinate] +=
                        σ * conj(left_factors[row, index]) *
                        right_factors[column, index]
                end
            end
            projected = transpose(direction_cache.coefficients) *
                        kernel_values
            gradient = RT.(real.(projected))
        end
    end

    relative_tail = tail_norm / max(core_norm, eps(WRT))
    direction_norms = direction_cache === nothing ?
        WRT[norm(direction) for direction in dirs] :
        WRT.(direction_cache.direction_norms)
    direction_error_bounds = tail_norm .* direction_norms
    diagnostics = (
        gradient=gradient,
        core_rank=selected_rank,
        core_full_rank=full_rank,
        core_relative_tail=relative_tail,
        direction_error_bounds=direction_error_bounds,
    )
    return_diagnostics ? diagnostics : gradient
end

"""Differentiable action `exp((A0 + sum(θ[j]dirs[j]))dt)p0`."""
struct SemigroupLayer{T, M <: AbstractMatrix{T}, C}
    A0         :: M
    dirs       :: Vector{M}
    m_krylov   :: Int
    n_quad     :: Int
    vjp_method :: Symbol
    core_rtol  :: Float64
    core_atol  :: Float64
    direction_cache :: C
end

function _validate_vjp_options(vjp_method, core_rtol, core_atol)
    vjp_method in (:quadrature, :compressed_core) || throw(ArgumentError(
        "vjp_method must be :quadrature or :compressed_core, got " *
        repr(vjp_method),
    ))
    isfinite(core_rtol) && core_rtol >= 0 || throw(ArgumentError(
        "core_rtol must be finite and nonnegative",
    ))
    isfinite(core_atol) && core_atol >= 0 || throw(ArgumentError(
        "core_atol must be finite and nonnegative",
    ))
    nothing
end

function SemigroupLayer(
    A0::M,
    dirs::Vector{M};
    m_krylov=25,
    n_quad=5,
    vjp_method=:quadrature,
    core_rtol=1e-7,
    core_atol=0.0,
) where {T, M <: AbstractMatrix{T}}
    _validate_vjp_options(vjp_method, core_rtol, core_atol)
    direction_cache = vjp_method === :compressed_core &&
                      M <: SparseMatrixCSC ?
        SparseDirectionCache(dirs) : nothing
    SemigroupLayer{T, M, typeof(direction_cache)}(
        A0,
        dirs,
        m_krylov,
        n_quad,
        vjp_method,
        Float64(core_rtol),
        Float64(core_atol),
        direction_cache,
    )
end

function SemigroupLayer(
    A0,
    dirs;
    m_krylov=25,
    n_quad=5,
    vjp_method=:quadrature,
    core_rtol=1e-7,
    core_atol=0.0,
)
    _validate_vjp_options(vjp_method, core_rtol, core_atol)
    T   = promote_type(eltype(A0), eltype(first(dirs)))
    A0c = convert(SparseMatrixCSC{T,Int}, A0)
    dc  = SparseMatrixCSC{T,Int}[convert(SparseMatrixCSC{T,Int}, d) for d in dirs]
    direction_cache = vjp_method === :compressed_core ?
        SparseDirectionCache(dc) : nothing
    SemigroupLayer{
        T,
        SparseMatrixCSC{T,Int},
        typeof(direction_cache),
    }(
        A0c,
        dc,
        m_krylov,
        n_quad,
        vjp_method,
        Float64(core_rtol),
        Float64(core_atol),
        direction_cache,
    )
end

function (sl::SemigroupLayer)(p0, θ, dt)
    A = sl.A0 + sum(θ[j] * sl.dirs[j] for j in eachindex(θ))
    expv(real(dt), A, p0; m=sl.m_krylov)
end

function ChainRulesCore.rrule(sl::SemigroupLayer, p0, θ, dt)
    A      = sl.A0 + sum(θ[j] * sl.dirs[j] for j in eachindex(θ))
    p_pred = expv(real(dt), A, p0; m=sl.m_krylov)

    function semigroup_pullback(Δ)
        Δu = unthunk(Δ)
        if Δu isa AbstractZero
            return NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()
        end

        grad_θ = if sl.vjp_method === :compressed_core
            compressed_core_frechet_vjp(
                A,
                sl.dirs,
                p0,
                real(dt),
                Δu;
                m_krylov=sl.m_krylov,
                core_rtol=sl.core_rtol,
                core_atol=sl.core_atol,
                direction_cache=sl.direction_cache,
            )
        else
            krylov_frechet_vjp(
                A,
                sl.dirs,
                p0,
                real(dt),
                Δu;
                m_krylov=sl.m_krylov,
                n_quad=sl.n_quad,
            )
        end

        # Input-state and interval cotangents make the layer composable.
        grad_p0 = @thunk expv(real(dt), A', Δu; m=sl.m_krylov)
        grad_dt = @thunk real(dot(Δu, A * p_pred))

        return NoTangent(), grad_p0, grad_θ, grad_dt
    end

    return p_pred, semigroup_pullback
end

"""Return the semigroup least-squares gradient and loss over snapshot pairs."""
function shared_gradient(
    A,
    dirs,
    p_data,
    dts;
    obs_op=nothing,
    m_krylov=25,
    n_quad=5,
    vjp_method=:quadrature,
    core_rtol=1e-7,
    core_atol=0.0,
    direction_cache=nothing,
)
    _validate_vjp_options(vjp_method, core_rtol, core_atol)
    M  = length(dirs)
    n  = size(A, 1)
    T  = eltype(first(p_data))
    RT = real_scalar_type(T)
    raw_nodes, raw_weights = gauss_legendre_01(n_quad)
    nodes = RT.(raw_nodes)
    weights = RT.(raw_weights)
    At = A'
    grad = zeros(RT, M)
    loss = zero(RT)

    ws_f = ArnoldiWorkspace(T, n, m_krylov)
    ws_a = ArnoldiWorkspace(T, n, m_krylov)
    buf  = zeros(T, n)
    cache = if vjp_method === :compressed_core &&
               direction_cache === nothing &&
               dirs isa AbstractVector{<:SparseMatrixCSC}
        SparseDirectionCache(dirs)
    else
        direction_cache
    end

    for t in eachindex(dts)
        dt          = dts[t]
        p_t, p_next = p_data[t], p_data[t+1]

        ks_f   = arnoldi_basis!(ws_f, A, p_t; m=m_krylov)
        p_pred = _eval_krylov_batch(ks_f, [dt], ks_f.beta)[:, 1]

        if obs_op === nothing
            r     = p_next - p_pred
            r_adj = r
        else
            r     = obs_op * p_next - obs_op * p_pred
            r_adj = obs_op' * r
        end
        loss += 0.5 * norm(r)^2

        norm(r_adj) < RT(1e-15) && continue
        ks_a = arnoldi_basis!(ws_a, At, r_adj; m=m_krylov)

        if vjp_method === :compressed_core
            # r_adj = target - prediction, so the loss upstream cotangent is
            # -r_adj.  The reduced core is linear in that cotangent.
            grad .-= _compressed_core_from_bases(
                ks_f,
                ks_a,
                dirs,
                dt;
                core_rtol=core_rtol,
                core_atol=core_atol,
                direction_cache=cache,
            )
        else
            F_k = _eval_krylov_batch(
                ks_f, nodes .* dt, ks_f.beta,
            )
            Λ_k = _eval_krylov_batch(
                ks_a, (1 .- nodes) .* dt, ks_a.beta,
            )

            for j in 1:M
                val = zero(T)
                @inbounds for q in 1:n_quad
                    mul!(buf, dirs[j], view(F_k, :, q))
                    val -= weights[q] *
                           dot(view(Λ_k, :, q), buf)
                end
                grad[j] += real(dt * val)
            end
        end
    end

    return grad, loss
end

"""Return the gradient, Gauss--Newton diagonal, and snapshot loss."""
function shared_gradient_fisher(
    A,
    dirs,
    p_data,
    dts;
    obs_op=nothing,
    m_krylov=25,
    n_quad=16,
)
    M  = length(dirs)
    n  = size(A, 1)
    T  = eltype(first(p_data))
    RT = real_scalar_type(T)
    raw_nodes, raw_weights = gauss_legendre_01(n_quad)
    nodes   = RT.(raw_nodes)
    weights = RT.(raw_weights)
    At = A'

    grad   = zeros(RT, M)
    fisher = zeros(RT, M)
    loss   = zero(RT)

    ws_f = ArnoldiWorkspace(T, n, m_krylov)
    ws_a = ArnoldiWorkspace(T, n, m_krylov)
    buf  = zeros(T, n)

    for t in eachindex(dts)
        dt          = dts[t]
        p_t, p_next = p_data[t], p_data[t+1]

        ks_f   = arnoldi_basis!(ws_f, A, p_t; m=m_krylov)
        p_pred = _eval_krylov_batch(ks_f, [dt], ks_f.beta)[:, 1]

        if obs_op === nothing
            r     = p_next - p_pred
            r_adj = r
        else
            r     = obs_op * p_next - obs_op * p_pred
            r_adj = obs_op' * r
        end
        loss += 0.5 * norm(r)^2

        # The Fisher diagonal is a property of the forward paths alone, so it is
        # still well defined when the residual vanishes and the gradient is zero.
        F_k = _eval_krylov_batch(ks_f, nodes .* dt, ks_f.beta)

        residual_active = norm(r_adj) >= RT(1e-15)
        Λ_k = if residual_active
            ks_a = arnoldi_basis!(ws_a, At, r_adj; m=m_krylov)
            _eval_krylov_batch(ks_a, (1 .- nodes) .* dt, ks_a.beta)
        else
            nothing
        end

        for j in 1:M
            gval = zero(T)
            dval = zero(RT)
            @inbounds for q in 1:n_quad
                mul!(buf, dirs[j], view(F_k, :, q))
                dval += weights[q] * real(dot(buf, buf))
                if residual_active
                    gval -= weights[q] * dot(view(Λ_k, :, q), buf)
                end
            end
            residual_active && (grad[j] += real(dt * gval))
            fisher[j] += dt^2 * dval
        end
    end

    return grad, fisher, loss
end

"""Negative-log-likelihood gradient for transition counts, threaded by start state."""
function nll_gradient(A, dirs, counts, dt; m_krylov=25, n_quad=5)
    M      = length(dirs)
    n      = size(A, 1)
    starts = collect(keys(counts))
    K      = length(starts)

    grad_parts = Vector{Vector{Float64}}(undef, K)
    loss_parts = Vector{Float64}(undef, K)

    Threads.@threads for k in 1:K
        i    = starts[k]
        js   = counts[i]
        ei   = zeros(Float64, n);  ei[i] = 1.0

        ρ    = expv(real(dt), A, ei; m=m_krylov)
        ρ_r  = max.(real.(ρ), 1e-30)

        # NLL contribution from this starting cell
        loss_parts[k] = -sum(log(ρ_r[j]) for j in js)

        # Upstream gradient: Δᵢ = -∑_{j∈counts[i]} eⱼ/ρᵢ[j]
        Δ = zeros(Float64, n)
        for j in js;  Δ[j] -= 1.0 / ρ_r[j];  end

        grad_parts[k] = krylov_frechet_vjp(A, dirs, ei, real(dt), Δ;
                                           m_krylov=m_krylov, n_quad=n_quad)
    end

    n_obs = sum(length(counts[i]) for i in starts)
    loss  = sum(loss_parts) / max(n_obs, 1)
    grad  = reduce(+, grad_parts) ./ max(n_obs, 1)
    return grad, loss
end
