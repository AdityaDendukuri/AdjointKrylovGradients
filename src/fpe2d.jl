using SparseArrays

# Builds the 2D FPE generator decomposition A(θ) = A₀ + Σⱼ θⱼ Aⱼ
#   ∂_t p = -∂_x(bx(x,y;θ)·p) - ∂_y(by(x,y;θ)·p) + D·(∂²_xx + ∂²_yy)p
#
# State ordering: column-major, lin(i,j) = i + (j-1)*n_x
# Discretization: uniform grids xs (n_x pts) and ys (n_y pts), reflecting BCs.
# Drift uses conservative centered differences (same scheme as fpe.jl).
# Valid Markov generator when D/h² > max|b_k|/(2h) for each direction.
function fpe_library_2d(xs::AbstractVector, ys::AbstractVector,
                        drift_fns_x, drift_fns_y; D=1.0)
    n_x = length(xs);  n_y = length(ys)
    n   = n_x * n_y
    h_x = (xs[end] - xs[1]) / (n_x - 1)
    h_y = (ys[end] - ys[1]) / (n_y - 1)

    lin(i, j) = i + (j - 1) * n_x   # 1-indexed, column-major

    # ── Diffusion: D*(∂²_xx + ∂²_yy) with reflecting BCs ─────────────────────
    ri = Int[]; ci = Int[]; vi = Float64[]
    for j in 1:n_y, i in 1:n_x
        l = lin(i, j);  d = 0.0
        if i > 1;   push!(ri, lin(i-1,j)); push!(ci,l); push!(vi, D/h_x^2); d -= D/h_x^2; end
        if i < n_x; push!(ri, lin(i+1,j)); push!(ci,l); push!(vi, D/h_x^2); d -= D/h_x^2; end
        if j > 1;   push!(ri, lin(i,j-1)); push!(ci,l); push!(vi, D/h_y^2); d -= D/h_y^2; end
        if j < n_y; push!(ri, lin(i,j+1)); push!(ci,l); push!(vi, D/h_y^2); d -= D/h_y^2; end
        push!(ri, l); push!(ci, l); push!(vi, d)
    end
    A₀ = sparse(ri, ci, vi, n, n)

    # ── Drift directions ───────────────────────────────────────────────────────
    @assert length(drift_fns_x) == length(drift_fns_y)
    dirs = map(1:length(drift_fns_x)) do k
        bx_fn = drift_fns_x[k];  by_fn = drift_fns_y[k]
        rd = Int[]; cd = Int[]; vd = Float64[]
        for j in 1:n_y, i in 1:n_x
            l  = lin(i, j)
            bx = bx_fn(xs[i], ys[j])
            by = by_fn(xs[i], ys[j])

            # x-drift off-diagonals
            if i < n_x; push!(rd, lin(i+1,j)); push!(cd,l); push!(vd,  bx/(2h_x)); end
            if i > 1;   push!(rd, lin(i-1,j)); push!(cd,l); push!(vd, -bx/(2h_x)); end
            # y-drift off-diagonals
            if j < n_y; push!(rd, lin(i,j+1)); push!(cd,l); push!(vd,  by/(2h_y)); end
            if j > 1;   push!(rd, lin(i,j-1)); push!(cd,l); push!(vd, -by/(2h_y)); end

            # Reflecting BCs: fold boundary flux back onto diagonal
            if i == 1;   push!(rd,l); push!(cd,l); push!(vd, -bx/(2h_x)); end
            if i == n_x; push!(rd,l); push!(cd,l); push!(vd,  bx/(2h_x)); end
            if j == 1;   push!(rd,l); push!(cd,l); push!(vd, -by/(2h_y)); end
            if j == n_y; push!(rd,l); push!(cd,l); push!(vd,  by/(2h_y)); end
        end
        sparse(rd, cd, vd, n, n)
    end

    return A₀, dirs
end
