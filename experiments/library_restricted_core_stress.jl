"""Stress test: global Frechet rank versus library-restricted quadrature order.

The generator is block diagonal.  Each two-state block has the same nilpotent
local dynamics but a different decay rate.  The library contains only local,
noncommuting block directions.  Cross-block entries make the full Frechet
matrix high rank, although they are invisible to every requested gradient.
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: SparseDirectionCache, real_scalar_type,
    ArnoldiWorkspace, arnoldi_basis!, _eval_krylov_batch
using LinearAlgebra, SparseArrays, Printf, Statistics
using FastGaussQuadrature: gausslegendre

include("optimized_shared_krylov_utils.jl")

function problem(blocks; dt=1.0, frequency_span=40.0, coupling=0.0)
    n = 2blocks
    A = zeros(ComplexF64, n, n)
    y = zeros(ComplexF64, n)
    z = zeros(ComplexF64, n)
    dirs = SparseMatrixCSC{ComplexF64,Int}[]
    N = ComplexF64[0.0 1.0; 0.0 0.0]
    for b in 1:blocks
        I = (2b-1):(2b)
        omega = frequency_span * (b-1) / max(blocks-1, 1)
        shift = -0.05 + im*omega
        A[I, I] .= shift .* Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) .+ N
        y[I] .= normalize([1.0 + 0.13sin(b), 0.7 + 0.11cos(2b)]) / sqrt(blocks)
        z[I] .= normalize([0.8 + 0.09cos(b), -0.6 + 0.07sin(3b)]) / sqrt(blocks)
        for D in ([0.0 1.0; 0.0 0.0], [0.0 0.0; 1.0 0.0], [1.0 0.0; 0.0 -1.0])
            rows, cols, vals = Int[], Int[], ComplexF64[]
            for j in 1:2, i in 1:2
                iszero(D[i,j]) && continue
                push!(rows, first(I)+i-1)
                push!(cols, first(I)+j-1)
                push!(vals, D[i,j])
            end
            push!(dirs, sparse(rows, cols, vals, n, n))
        end
    end
    # Weak nearest-neighbour coupling removes exact block separability while
    # retaining a local operator library.
    for b in 1:(blocks-1)
        I = (2b-1):(2b)
        J = (2b+1):(2b+2)
        A[I, J] .+= coupling .* Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
        A[J, I] .-= coupling .* Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
    end
    A, y, z, dirs, dt
end

function exact_frechet_matrix(A, y, z, dt)
    n = length(y)
    B = zeros(eltype(A), 2n, 2n)
    B[1:n, 1:n] .= A'
    B[n+1:end, n+1:end] .= A'
    B[1:n, n+1:end] .= z * y'
    E = exp(dt .* B)
    Matrix(E[1:n, n+1:end])
end

function library_gradient_from_matrix(G, dirs)
    [dot(G, Matrix(D)) for D in dirs]
end

function quadrature_gradient(A, y, z, dirs, dt, q)
    x, w = gausslegendre(q)
    nodes, weights = (x .+ 1) ./ 2, w ./ 2
    g = zeros(ComplexF64, length(dirs))
    for (s, weight) in zip(nodes, weights)
        f = exp((dt*s) .* A) * y
        lambda = exp((dt*(1-s)) .* A') * z
        for j in eachindex(dirs)
            g[j] += dt * weight * dot(lambda, dirs[j] * f)
        end
    end
    g
end

function rank_for_relative_frobenius(s, tol)
    total = sum(abs2, s)
    tail = total
    for r in 0:length(s)
        sqrt(max(tail, 0.0) / total) <= tol && return r
        r < length(s) && (tail -= abs2(s[r+1]))
    end
    length(s)
end

function run(blocks; coupling=0.0)
    A, y, z, dirs, dt = problem(blocks; frequency_span=4blocks, coupling=coupling)
    G = exact_frechet_matrix(A, y, z, dt)
    exact = library_gradient_from_matrix(G, dirs)
    F = svd(G)
    @printf("blocks=%3d n=%3d directions=%3d coupling=%.2g numerical ranks:",
        blocks, 2blocks, length(dirs), coupling)
    for tol in (1e-4, 1e-6, 1e-8, 1e-10)
        @printf(" r(%.0e)=%d", tol, rank_for_relative_frobenius(F.S, tol))
    end
    println()
    for q in 1:8
        approx = quadrature_gradient(A, y, z, dirs, dt, q)
        rel = norm(approx-exact) / norm(exact)
        maxrel = maximum(abs.(approx-exact)) / maximum(abs.(exact))
        @printf("  q=%d library rel2=%.3e max-scaled=%.3e\n", q, rel, maxrel)
    end
    # Show what global truncated SVD accuracy buys in the requested library.
    for r in unique(filter(<=(length(F.S)), [1, 2, 4, 8, 16, 32, 64]))
        Gr = F.U[:,1:r] * Diagonal(F.S[1:r]) * F.V[:,1:r]'
        approx = library_gradient_from_matrix(Gr, dirs)
        global_rel = norm(G-Gr) / norm(G)
        library_rel = norm(approx-exact) / norm(exact)
        @printf("  svd r=%2d global=%.3e library=%.3e\n", r, global_rel, library_rel)
    end
end

function benchmark_matched(blocks; coupling=0.05, q=6, repeats=7)
    A_dense, y, z, dirs, dt = problem(blocks;
        frequency_span=4blocks, coupling=coupling)
    A = sparse(A_dense)
    n = length(y)
    nodes, weights = AdjointKrylovGradients.gauss_legendre_01(q)
    qwork = OptimizedQuadratureWorkspace(dirs, n, n, q)
    quad = () -> optimized_linear_loss_and_vjp!(
        qwork, A, y, z, dt, nodes, weights; m_krylov=n,
    ).parameter_vjp
    cache = SparseDirectionCache(dirs)
    core = () -> AdjointKrylovGradients.compressed_core_frechet_vjp(
        A, dirs, y, dt, z; m_krylov=n, core_rtol=1e-8,
        direction_cache=cache, return_diagnostics=true,
    )
    exact = real.(library_gradient_from_matrix(
        exact_frechet_matrix(A_dense, y, z, dt), dirs))
    qvalue = copy(quad())
    cresult = core()
    qtimes = [@elapsed(quad()) for _ in 1:repeats]
    ctimes = [@elapsed(core()) for _ in 1:repeats]
    @printf("TIMING blocks=%d n=%d q=%d r=%d quad=%.3fms core=%.3fms speedup=%.2fx qerr=%.2e cerr=%.2e\n",
        blocks, n, q, cresult.core_rank, 1e3median(qtimes), 1e3median(ctimes),
        median(ctimes)/median(qtimes), norm(qvalue-exact)/norm(exact),
        norm(cresult.gradient-exact)/norm(exact))
end

function wide_library_problem(groups, width; coupling=0.05)
    iseven(width) || error("width must be even")
    n = groups * width
    N = spzeros(ComplexF64, width, width)
    for i in 1:(width÷2)
        N[i, width÷2+i] = 1
    end
    A = spzeros(ComplexF64, n, n)
    y = zeros(ComplexF64, n)
    z = zeros(ComplexF64, n)
    dirs = SparseMatrixCSC{ComplexF64,Int}[]
    for b in 1:groups
        idx = ((b-1)*width+1):(b*width)
        omega = 4groups * (b-1) / max(groups-1, 1)
        A[idx,idx] .= (-0.05 + im*omega) .* sparse(Matrix{ComplexF64}(LinearAlgebra.I, width, width)) .+ N
        for k in 1:width
            y[idx[k]] = (1 + 0.1sin(k+b)) / sqrt(n)
            z[idx[k]] = (cos(0.3k+b) + im*sin(0.2k-b)) / sqrt(n)
        end
        for pattern in 1:3
            rows, cols, vals = Int[], Int[], ComplexF64[]
            for (jj,j) in enumerate(idx), (ii,i) in enumerate(idx)
                value = pattern == 1 ? complex(cos(0.17ii+0.11jj), sin(0.07ii-0.13jj)) :
                        pattern == 2 ? complex(cos(0.05ii*jj), sin(0.03(ii+jj))) :
                                       complex(sin(0.09ii-0.04jj), cos(0.06ii+0.08jj))
                push!(rows, i); push!(cols, j); push!(vals, value/sqrt(width))
            end
            push!(dirs, sparse(rows, cols, vals, n, n))
        end
    end
    for b in 1:(groups-1)
        I1 = ((b-1)*width+1):(b*width)
        I2 = (b*width+1):((b+1)*width)
        for k in 1:width
            A[I1[k],I2[k]] += coupling
            A[I2[k],I1[k]] -= coupling
        end
    end
    A, y, z, dirs
end

function benchmark_wide_library(width; groups=20, coupling=0.05, q=6, repeats=5)
    A, y, z, dirs = wide_library_problem(groups, width; coupling=coupling)
    n, m, dt = length(y), 2groups, 1.0
    nodes, weights = AdjointKrylovGradients.gauss_legendre_01(q)
    qwork = OptimizedQuadratureWorkspace(dirs, n, m, q)
    quad = () -> optimized_linear_loss_and_vjp!(
        qwork, A, y, z, dt, nodes, weights; m_krylov=m,
    ).parameter_vjp
    cache = SparseDirectionCache(dirs)
    core = () -> AdjointKrylovGradients.compressed_core_frechet_vjp(
        A, dirs, y, dt, z; m_krylov=m, core_rtol=1e-8,
        direction_cache=cache, return_diagnostics=true,
    )
    qvalue = copy(quad())
    cresult = core()
    qtimes = [@elapsed(quad()) for _ in 1:repeats]
    ctimes = [@elapsed(core()) for _ in 1:repeats]
    @printf("WIDE groups=%d width=%d n=%d M=%d m=%d q=%d r=%d quad=%.3fms core=%.3fms speedup=%.2fx diff=%.2e\n",
        groups, width, n, length(dirs), m, q, cresult.core_rank,
        1e3median(qtimes), 1e3median(ctimes), median(ctimes)/median(qtimes),
        norm(qvalue-cresult.gradient)/norm(cresult.gradient))
end

if get(ENV, "STRESS_TIMING_ONLY", "0") != "1" &&
   get(ENV, "STRESS_WIDE_ONLY", "0") != "1"
    for coupling in (0.0, 0.05, 0.2)
        for blocks in (10, 20, 40)
            run(blocks; coupling=coupling)
        end
    end
end


if get(ENV, "STRESS_WIDE_ONLY", "0") != "1"
    for blocks in (10, 20, 40)
        benchmark_matched(blocks)
    end
end

for width in (8, 16, 32, 64, 128, 256)
    benchmark_wide_library(width)
end

for width in (64, 128)
    benchmark_wide_library(width; groups=40, repeats=3)
end
