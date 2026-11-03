using Test
using LinearAlgebra
using SparseArrays
using Random
using Optim
using AdjointKrylovGradients

# The selected support must be a property of the data and the library, not of
# the optimizer's path. A birth-death chain with a correlated pair of death
# terms (linear and quadratic in the count) is the case where thresholding a
# partially converged iterate picks the wrong one.
@testset "STLS support is invariant to the line search" begin
    n = 60
    births(rate) = spdiagm(-1 => [rate(i) for i in 0:n-1]) - spdiagm(0 => [rate(i) for i in 0:n])
    deaths(rate) = spdiagm(1 => [rate(i) for i in 1:n]) - spdiagm(0 => [rate(i) for i in 0:n])
    lib = SparseMatrixCSC{Float64,Int}[]
    for rate in (i -> 1.0, i -> float(i), i -> float(i)^2 / 10)
        push!(lib, births(rate))
        push!(lib, deaths(rate))
    end
    θ_true = [2.0, 0.0, 0.0, 0.3, 0.0, 0.0]
    A_true = assemble_generator(spzeros(n + 1, n + 1), lib, θ_true)
    p0 = zeros(n + 1); p0[1] = 1.0
    dts = fill(0.5, 8)
    snaps = [p0]
    for dt in dts
        push!(snaps, exp(Matrix(A_true) * dt) * snaps[end])
    end

    line_searches = (Optim.LineSearches.HagerZhang(),
                     Optim.LineSearches.MoreThuente(),
                     Optim.LineSearches.BackTracking())
    starts = (fill(0.5, 6), 0.1 .+ rand(MersenneTwister(1), 6))
    for θ0 in starts, ls in line_searches
        r = sparse_generator_stls(spzeros(n + 1, n + 1), lib, snaps, dts, θ0;
                                  threshold_fraction=0.08, outer_iter=12,
                                  n_quad=16, m_krylov=30, fisher_precond=true,
                                  enforce_positive=true, verbose=false,
                                  optimizer=LBFGS(linesearch=ls))
        @test r.support == [1, 4]
        @test norm(r.theta[[1, 4]] - θ_true[[1, 4]]) / norm(θ_true) < 1e-6
    end
end
