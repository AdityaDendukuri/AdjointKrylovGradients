using Test
using LinearAlgebra
using SparseArrays
using Random
using AdjointKrylovGradients

@testset "Sparse-direct rational exponential gradient" begin
    rng = MersenneTwister(20261010)
    n = 18
    lower = 0.4 .+ rand(rng, n - 1)
    upper = 0.3 .+ rand(rng, n - 1)
    A = spdiagm(
        -1 => lower,
         0 => -vcat(lower, 0.0) .- vcat(0.0, upper),
         1 => upper,
    )
    directions = [
        sparse([j + 1, j], [j, j], [1.0, -1.0], n, n)
        for j in (2, 6, 11)
    ]
    v = rand(rng, n)
    v ./= sum(v)
    upstream = randn(rng, n)
    dt = 0.7

    result = sparse_direct_exponential_vjp(
        A, directions, v, dt, upstream,
    )
    dense_prediction = exp(dt .* Matrix(A)) * v
    dense_state_vjp = exp(dt .* Matrix(A')) * upstream
    @test isapprox(
        result.prediction, dense_prediction; rtol=2e-13, atol=2e-14,
    )
    @test isapprox(
        result.state_vjp, dense_state_vjp; rtol=2e-13, atol=2e-14,
    )

    zero_block = zeros(n, n)
    dense_gradient = [
        dot(
            upstream,
            exp(dt .* [Matrix(A) Matrix(D); zero_block Matrix(A)])[1:n, n+1:2n] * v,
        ) for D in directions
    ]
    @test isapprox(
        result.parameter_vjp, dense_gradient; rtol=3e-12, atol=3e-13,
    )
    @test result.factorizations == 8
    @test isapprox(
        sparse_direct_exponential(A, v, dt), dense_prediction;
        rtol=2e-13, atol=2e-14,
    )
end
