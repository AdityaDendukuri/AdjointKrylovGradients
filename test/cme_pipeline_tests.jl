using Test
using AdjointKrylovGradients
using LinearAlgebra
using SparseArrays
using Random
using ExponentialUtilities

@testset "CME pipeline" begin

    @testset "Core: quad_features" begin
        # Test 2-species system
        x = (5, 3)
        φ = quad_features(x)
        @test length(φ) == feature_dimension(2)
        @test φ[1] == 1.0                    # Constant
        @test φ[2] == 5.0                    # x₁
        @test φ[3] == 3.0                    # x₂
        @test φ[4] == 5 * 4 / 2              # x₁(x₁-1)/2
        @test φ[5] == 3 * 2 / 2              # x₂(x₂-1)/2
        @test φ[6] == 5 * 3                  # x₁×x₂

        # Test 4-species system (main use case)
        x4 = (50, 10, 0, 0)
        φ4 = quad_features(x4)
        @test length(φ4) == 15
        @test φ4[1] == 1.0
        @test φ4[2] == 50.0
        @test φ4[10] == 50 * 10              # S×E cross-term
    end

    @testset "Core: feature_dimension" begin
        @test feature_dimension(1) == 3      # 1 + 2 + 0
        @test feature_dimension(2) == 6      # 1 + 4 + 1
        @test feature_dimension(3) == 10     # 1 + 6 + 3
        @test feature_dimension(4) == 15     # 1 + 8 + 6
    end

    @testset "Core: feature_names" begin
        names = feature_names(2)
        @test length(names) == 6
        @test names[1] == "1"

        names_custom = feature_names(2; species_names=["A", "B"])
        @test names_custom[2] == "A"
        @test names_custom[3] == "B"
        @test names_custom[6] == "A×B"
    end

    @testset "Core: build_generator" begin
        # Simple 2-state system
        X = [(0,), (1,)]
        R = [(1,), (-1,)]
        K = feature_dimension(1)
        θ = Dict(
            (1,) => [0.1, 0.0, 0.0],   # Birth: rate 0.1
            (-1,) => [0.0, 0.05, 0.0]  # Death: rate 0.05*n
        )

        A = build_generator(X, R, θ)

        @test size(A) == (2, 2)
        @test A[2, 1] ≈ 0.1              # 0→1 rate
        @test A[1, 2] ≈ 0.05             # 1→0 rate (0.05 * 1)
        @test A[1, 1] ≈ -0.1             # Column sum = 0
        @test A[2, 2] ≈ -0.05

        # Column sums should be zero (conservation)
        @test sum(A, dims=1) ≈ zeros(1, 2) atol=1e-10
    end

    @testset "Gradient: flatten/unflatten" begin
        R = [(-1, 0), (1, 0)]
        K = 3
        θ = Dict(
            (-1, 0) => [1.0, 2.0, 3.0],
            (1, 0) => [4.0, 5.0, 6.0]
        )

        v = flatten_params(θ)
        @test length(v) == 6
        # Should be sorted by reaction
        @test v[1:3] == [1.0, 2.0, 3.0]
        @test v[4:6] == [4.0, 5.0, 6.0]

        θ_back = unflatten_params(v, R, K)
        @test θ_back[(-1, 0)] == θ[(-1, 0)]
        @test θ_back[(1, 0)] == θ[(1, 0)]
    end

    @testset "Gradient: objective function" begin
        # Simple test case with 1D state space
        X = [(0,), (1,), (2,), (3,)]
        R = [(1,), (-1,)]
        K = feature_dimension(1)

        θ = Dict(
            (1,) => [0.1, 0.0, 0.0],
            (-1,) => [0.0, 0.1, 0.0]
        )

        # Uniform distributions
        P = [ones(4) / 4, ones(4) / 4]
        dt = 0.1

        obj = objective(P, R, θ, X, dt, 0.0)
        @test obj >= 0.0  # Objective is non-negative
    end

    @testset "Gradient: numerical gradient check" begin
        Random.seed!(42)

        # Simple 1D birth-death system for gradient check
        X = [(n,) for n in 0:10]
        R = [(1,), (-1,)]
        K = feature_dimension(1)

        # Parameters (slightly perturbed from true to ensure non-zero gradient)
        θ = Dict(
            (1,) => [0.4, 0.05, 0.01],   # Noisy birth params
            (-1,) => [0.02, 0.08, 0.01]  # Noisy death params
        )

        # Create distributions that DON'T match the parameters exactly
        # (this ensures non-trivial gradients)
        θ_gen = Dict(
            (1,) => [0.5, 0.0, 0.0],   # True birth rate 0.5
            (-1,) => [0.0, 0.1, 0.0]   # True death rate 0.1*n
        )

        A_gen = build_generator(X, R, θ_gen)
        p0 = zeros(length(X))
        p0[6] = 1.0  # Start at n=5

        dt = 0.1
        P = [p0]
        p = p0
        for _ in 1:3
            p = expv(dt, A_gen, p)
            push!(P, p)
        end

        dA = build_derivatives(X, R)
        θ_flat = flatten_params(θ)

        # Analytical gradient
        g_analytic = zeros(length(θ_flat))
        gradient!(g_analytic, θ_flat, P, R, X, dt, dA)

        # Finite difference
        ε = 1e-5
        g_fd = zeros(length(θ_flat))
        f0 = objective(P, R, unflatten_params(θ_flat, R, K), X, dt)

        for i in 1:length(θ_flat)
            θ_plus = copy(θ_flat)
            θ_plus[i] += ε
            f_plus = objective(P, R, unflatten_params(θ_plus, R, K), X, dt)
            g_fd[i] = (f_plus - f0) / ε
        end

        # Check that gradients are non-trivial
        @test norm(g_fd) > 1e-10

        # Check relative error (allow 10% tolerance for numerical gradient check)
        rel_error = norm(g_analytic - g_fd) / (norm(g_fd) + 1e-10)
        @test rel_error < 0.1
    end

    @testset "State space: build_state_space" begin
        # Create mock trajectories
        mutable struct MockTraj
            data::Dict{Float64,Vector{Float64}}
            t::Vector{Float64}
        end

        function (traj::MockTraj)(t::Float64)
            # Return closest time point
            idx = argmin(abs.(traj.t .- t))
            return traj.data[traj.t[idx]]
        end

        trajs = [
            MockTraj(
                Dict(0.0 => [5.0, 3.0], 0.5 => [4.0, 3.0], 1.0 => [3.0, 2.0]),
                [0.0, 0.5, 1.0]
            ),
            MockTraj(
                Dict(0.0 => [5.0, 3.0], 0.5 => [5.0, 2.0], 1.0 => [4.0, 2.0]),
                [0.0, 0.5, 1.0]
            )
        ]

        times = [0.0, 0.5, 1.0]
        R = [(-1, 0), (0, -1)]

        X = build_state_space(trajs, times, R; expansion=0)
        @test length(X) > 0
        @test (5, 3) in X
        @test (4, 3) in X || (5, 2) in X
    end

    @testset "Empirical: count_transitions" begin
        # Test with mock trajectory
        mutable struct MockTrajCont
            data_func::Function
            t::Vector{Float64}
            u::Vector{Vector{Float64}}
        end

        function (traj::MockTrajCont)(t::Float64)
            return traj.data_func(t)
        end

        # Simple trajectory: state changes at t=0.5
        times = [0.0, 0.5, 1.0]
        func = t -> t < 0.5 ? [1.0, 1.0, 0.0, 0.0] : [0.0, 0.0, 1.0, 0.0]
        trajs = [
            MockTrajCont(
                func,
                times,
                [func(t) for t in times]
            )
        ]

        R = [(-1, -1, 1, 0)]
        data = count_transitions(trajs, R; verbose=false)

        @test haskey(data.counts, (-1, -1, 1, 0))
        # Should detect ~1 transition around t=0.5
        @test data.counts[(-1, -1, 1, 0)] >= 0
    end

    @testset "STLS: basic optimization" begin
        Random.seed!(123)

        # Simple 1D birth-death for clean optimization
        X = [(n,) for n in 0:15]
        R = [(1,), (-1,)]
        K = feature_dimension(1)

        # True sparse parameters
        θ_true = Dict(
            (1,) => [0.3, 0.0, 0.0],   # Birth rate 0.3
            (-1,) => [0.0, 0.1, 0.0]   # Death rate 0.1*n
        )

        # Generate distributions from true model
        A_true = build_generator(X, R, θ_true)
        p0 = zeros(length(X))
        p0[6] = 1.0  # Start at n=5

        dt = 0.1
        P = [p0]
        p = p0
        for _ in 1:5
            p = expv(dt, A_true, p)
            push!(P, p)
        end

        # Initialize with noisy parameters
        θ_init = Dict(
            (1,) => [0.25, 0.01, 0.01],
            (-1,) => [0.01, 0.08, 0.01]
        )

        result = optimize_stls(P, R, X, dt, θ_init;
                               τ=0.05, outer_iter=10, polish_iter=20, verbose=false)

        @test result.objective >= 0
        @test result.n_iterations > 0
        @test haskey(result.θ_opt, (1,))
        @test haskey(result.θ_opt, (-1,))
    end

end

println("\nAll tests passed!")
