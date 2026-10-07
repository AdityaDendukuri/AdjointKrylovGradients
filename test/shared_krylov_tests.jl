using Test
using LinearAlgebra
using SparseArrays
using Random
using Zygote
using ChainRulesCore
using AdjointKrylovGradients
const AKG = AdjointKrylovGradients

central_difference(f, x; h=1e-6) = (f(x + h) - f(x - h)) / (2h)

function parameter_finite_difference(f, θ; h=1e-6)
    g = zeros(Float64, length(θ))
    for j in eachindex(θ)
        θp = copy(θ)
        θm = copy(θ)
        θp[j] += h
        θm[j] -= h
        g[j] = (f(θp) - f(θm)) / (2h)
    end
    g
end

@testset "Reactive SINDy nonnegative elastic net" begin
    X = [
        1.0  0.0  0.5
        0.0  1.0  0.5
        1.0  1.0  0.0
        2.0  0.5  1.0
        0.5  2.0  1.0
    ]
    β_true = [1.2, 0.0, 0.7]
    y = X * β_true
    stats = reactive_sindy_regression_stats(X, y)
    β = nonnegative_elastic_net(
        stats; alpha=1e-10, l1_ratio=0.5, tol=1e-12,
    )
    @test all(β .>= 0)
    @test isapprox(β, β_true; rtol=2e-7, atol=2e-8)

    # Opposite stoichiometric columns make the raw design rank deficient.
    # The constrained regularized solution must remain finite and select the
    # column with the physically correct sign.
    x = collect(range(0.2, 2.0; length=40))
    X_rank_deficient = hcat(x, -x, x)
    y_rank_deficient = 0.8 .* x
    rank_stats = reactive_sindy_regression_stats(
        X_rank_deficient, y_rank_deficient,
    )
    β_rank = nonnegative_elastic_net(
        rank_stats; alpha=1e-4, l1_ratio=0.9,
    )
    @test all(isfinite, β_rank)
    @test all(β_rank .>= 0)
    @test β_rank[2] == 0.0
    @test norm(X_rank_deficient * β_rank - y_rank_deficient) / norm(y_rank_deficient) < 1e-3
end

@testset "Sparse Lindblad operators" begin
    for n in (2, 3), site in 1:n
        dense_op = nsite_op(σx, site, n)
        sparse_op = nsite_op_sparse(σx, site, n)
        @test issparse(sparse_op)
        @test Matrix(sparse_op) == dense_op
    end

    H = nsite_op(σx, 1, 2) * nsite_op(σx, 2, 2) +
        0.3 .* nsite_op(σz, 1, 2)
    H_sparse = sparse(H)
    LH_dense = hamiltonian_superop(H)
    LH_sparse = hamiltonian_superop_sparse(H_sparse)
    @test issparse(LH_sparse)
    @test Matrix(LH_sparse) == LH_dense

    L = nsite_op(σz, 2, 2)
    LD_dense = lindblad_superop(L)
    LD_sparse = lindblad_superop_sparse(sparse(L))
    @test issparse(LD_sparse)
    @test Matrix(LD_sparse) == LD_dense
end

@testset "Batched reduced exponential" begin
    Random.seed!(20260723)
    ts64 = AKG.GL_NODES_8 .* 0.4

    for T in (Float64, ComplexF64)
        n = 28
        A = T.(0.15 .* randn(n, n))
        for index in 1:n
            A[index, index] -= T(1.5)
        end
        if T <: Complex
            A .+= T.(0.08im .* randn(n, n))
        end
        v = T.(randn(n))
        if T <: Complex
            v .+= T.(0.1im .* randn(n))
        end
        basis = arnoldi_basis(sparse(A), v; m=20)
        optimized = AKG._eval_krylov_batch(
            basis, ts64, basis.beta,
        )
        reference = AKG._eval_krylov_batch_dense(
            basis, ts64, basis.beta,
        )
        @test isapprox(optimized, reference; rtol=2e-11, atol=2e-12)
    end

    let
        T = Float32
        n = 24
        A = T.(0.12 .* randn(n, n))
        for index in 1:n
            A[index, index] -= T(1.2)
        end
        v = T.(randn(n))
        basis = arnoldi_basis(sparse(A), v; m=18, tol=1f-6)
        ts32 = Float32.(ts64)
        optimized = AKG._eval_krylov_batch(
            basis, ts32, basis.beta,
        )
        reference = AKG._eval_krylov_batch_dense(
            basis, ts32, basis.beta,
        )
        @test eltype(optimized) == Float32
        @test isapprox(optimized, reference; rtol=2e-4, atol=2e-5)
    end

    # A defective reduced matrix must reject diagonalization and use the
    # original dense exponential path.
    let
        n = 12
        H = -Matrix{Float64}(I, n, n)
        for index in 1:(n - 1)
            H[index, index + 1] = 1.0
        end
        e1 = zeros(n)
        e1[1] = 1.0
        @test isnothing(AKG._eval_reduced_exponential_batch(H, ts64, e1))

        basis = (V=Matrix{Float64}(I, n, n), H=H, beta=1.0, m=n)
        optimized = AKG._eval_krylov_batch(basis, ts64, 1.0)
        reference = AKG._eval_krylov_batch_dense(basis, ts64, 1.0)
        @test optimized == reference
    end
end

@testset "Full-dimensional Arnoldi" begin
    Random.seed!(20260731)
    n = 12
    A = 0.08 .* randn(n, n)
    A .-= 1.4 .* Matrix{Float64}(I, n, n)
    v = randn(n)
    dt = 0.7

    basis = arnoldi_basis(sparse(A), v; m=n)
    approximation = AKG._eval_krylov_batch(
        basis, [dt], basis.beta,
    )[:, 1]
    reference = exp(A * dt) * v

    @test basis.m == n
    @test isapprox(
        approximation, reference; rtol=5e-12, atol=5e-13,
    )
end

@testset "Compressed reduced sensitivity core" begin
    @testset "real VJP and certified compression" begin
        n = 14
        A0 = spdiagm(
            -1 => fill(0.22, n - 1),
             0 => fill(-1.10, n),
             1 => fill(0.48, n - 1),
        )
        dirs = [
            spdiagm(0 => collect(range(-0.25, 0.30; length=n))),
            spdiagm(-1 => fill(0.11, n - 1)),
            spdiagm(1 => collect(range(0.03, 0.09; length=n - 1))),
        ]
        θ = [0.12, -0.08, 0.16]
        A = A0 + sum(θ[j] * dirs[j] for j in eachindex(θ))
        p0 = collect(range(0.4, 1.3; length=n))
        p0 ./= norm(p0)
        Δ = collect(range(-0.6, 0.7; length=n))
        dt = 0.25

        full = compressed_core_frechet_vjp(
            A, dirs, p0, dt, Δ;
            m_krylov=n - 1,
            core_rtol=0.0,
            return_diagnostics=true,
        )
        objective(θ_arg) = dot(
            Δ,
            exp(
                Matrix(A0 + sum(
                    θ_arg[j] * dirs[j] for j in eachindex(θ_arg)
                )) * dt,
            ) * p0,
        )
        reference = parameter_finite_difference(
            objective, θ; h=2e-6,
        )
        @test isapprox(
            full.gradient, reference; rtol=2e-7, atol=2e-9,
        )
        @test full.core_rank == full.core_full_rank
        @test full.core_relative_tail <= 10eps()

        compressed = compressed_core_frechet_vjp(
            A, dirs, p0, dt, Δ;
            m_krylov=n - 1,
            core_rtol=1e-3,
            return_diagnostics=true,
        )
        cached = compressed_core_frechet_vjp(
            A, dirs, p0, dt, Δ;
            m_krylov=n - 1,
            core_rtol=1e-3,
            direction_cache=SparseDirectionCache(dirs),
            return_diagnostics=true,
        )
        @test compressed.core_rank <= full.core_rank
        @test compressed.core_relative_tail <= 1.01e-3
        @test cached.core_rank == compressed.core_rank
        @test isapprox(
            cached.gradient,
            compressed.gradient;
            rtol=2e-13,
            atol=2e-14,
        )
        for index in eachindex(dirs)
            @test abs(
                compressed.gradient[index] - full.gradient[index],
            ) <= compressed.direction_error_bounds[index] + 2e-12
        end
    end

    @testset "complex VJP" begin
        n = 10
        A0 = spdiagm(
            -1 => fill(0.07 + 0.03im, n - 1),
             0 => ComplexF64.(
                -0.6 .+ 0.15im .* collect(range(-1, 1; length=n)),
            ),
             1 => fill(0.04 - 0.05im, n - 1),
        )
        dirs = SparseMatrixCSC{ComplexF64,Int}[
            spdiagm(
                0 => ComplexF64.(
                    collect(range(-0.15, 0.18; length=n)),
                ),
            ),
            spdiagm(1 => fill(0.02 + 0.04im, n - 1)),
        ]
        θ = [0.11, -0.09]
        A = A0 + sum(θ[j] * dirs[j] for j in eachindex(θ))
        p0 = ComplexF64.(collect(range(0.3, 1.1; length=n))) .+
             0.08im .* collect(range(-0.4, 0.5; length=n))
        Δ = ComplexF64.(collect(range(-0.5, 0.6; length=n))) .+
            0.12im .* collect(range(0.4, -0.3; length=n))
        dt = 0.2

        gradient = compressed_core_frechet_vjp(
            A, dirs, p0, dt, Δ;
            m_krylov=n - 1,
            core_rtol=0.0,
            direction_cache=SparseDirectionCache(dirs),
        )
        objective(θ_arg) = real(dot(
            Δ,
            exp(
                Matrix(A0 + sum(
                    θ_arg[j] * dirs[j] for j in eachindex(θ_arg)
                )) * dt,
            ) * p0,
        ))
        reference = parameter_finite_difference(
            objective, θ; h=2e-6,
        )
        @test isapprox(
            gradient, reference; rtol=2e-6, atol=2e-8,
        )
    end
end

@testset "SemigroupLayer pullback" begin
    @testset "compressed-core reverse rule is opt-in" begin
        n = 15
        A0 = spdiagm(
            -1 => fill(0.24, n - 1),
             0 => fill(-1.08, n),
             1 => fill(0.51, n - 1),
        )
        dirs = [
            spdiagm(0 => collect(range(-0.22, 0.31; length=n))),
            spdiagm(-1 => fill(0.09, n - 1)),
            spdiagm(1 => collect(range(0.02, 0.08; length=n - 1))),
        ]
        p0 = collect(range(0.5, 1.4; length=n))
        p0 ./= sum(p0)
        θ = [0.13, -0.07, 0.04]
        dt = 0.28
        Δ = collect(range(-0.6, 0.7; length=n))
        sl = SemigroupLayer(
            A0,
            dirs;
            m_krylov=n - 1,
            vjp_method=:compressed_core,
            core_rtol=0.0,
        )

        @test sl.vjp_method === :compressed_core
        @test SemigroupLayer(A0, dirs).vjp_method === :quadrature
        @test_throws ArgumentError SemigroupLayer(
            A0, dirs; vjp_method=:unknown,
        )
        @test_throws ArgumentError SemigroupLayer(
            A0, dirs; core_rtol=-1e-6,
        )

        _, pullback = ChainRulesCore.rrule(sl, p0, θ, dt)
        _, p̄, θ̄, dt̄ = pullback(Δ)
        p̄ = ChainRulesCore.unthunk(p̄)
        dt̄ = ChainRulesCore.unthunk(dt̄)

        A = A0 + sum(θ[j] * dirs[j] for j in eachindex(θ))
        objective(θ_arg, dt_arg=dt) =
            dot(Δ, sl(p0, θ_arg, dt_arg))
        θ̄_fd = parameter_finite_difference(objective, θ)
        dt̄_fd = central_difference(t -> objective(θ, t), dt)

        @test isapprox(
            p̄, exp(Matrix(A)' * dt) * Δ;
            rtol=2e-9, atol=2e-10,
        )
        @test isapprox(θ̄, θ̄_fd; rtol=3e-7, atol=3e-9)
        @test isapprox(dt̄, dt̄_fd; rtol=3e-7, atol=3e-9)
    end

    @testset "real state, parameters, and time" begin
        n = 18
        A0 = spdiagm(
            -1 => fill(0.30, n - 1),
             0 => fill(-1.20, n),
             1 => fill(0.70, n - 1),
        )
        dirs = [
            spdiagm(0 => collect(range(-0.3, 0.4; length=n))),
            spdiagm(1 => fill(0.2, n - 1)),
        ]
        p0 = collect(range(1.0, 2.0; length=n))
        p0 ./= sum(p0)
        θ = [0.2, -0.15]
        dt = 0.4
        Δ = collect(range(-0.5, 0.8; length=n))
        sl = SemigroupLayer(A0, dirs; m_krylov=n - 1, n_quad=8)

        _, pullback = ChainRulesCore.rrule(sl, p0, θ, dt)
        _, p̄, θ̄, dt̄ = pullback(Δ)
        p̄ = ChainRulesCore.unthunk(p̄)
        dt̄ = ChainRulesCore.unthunk(dt̄)

        A = A0 + sum(θ[j] * dirs[j] for j in eachindex(θ))
        p̄_ref = exp(Matrix(A)' * dt) * Δ
        objective(θ_arg, dt_arg=dt) = dot(Δ, sl(p0, θ_arg, dt_arg))
        θ̄_fd = parameter_finite_difference(objective, θ)
        dt̄_fd = central_difference(t -> objective(θ, t), dt)

        @test isapprox(p̄, p̄_ref; rtol=2e-9, atol=2e-10)
        @test isapprox(θ̄, θ̄_fd; rtol=2e-7, atol=2e-9)
        @test isapprox(dt̄, dt̄_fd; rtol=2e-7, atol=2e-9)
    end

    @testset "chained layers propagate through the intermediate state" begin
        n = 16
        A0 = spdiagm(
            -1 => fill(0.25, n - 1),
             0 => fill(-1.05, n),
             1 => fill(0.55, n - 1),
        )
        dirs = [
            spdiagm(0 => collect(range(-0.2, 0.3; length=n))),
            spdiagm(-1 => fill(0.12, n - 1)),
        ]
        sl = SemigroupLayer(A0, dirs; m_krylov=n - 1, n_quad=8)
        p0 = collect(range(0.5, 1.5; length=n))
        p0 ./= sum(p0)
        θ1 = [0.10, -0.08]
        θ2 = [-0.05, 0.12]
        dt = 0.25
        target = collect(range(-0.4, 0.6; length=n))

        loss_from_p0(p) = dot(target, sl(sl(p, θ1, dt), θ2, dt))
        p̄ = Zygote.gradient(loss_from_p0, p0)[1]

        A1 = A0 + sum(θ1[j] * dirs[j] for j in eachindex(θ1))
        A2 = A0 + sum(θ2[j] * dirs[j] for j in eachindex(θ2))
        p̄_ref = exp(Matrix(A1)' * dt) * exp(Matrix(A2)' * dt) * target
        @test isapprox(p̄, p̄_ref; rtol=2e-8, atol=2e-9)
    end

    @testset "complex Lindblad-like state" begin
        n = 8
        main = ComplexF64.(-0.35 .+ 0.2im .* collect(range(-1, 1; length=n)))
        A0 = spdiagm(
            -1 => fill(0.08 + 0.03im, n - 1),
             0 => main,
             1 => fill(0.05 - 0.04im, n - 1),
        )
        dirs = SparseMatrixCSC{ComplexF64,Int}[
            spdiagm(0 => ComplexF64.(collect(range(-0.2, 0.25; length=n)))),
            spdiagm(1 => fill(0.03 + 0.06im, n - 1)),
        ]
        p0 = ComplexF64.(collect(range(0.2, 1.0; length=n))) .+
             0.1im .* collect(range(-0.5, 0.5; length=n))
        θ = [0.15, -0.07]
        dt = 0.3
        Δ = ComplexF64.(collect(range(-0.4, 0.6; length=n))) .+
            0.2im .* collect(range(0.3, -0.2; length=n))
        sl = SemigroupLayer(A0, dirs; m_krylov=n - 1, n_quad=8)

        _, pullback = ChainRulesCore.rrule(sl, p0, θ, dt)
        _, p̄, θ̄, dt̄ = pullback(Δ)
        p̄ = ChainRulesCore.unthunk(p̄)
        dt̄ = ChainRulesCore.unthunk(dt̄)

        A = A0 + sum(θ[j] * dirs[j] for j in eachindex(θ))
        p̄_ref = exp(Matrix(A)' * dt) * Δ
        objective(θ_arg, dt_arg=dt) = real(dot(Δ, sl(p0, θ_arg, dt_arg)))
        θ̄_fd = parameter_finite_difference(objective, θ)
        dt̄_fd = central_difference(t -> objective(θ, t), dt)

        @test isapprox(p̄, p̄_ref; rtol=2e-8, atol=2e-9)
        @test isapprox(θ̄, θ̄_fd; rtol=2e-6, atol=2e-8)
        @test isapprox(dt̄, dt̄_fd; rtol=2e-6, atol=2e-8)
    end

    @testset "Float32 precision is preserved" begin
        n = 12
        A0 = spdiagm(
            -1 => fill(0.20f0, n - 1),
             0 => fill(-0.90f0, n),
             1 => fill(0.45f0, n - 1),
        )
        dirs = SparseMatrixCSC{Float32,Int}[
            spdiagm(0 => collect(range(-0.15f0, 0.20f0; length=n))),
            spdiagm(-1 => fill(0.08f0, n - 1)),
        ]
        p0 = collect(range(0.5f0, 1.5f0; length=n))
        p0 ./= sum(p0)
        θ = Float32[0.12, -0.06]
        dt = 0.35f0
        Δ = collect(range(-0.4f0, 0.6f0; length=n))
        sl = SemigroupLayer(A0, dirs; m_krylov=n - 1, n_quad=8)

        pred, pullback = ChainRulesCore.rrule(sl, p0, θ, dt)
        _, p̄, θ̄, dt̄ = pullback(Δ)
        p̄ = ChainRulesCore.unthunk(p̄)
        dt̄ = ChainRulesCore.unthunk(dt̄)

        @test eltype(pred) == Float32
        @test eltype(p̄) == Float32
        @test eltype(θ̄) == Float32
        @test dt̄ isa Float32

        A64 = Float64.(A0 + sum(θ[j] * dirs[j] for j in eachindex(θ)))
        p̄_ref = exp(Matrix(A64)' * Float64(dt)) * Float64.(Δ)
        Aadj = Matrix(A64)' * Float64(dt)
        X = Float64(dt) .* (Float64.(Δ) * Float64.(p0)')
        Z = zeros(Float64, n, n)
        G = exp([Aadj X; Z Aadj])[1:n, n+1:2n]
        θ̄_ref = [dot(G, Float64.(dir)) for dir in dirs]

        @test isapprox(Float64.(p̄), p̄_ref; rtol=3e-6, atol=3e-7)
        @test isapprox(Float64.(θ̄), θ̄_ref; rtol=3e-6, atol=3e-7)
    end

    @testset "proxy VJP matches ordinary reverse mode" begin
        n = 14
        A0 = spdiagm(
            -1 => fill(0.22, n - 1),
             0 => fill(-1.00, n),
             1 => fill(0.52, n - 1),
        )
        dirs = [
            spdiagm(0 => collect(range(-0.25, 0.30; length=n))),
            spdiagm(1 => fill(0.10, n - 1)),
            spdiagm(-1 => fill(-0.07, n - 1)),
        ]
        θ_eval = [0.10, -0.08, 0.05]
        θ_true = [0.16, -0.02, -0.03]
        dts = [0.2, 0.35]
        p0 = collect(range(0.6, 1.4; length=n))
        p0 ./= sum(p0)
        A_true = A0 + sum(θ_true[j] * dirs[j] for j in eachindex(θ_true))
        p1 = exp(Matrix(A_true) * dts[1]) * p0
        p2 = exp(Matrix(A_true) * dts[2]) * p1
        p_data = [p0, p1, p2]
        sl = SemigroupLayer(A0, dirs; m_krylov=n - 1, n_quad=8)

        function reverse_loss(θ)
            sum(
                0.5 * sum(abs2, p_data[t + 1] - sl(p_data[t], θ, dts[t]))
                for t in eachindex(dts)
            )
        end
        grad_reverse = Zygote.gradient(reverse_loss, θ_eval)[1]
        A_eval = A0 + sum(θ_eval[j] * dirs[j] for j in eachindex(θ_eval))
        grad_proxy, loss_proxy = shared_gradient(
            A_eval, dirs, p_data, dts;
            m_krylov=n - 1, n_quad=8,
        )
        sl_core = SemigroupLayer(
            A0,
            dirs;
            m_krylov=n - 1,
            vjp_method=:compressed_core,
            core_rtol=0.0,
        )
        reverse_core(θ) = sum(
            0.5 * sum(
                abs2,
                p_data[t + 1] - sl_core(p_data[t], θ, dts[t]),
            ) for t in eachindex(dts)
        )
        grad_reverse_core = Zygote.gradient(reverse_core, θ_eval)[1]
        grad_proxy_core, loss_proxy_core = shared_gradient(
            A_eval,
            dirs,
            p_data,
            dts;
            m_krylov=n - 1,
            vjp_method=:compressed_core,
            core_rtol=0.0,
        )

        @test isapprox(loss_proxy, reverse_loss(θ_eval); rtol=1e-11, atol=1e-13)
        @test isapprox(grad_proxy, grad_reverse; rtol=2e-9, atol=2e-11)
        @test isapprox(
            loss_proxy_core, reverse_core(θ_eval);
            rtol=1e-11, atol=1e-13,
        )
        @test isapprox(
            grad_proxy_core, grad_reverse_core;
            rtol=2e-9, atol=2e-11,
        )
    end
end
