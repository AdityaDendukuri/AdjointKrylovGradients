using AdjointKrylovGradients, LinearAlgebra, SparseArrays, Random, Printf, Statistics

Random.seed!(0)
function setup(nmax, nsteps)
    X = [(i,) for i in 0:nmax]
    R = [(1,), (-1,)]
    K = feature_dimension(1)
    dA = build_derivatives(X, R; feature_map=quad_features, K=K)
    θ = Dict(R[1] => zeros(K), R[2] => zeros(K))
    θ[R[1]][1] = 1.5; θ[R[2]][2] = 0.35
    θf = flatten_params(θ)
    A = build_generator(X, R, θ; feature_map=quad_features)
    p0 = zeros(length(X)); p0[1] = 1.0
    dt = 0.4
    P = [p0]; for _ in 1:nsteps; push!(P, exp(Matrix(A)*dt)*P[end]); end
    θe = copy(θf); θe .*= 1.0 .+ 0.15 .* [(-1.0)^j for j in eachindex(θe)]; θe .+= 0.02
    Ae = build_generator(X, R, unflatten_params(θe, R, K); feature_map=quad_features)
    dirs = vcat([dA[z] for z in sort(collect(keys(dA)))]...)
    (; X, R, K, dA, dirs, θe, Ae, P, dt)
end

bench(f) = (f(); minimum(@elapsed(f()) for _ in 1:7) * 1e3)

println(" n    steps | agreement |gA-gS|/|gA| | CME ms | shared ms")
println("----------- +-----------------------+--------+----------")
for (nmax, nsteps) in ((40,6), (200,10), (800,10), (2000,10))
    s = setup(nmax, nsteps)
    g1 = similar(s.θe)
    f1 = () -> gradient_adjoint_krylov!(g1, s.θe, s.P, s.R, s.X, s.dt, s.dA, 0.0; m_krylov=30, n_quadrature=16)
    f2 = () -> shared_gradient(s.Ae, s.dirs, s.P, fill(s.dt, length(s.P)-1); m_krylov=30, n_quad=16)
    f1(); g2, _ = f2()
    @printf("%5d %5d | %21.3e | %6.2f | %8.2f\n",
            nmax+1, nsteps, norm(g1-g2)/norm(g1), bench(f1), bench(f2))
end
