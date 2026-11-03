using AdjointKrylovGradients, LinearAlgebra, SparseArrays, Random, Printf

Random.seed!(0)
# 2-species grid with many candidate stoichiometries -> large MK
function setup(gridmax, nreac, nsteps; fmap=cubic_features)
    X = [(i,j) for i in 0:gridmax for j in 0:gridmax]
    allR = [(a,b) for a in -1:1 for b in -1:1 if !(a==0 && b==0)]
    R = allR[1:min(nreac, length(allR))]
    K = length(fmap(first(X)))
    dA = build_derivatives(X, R; feature_map=fmap, K=K)
    θ = Dict(z => zeros(K) for z in R)
    θ[R[1]][1] = 1.2; θ[R[2]][2] = 0.4
    θf = flatten_params(θ)
    A = build_generator(X, R, θ; feature_map=fmap)
    p0 = zeros(length(X)); p0[1] = 1.0
    dt = 0.25
    P = [p0]; for _ in 1:nsteps; push!(P, exp(Matrix(A)*dt)*P[end]); end
    θe = copy(θf); θe .+= 0.01
    Ae = build_generator(X, R, unflatten_params(θe, R, K); feature_map=fmap)
    dirs = vcat([dA[z] for z in sort(collect(keys(dA)))]...)
    (; X, R, K, dA, dirs, θe, Ae, P, dt, fmap)
end
bench(f) = (f(); minimum(@elapsed(f()) for _ in 1:5) * 1e3)

println("Julia threads: ", Threads.nthreads())
println("    n     MK | agreement | CME ms | shared ms | ratio")
println("-------------+-----------+--------+-----------+------")
for (g, nr) in ((12, 4), (12, 8), (20, 8))
    s = setup(g, nr, 8)
    g1 = similar(s.θe)
    f1 = () -> gradient_adjoint_krylov!(g1, s.θe, s.P, s.R, s.X, s.dt, s.dA, 0.0; feature_map=s.fmap, m_krylov=30, n_quadrature=16)
    f2 = () -> shared_gradient(s.Ae, s.dirs, s.P, fill(s.dt, length(s.P)-1); m_krylov=30, n_quad=16)
    f1(); g2, _ = f2()
    t1 = bench(f1); t2 = bench(f2)
    @printf("%5d %6d | %9.2e | %6.1f | %9.1f | %.2fx\n",
            length(s.X), length(s.θe), norm(g1-g2)/norm(g1), t1, t2, t2/t1)
end
