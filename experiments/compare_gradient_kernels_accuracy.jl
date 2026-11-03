using AdjointKrylovGradients, LinearAlgebra, SparseArrays, Random, Printf

Random.seed!(0)
nmax = 40
X = [(i,) for i in 0:nmax]          # 1-species state space, states are tuples
R = [(1,), (-1,)]                   # net stoichiometry: birth, death
K = feature_dimension(1)
dA = build_derivatives(X, R; feature_map=quad_features, K=K)

θ = Dict(R[1] => zeros(K), R[2] => zeros(K))
θ[R[1]][1] = 1.5     # constant propensity -> birth
θ[R[2]][2] = 0.35    # linear propensity   -> death
θ_flat = flatten_params(θ)

A  = build_generator(X, R, θ; feature_map=quad_features)
p0 = zeros(length(X)); p0[1] = 1.0
dt = 0.4
P  = [p0]
for k in 1:6
    push!(P, exp(Matrix(A) * dt) * P[end])
end

# flatten dA into a direction vector matching flatten_params' sorted order
dirs = vcat([dA[z] for z in sort(collect(keys(dA)))]...)
@assert length(dirs) == length(θ_flat) "dirs $(length(dirs)) vs θ $(length(θ_flat))"

# evaluate away from the generating parameters so the residual is nonzero
θ_eval = copy(θ_flat)
θ_eval .*= 1.0 .+ 0.15 .* [(-1.0)^j for j in eachindex(θ_eval)]
θ_eval .+= 0.02

obj(v) = objective(P, R, unflatten_params(v, R, K), X, dt, 0.0)
h = 1e-6
fd = similar(θ_eval)
for j in eachindex(θ_eval)
    a = copy(θ_eval); b = copy(θ_eval); a[j] += h; b[j] -= h
    fd[j] = (obj(a) - obj(b)) / (2h)
end

println("problem: n=", length(X), "  params=", length(θ_flat), "  steps=", length(P)-1)
A_eval = build_generator(X, R, unflatten_params(θ_eval, R, K); feature_map=quad_features)
println("|fd| = ", norm(fd))
println("\n n_q |   gradient_adjoint.jl (CME) |  shared_krylov.jl (NeurIPS)")
println("-----+-----------------------------+---------------------------")
for nq in (5, 8, 12, 16, 24)
    g1 = similar(θ_eval)
    gradient_adjoint_krylov!(g1, θ_eval, P, R, X, dt, dA, 0.0; m_krylov=40, n_quadrature=nq)
    g2, _ = shared_gradient(A_eval, dirs, P, fill(dt, length(P)-1); m_krylov=40, n_quad=nq)
    @printf("%4d | %27.3e | %25.3e\n", nq, norm(g1-fd)/norm(fd), norm(g2-fd)/norm(fd))
end
