using AdjointKrylovGradients, LinearAlgebra, SparseArrays, Random, Printf

Random.seed!(1)
# birth-death generator on 0..n, with an overcomplete library of candidate rates
n = 60
births(rate_fn) = spdiagm(-1 => [rate_fn(i) for i in 0:n-1]) - spdiagm(0 => [rate_fn(i) for i in 0:n])
deaths(rate_fn) = spdiagm( 1 => [rate_fn(i) for i in 1:n]) - spdiagm(0 => [rate_fn(i) for i in 0:n])

lib = SparseMatrixCSC{Float64,Int}[]
labels = String[]
for (nm, f) in (("const", i->1.0), ("lin", i->float(i)), ("quad", i->float(i)^2/10))
    push!(lib, births(f)); push!(labels, "birth·"*nm)
    push!(lib, deaths(f)); push!(labels, "death·"*nm)
end
M = length(lib)

θ_true = zeros(M)
θ_true[1] = 2.0    # birth·const
θ_true[4] = 0.30   # death·lin
A_true = assemble_generator(spzeros(n+1, n+1), lib, θ_true)

p0 = zeros(n+1); p0[1] = 1.0
dts = fill(0.5, 8)
snaps = [p0]
for dt in dts; push!(snaps, exp(Matrix(A_true)*dt) * snaps[end]); end

θ0 = fill(0.5, M)   # uninformative start, all directions active
res = sparse_generator_stls(spzeros(n+1, n+1), lib, snaps, dts, θ0;
                            threshold_fraction=0.08, outer_iter=12,
                            n_quad=16, m_krylov=30,
                            fisher_precond=true, enforce_positive=true)

println("\n direction        true      recovered")
for j in 1:M
    @printf("  %-14s %8.4f  %10.5f%s\n", labels[j], θ_true[j], res.theta[j],
            j in res.support ? "  <- selected" : "")
end
true_support = findall(!iszero, θ_true)
@printf("\ntrue support     : %s\n", string(true_support))
@printf("recovered support: %s\n", string(res.support))
@printf("exact support match: %s\n", Set(res.support) == Set(true_support))
active_err = norm(res.theta[true_support] - θ_true[true_support]) / norm(θ_true[true_support])
@printf("relative rate error on true support: %.3e\n", active_err)
