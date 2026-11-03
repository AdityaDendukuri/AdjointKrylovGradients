using AdjointKrylovGradients, LinearAlgebra, SparseArrays, Random, Printf

# Semilinear test mirroring the KS structure: exponential layer on linear terms,
# explicit nonlinear flux pre-step, overcomplete library on both blocks.
Random.seed!(3)
n, L = 48, 22.0
dx = L / n
D1 = spdiagm(-1 => fill(-1/(2dx), n-1), 1 => fill(1/(2dx), n-1)); D1[1,n] = -1/(2dx); D1[n,1] = 1/(2dx)
D2 = spdiagm(-1 => fill(1/dx^2, n-1), 0 => fill(-2/dx^2, n), 1 => fill(1/dx^2, n-1)); D2[1,n]=1/dx^2; D2[n,1]=1/dx^2
D4 = sparse(D2*D2)

lin_dirs = [sparse(-D1), sparse(-D2), sparse(D1*D2), sparse(-D4)]   # 4 candidates, true: 2 and 4
n_gen = length(lin_dirs)
flux(u, p) = -(D1 * (u .^ p)) ./ p                                   # nonlinear candidates p=2..4

θ_true = zeros(n_gen + 3)
θ_true[2] = 1.0     # -D2
θ_true[4] = 1.0     # -D4
θ_true[n_gen + 1] = 1.0   # flux p=2  (the true -u u_x term)

dt = 0.0125
pre_step(u, θ, dt) = u .+ dt .* sum(θ[n_gen+i] .* flux(u, i+1) for i in 1:3)

x = range(0, L; length=n+1)[1:n]
u0 = cos.(2π .* x ./ L) .+ 0.1 .* sin.(4π .* x ./ L)
A_true = assemble_generator(spzeros(n, n), lin_dirs, θ_true[1:n_gen])
E = exp(Matrix(A_true) * dt)
snaps = [u0]
for _ in 1:12
    push!(snaps, E * pre_step(snaps[end], θ_true, dt))
end
dts = fill(dt, length(snaps)-1)

# Data-based scale proxy: RMS action of each candidate on the observed states.
scale = Float64[]
for j in 1:n_gen
    push!(scale, sum(sum(abs2, lin_dirs[j] * s) for s in snaps[1:end-1]))
end
for i in 1:3
    push!(scale, sum(sum(abs2, flux(s, i+1)) for s in snaps[1:end-1]))
end

θ0 = fill(0.4, n_gen + 3)
res = sparse_generator_stls(spzeros(n, n), lin_dirs, snaps, dts, θ0;
        pre_step=pre_step, n_generator=n_gen, fisher_scale=scale,
        threshold_fraction=0.15, outer_iter=12, polish_iter=120,
        m_krylov=30, n_quad=16, enforce_positive=false, verbose=false)

labels = ["-D1","-D2","D1D2","-D4","flux2","flux3","flux4"]
println(" term    true   recovered")
for j in eachindex(θ_true)
    @printf("  %-6s %5.2f   %9.5f%s\n", labels[j], θ_true[j], res.theta[j],
            j in res.support ? "  <-" : "")
end
ts = findall(!iszero, θ_true)
@printf("\ntrue support %s / recovered %s / exact %s\n", string(ts), string(res.support), Set(res.support)==Set(ts))
@printf("rate error on true support: %.3e\n", norm(res.theta[ts]-θ_true[ts])/norm(θ_true[ts]))
