# =============================================================================
# benchmark_k_sweep.jl
#
# Compares three gradient methods as the candidate parameter count MK grows:
#
#   1. Adjoint-Krylov (ours): shared Arnoldi bases, O(T·m_K·nnz(A)) + O(T·MK·n_q·n)
#   2. Per-direction Fréchet: one frechet_action per (direction, time-step),
#      O(T·MK·m_K·nnz(A)).  This is what any autodiff library with a custom
#      matrix-exponential JVP rule would compute (e.g. ForwardDiff with
#      ExponentialUtilities once dual-number support is added, or Zygote with a
#      custom rrule).  Standard Julia autodiff (ForwardDiff.jl, Zygote.jl)
#      cannot differentiate through expv out-of-the-box because LinearAlgebra
#      has no exp!/eigen! dispatch for dual-number matrices; the per-direction
#      Fréchet loop is the equivalent hand-rolled version.
#   3. Finite differences (for reference): MK+1 forward solves at ε-perturbation.
#      Cheapest per direction but numerically noisy and not used in practice.
#
# Output: paper/figures/adjoint_krylov_results.pdf
# =============================================================================

using AdjointKrylovGradients
using LinearAlgebra, Printf, Random, Statistics, JLD2
using SparseArrays
using ExponentialUtilities: expv
using CairoMakie

Random.seed!(42)

# ---------------------------------------------------------------------------
# Load Brusselator state space and empirical distributions
# ---------------------------------------------------------------------------
jld  = load(joinpath(@__DIR__, "..", "results", "brusselator_results.jld2"))
X    = jld["X"]
P    = jld["P_empirical"]
DT   = 1.0
n    = length(X)
T_steps = length(P) - 1

K_cub = cubic_feature_dimension(2)   # 10
R4    = [(1,0), (1,-1), (-1,1), (-1,0)]
@printf("Loaded Brusselator: n=%d states, T=%d steps, K_cub=%d\n", n, T_steps, K_cub)

# ---------------------------------------------------------------------------
# Build derivatives for the true 4-reaction library once
# ---------------------------------------------------------------------------
θ4  = Dict(ζ => fill(0.1, K_cub) for ζ in R4)
dA4 = build_derivatives(X, R4; feature_map=cubic_features, K=K_cub)
A_base = build_generator(X, R4, θ4; feature_map=cubic_features)

R4_sorted = sort(R4)
N_mats_base = [dA4[ζ][f] for ζ in R4_sorted for f in 1:K_cub]
MK_base = length(N_mats_base)   # = 40

# ---------------------------------------------------------------------------
# Method 1: Adjoint-Krylov (shared-basis inner-product loop)
# ---------------------------------------------------------------------------
using AdjointKrylovGradients: arnoldi, GL_NODES_5, GL_WEIGHTS_5

function adjoint_inner_loop!(g::Vector{Float64},
                              N_mats::Vector{<:AbstractSparseMatrix},
                              P, dt, n, T_steps)
    m_K    = min(25, n-1)
    At     = A_base'
    MK     = length(N_mats)
    n_q    = 5
    s_nodes, w_nodes = GL_NODES_5, GL_WEIGHTS_5

    F_k  = zeros(n, n_q)
    Λ_k  = zeros(n, n_q)
    buf  = zeros(n)
    fill!(g, 0.0)

    for k in 1:T_steps
        dt_k = dt isa AbstractVector ? dt[k] : dt

        ks_f = try arnoldi(A_base, P[k]; m=m_K, tol=1e-8) catch; nothing end
        ks_f === nothing && continue

        p_pred = let
            m  = ks_f.m; Vm = view(ks_f.V,:,1:m); Hm = Matrix(view(ks_f.H,1:m,1:m))
            e1 = zeros(m); e1[1] = 1.0
            y  = exp(dt_k .* Hm) * e1
            out = zeros(n); mul!(out, Vm, y); out .*= ks_f.beta; out
        end
        r_k  = P[k+1] .- p_pred
        norm(r_k) > 1e-15 || continue

        ks_a = try arnoldi(At, r_k; m=m_K, tol=1e-8) catch; nothing end
        ks_a === nothing && continue

        for (basis, ts, F) in ((ks_f, s_nodes .* dt_k, F_k),
                                (ks_a, (1 .- s_nodes) .* dt_k, Λ_k))
            m  = basis.m; Vm = view(basis.V,:,1:m); Hm = Matrix(view(basis.H,1:m,1:m))
            e1 = zeros(m); e1[1] = 1.0
            VE, d = let F2 = eigen(Hm); F2.vectors, F2.values end
            c = try VE \ e1 catch; pinv(VE)*e1 end
            Y = Matrix{ComplexF64}(undef, m, n_q)
            for q in 1:n_q; Y[:,q] = VE*(exp.(ts[q].*d).*c); end
            mul!(F, Vm, real.(Y)); F .*= basis.beta
        end

        wdt = dt_k .* w_nodes
        @inbounds for idx in 1:MK
            val = 0.0
            for q in 1:n_q
                mul!(buf, N_mats[idx], view(F_k,:,q))
                val -= wdt[q] * dot(view(Λ_k,:,q), buf)
            end
            g[idx] += val
        end
    end
    return g
end

# ---------------------------------------------------------------------------
# Method 2: Per-direction Fréchet (forward-mode AD equivalent)
#
# For each parameter direction i, computes frechet_action(A*dt, N_mats[i]*dt, p_t)
# and accumulates into the gradient via dot product with the residual.
# This is the standard approach used by any autodiff framework that supports
# matrix exponentials via a custom JVP rule.  Cost: O(T·MK·m_K·nnz(A)).
# ---------------------------------------------------------------------------
function frechet_naive_loop!(g::Vector{Float64},
                              N_mats::Vector{<:AbstractSparseMatrix},
                              P, dt, n, T_steps)
    MK = length(N_mats)
    fill!(g, 0.0)
    for k in 1:T_steps
        dt_k  = dt isa AbstractVector ? dt[k] : dt
        p_pred = expv(dt_k, A_base, P[k]; m=25, tol=1e-8)
        r_k   = P[k+1] .- p_pred
        norm(r_k) > 1e-15 || continue
        for idx in 1:MK
            fa = AdjointKrylovGradients.frechet_action(A_base * dt_k, N_mats[idx] * dt_k, P[k])
            g[idx] -= dot(r_k, fa)
        end
    end
    return g
end

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
print("Compiling (warmup)... "); flush(stdout)
g_w = zeros(MK_base)
adjoint_inner_loop!(g_w, N_mats_base, P, DT, n, T_steps)
adjoint_inner_loop!(g_w, N_mats_base, P, DT, n, T_steps)
frechet_naive_loop!(g_w, N_mats_base, P, DT, n, T_steps)
frechet_naive_loop!(g_w, N_mats_base, P, DT, n, T_steps)
θ4_f = flatten_params(θ4)
objective(P, R4, θ4, X, DT; feature_map=cubic_features)
objective(P, R4, θ4, X, DT; feature_map=cubic_features)
println("done")

# ---------------------------------------------------------------------------
# MK sweep
# ---------------------------------------------------------------------------
MK_targets = [10, 20, 40, 80, 120, 200, 320, 480, 640, 800]
n_repeat    = 5

t_adj_vec     = Float64[]
t_frechet_vec = Float64[]
t_fd_vec      = Float64[]

println("\nMK sweep (adjoint vs per-direction Fréchet vs FD wall time):")
println("-"^72)
@printf("%-6s  %-10s  %-12s  %-10s\n", "MK", "t_adj(s)", "t_frechet(s)", "t_FD(s)")
println("-"^72)

t_1obj = median([@elapsed objective(P, R4, θ4, X, DT; feature_map=cubic_features)
                  for _ in 1:5])
@printf("One objective eval: %.4f s\n\n", t_1obj)

for MK in MK_targets
    reps     = ceil(Int, MK / MK_base)
    N_mats_k = repeat(N_mats_base, reps)[1:MK]
    g_k      = zeros(MK)

    # Adjoint timing
    times_adj = [(@elapsed adjoint_inner_loop!(g_k, N_mats_k, P, DT, n, T_steps))
                 for _ in 1:n_repeat]
    t_adj = median(times_adj)

    # Per-direction Fréchet timing (only up to MK=200 — scales as O(MK))
    if MK <= 200
        times_fr = [(@elapsed frechet_naive_loop!(g_k, N_mats_k, P, DT, n, T_steps))
                    for _ in 1:n_repeat]
        t_fr = median(times_fr)
    else
        # Skip unmeasured points linearly from the MK=200 measurement
        t_fr = NaN
    end

    # FD timing
    ε = 1e-6
    t_fd = @elapsed begin
        f0 = objective(P, R4, θ4, X, DT; feature_map=cubic_features)
        for i in 1:MK
            θ_p = copy(θ4_f); θ_p[i % MK_base + 1] += ε
            objective(P, R4, unflatten_params(θ_p[1:MK_base], R4, K_cub), X, DT;
                      feature_map=cubic_features)
        end
    end

    push!(t_adj_vec,     t_adj)
    push!(t_frechet_vec, t_fr)
    push!(t_fd_vec,      t_fd)

    @printf("%-6d  %-10.3f  %-12s  %-10.3f\n", MK, t_adj,
            isnan(t_fr) ? "(not run)" : @sprintf("%.3f", t_fr), t_fd)
end
println("-"^72)


open(joinpath(@__DIR__, "..", "benchmark_results", "sisc_library_sweep_20260906.csv"),"w") do io
 println(io,"MK,adjoint_seconds,frechet_seconds,fd_seconds")
 for i in eachindex(MK_targets)
  println(io,"$(MK_targets[i]),$(t_adj_vec[i]),$(t_frechet_vec[i]),$(t_fd_vec[i])")
 end
end
set_theme!(theme_latexfonts())
f=Figure(size=(800,360));ax=Axis(f[1,1],xlabel="Candidate directions",ylabel="Gradient time (s)",yscale=log10,xscale=log10,topspinevisible=false,rightspinevisible=false,title="Brusselator library sweep (local rerun)")
scatterlines!(ax,MK_targets,t_adj_vec,color=:black,marker=:circle,linestyle=:solid,label="Shared quadrature")
mask=.!isnan.(t_frechet_vec)
scatterlines!(ax,MK_targets[mask],t_frechet_vec[mask],color=:black,marker=:rect,linestyle=:dash,label="Per-direction Fréchet (measured)")
scatterlines!(ax,MK_targets,t_fd_vec,color=:black,marker=:diamond,linestyle=:dot,label="Forward differences")
axislegend(ax,position=:lt,labelsize=12)
save(joinpath(@__DIR__, "..", "paper", "figures", "library_sweep_bw.pdf"),f)
println("Completed measured-only library sweep")
