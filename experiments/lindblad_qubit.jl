using LinearAlgebra, Printf, Optim

include("../src/shared_krylov.jl")
include("../src/lindblad.jl")

function run_lindblad_qubit(; N=2)
    d = 2^N

    # --- library construction ---
    # Hamiltonian basis: nearest-neighbor XX + single-site X,Z
    H_basis = AbstractMatrix{ComplexF64}[]
    for i in 1:N-1
        push!(H_basis, nsite_op(σx, i, N) * nsite_op(σx, i+1, N))  # XXᵢ,ᵢ₊₁
    end
    for i in 1:N; push!(H_basis, nsite_op(σx, i, N)); end
    for i in 1:N; push!(H_basis, nsite_op(σz, i, N)); end

    # Lindblad basis: X and Z dephasing/decay on each site
    L_basis = AbstractMatrix{ComplexF64}[]
    for i in 1:N; push!(L_basis, nsite_op(σx, i, N)); end
    for i in 1:N; push!(L_basis, nsite_op(σz, i, N)); end

    A₀, dirs = lindblad_library(H_basis, L_basis)
    M = length(dirs)

    n_H = length(H_basis)  # first n_H dirs are Hamiltonian terms

    # --- true parameters ---
    # H_true = -1.0 * Σ XXᵢ,ᵢ₊₁  -  0.5 * Σ Zᵢ
    # γ_true = 0.1  on  D[Zᵢ] (dephasing),  0 elsewhere
    θ_true = zeros(M)
    for i in 1:N-1; θ_true[i] = -1.0; end                   # XX couplings
    for i in 1:N;   θ_true[(N-1)+N+i] = -0.5; end            # Z fields
    for i in 1:N;   θ_true[n_H + N + i] = 0.1; end           # Z dephasing

    L_true = A₀ + sum(θ_true[j] * dirs[j] for j in eachindex(θ_true))

    # --- synthetic data: evolve from |+⟩^⊗N ---
    plus = ComplexF64[1; 1] / sqrt(2)
    ψ₀   = foldl(kron, fill(plus, N))
    ρ₀   = ψ₀ * ψ₀'
    ρ₀_vec = vec(ρ₀)

    T_obs = 12; dt = 0.4
    dts   = fill(dt, T_obs)
    ρ_data = Vector{Vector{ComplexF64}}(undef, T_obs + 1)
    ρ_data[1] = ρ₀_vec
    Mprop = exp(Matrix(L_true) * dt)
    for t in 1:T_obs
        ρ_data[t+1] = Mprop * ρ_data[t]
    end

    # --- gradient / loss wrappers with cache ---
    last_θ = fill(NaN, M)
    last_g = zeros(Float64, M)
    last_L = Ref(0.0)
    function compute_gl(θ_vec)
        if θ_vec != last_θ
            Lmat = A₀ + sum(θ_vec[j] * dirs[j] for j in eachindex(θ_vec))
            last_g, last_L[] = shared_gradient(Lmat, dirs, ρ_data, dts; m_krylov=min(30, d^2-1))
            copyto!(last_θ, θ_vec)
        end
        last_g, last_L[]
    end
    f_obj(θ_vec)     = compute_gl(θ_vec)[2]
    g_obj!(G, θ_vec) = (G .= compute_gl(θ_vec)[1]; nothing)

    # --- STLS recovery ---
    τ = 0.05; n_inner = 30
    θ = zeros(M)

    @printf("%-6s  %-12s  %-6s\n", "outer", "loss", "nnz")
    for outer in 1:25
        result = optimize(f_obj, g_obj!, θ, LBFGS(),
                          Optim.Options(iterations=n_inner, show_trace=false))
        θ = Optim.minimizer(result)
        θ[abs.(θ) .< τ * maximum(abs.(θ))] .= 0.0
        @printf("%-6d  %-12.4e  %-6d\n", outer, Optim.minimum(result), count(θ .!= 0))
        count(θ .!= 0) == count(θ_true .!= 0) && break
    end

    # --- polish ---
    support = findall(θ .!= 0)
    if !isempty(support)
        function f_pol(θ_s)
            θf = zeros(M); θf[support] = θ_s; compute_gl(θf)[2]
        end
        function g_pol!(G, θ_s)
            θf = zeros(M); θf[support] = θ_s; G .= compute_gl(θf)[1][support]
        end
        result_pol = optimize(f_pol, g_pol!, θ[support],
                              LBFGS(), Optim.Options(iterations=200, show_trace=false))
        θ[support] = Optim.minimizer(result_pol)
    end

    err = norm(θ - θ_true) / norm(θ_true)
    println("\nTrue θ:      ", round.(θ_true; digits=2))
    println("Recovered θ: ", round.(θ; digits=3))
    @printf("Relative ℓ₂ error: %.2f%%\n", 100err)
    return θ
end

run_lindblad_qubit(N=2)
