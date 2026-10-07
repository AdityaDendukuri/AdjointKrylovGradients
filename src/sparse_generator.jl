# STLS for the affine family `A(θ) = A0 + Σ θ[j]dirs[j]`.

using Optim
using Zygote   # only for the semilinear pre_step path; the generator-only path
               # uses shared_gradient directly and never enters AD

"""
    SparseGeneratorResult

Result of `sparse_generator_stls`, including the fitted coefficients, support,
loss, Fisher scaling, and iteration history.
"""
struct SparseGeneratorResult
    theta::Vector{Float64}
    support::Vector{Int}
    loss::Float64
    fisher::Union{Nothing,Vector{Float64}}
    history::Dict{String,Vector}
    converged::Bool
    iterations::Int
end

assemble_generator(A0, dirs, θ) =
    isempty(dirs) ? A0 : A0 + sum(θ[j] * dirs[j] for j in eachindex(dirs))

"""
    sparse_generator_stls(A0, dirs, snapshots, dts, θ0; kwargs...)

Sequential thresholded least squares for a sparse affine generator library.

Alternates L-BFGS with hard thresholding and a final fixed-support polish.
Fisher preconditioning thresholds in `φ[j] = θ[j]sqrt(D[j])`. For semilinear
models, `pre_step` transforms the input before the exponential and
`n_generator` identifies the coefficients belonging to `dirs`.
"""
function sparse_generator_stls(
    A0, dirs, snapshots, dts, θ0;
    threshold_fraction = 0.05,
    inner_iter         = 10,
    outer_iter         = 20,
    polish_iter        = 100,
    m_krylov           = 30,
    n_quad             = 16,
    fisher_precond     = true,
    enforce_positive   = false,
    obs_op             = nothing,
    pre_step           = nothing,
    n_generator        = length(dirs),
    fisher_scale       = nothing,
    elimination_patience = 1,
    verbose            = true,
)
    M = length(θ0)
    if pre_step === nothing
        M == length(dirs) || throw(ArgumentError(
            "θ0 has $M entries, library has $(length(dirs)); pass pre_step for extra terms"))
    else
        n_generator <= M || throw(ArgumentError("n_generator exceeds length(θ0)"))
        fisher_precond && fisher_scale === nothing && throw(ArgumentError(
            "supply fisher_scale when pre_step is given: the Gauss-Newton diagonal " *
            "from shared_gradient_fisher only covers generator directions"))
    end

    gradfn = if pre_step === nothing
        θ -> shared_gradient(
            assemble_generator(A0, dirs, θ), dirs, snapshots, dts;
            obs_op=obs_op, m_krylov=m_krylov, n_quad=n_quad,
        )
    else
        layer = SemigroupLayer(A0, dirs; m_krylov=m_krylov, n_quad=n_quad)
        function semilinear_loss(θ)
            total = zero(eltype(θ))
            for t in eachindex(dts)
                u_in  = pre_step(snapshots[t], θ, dts[t])
                u_out = layer(u_in, θ[1:n_generator], dts[t])
                r = obs_op === nothing ? snapshots[t+1] - u_out :
                                         obs_op * snapshots[t+1] - obs_op * u_out
                total += 0.5 * sum(abs2, r)
            end
            total
        end
        θ -> begin
            l, back = Zygote.withgradient(semilinear_loss, θ)
            (back[1], l)
        end
    end

    # Fisher scale is taken once at the initial point, as in stls.jl: it defines
    # the coordinates the threshold lives in and must not drift between cycles.
    fisher = nothing
    scale  = ones(Float64, M)
    if fisher_precond
        if fisher_scale !== nothing
            fisher = collect(Float64, fisher_scale)
            scale  = sqrt.(max.(fisher, eps()))
        else
            _, D, _ = shared_gradient_fisher(
                assemble_generator(A0, dirs, θ0), dirs, snapshots, dts;
                obs_op=obs_op, m_krylov=m_krylov, n_quad=n_quad,
            )
            fisher = collect(Float64, D)
            scale  = sqrt.(max.(fisher, eps()))
        end
    end

    # Work in scaled coordinates φ = scale .* θ throughout.
    to_phi(θ)   = θ .* scale
    to_theta(φ) = φ ./ scale

    function loss_and_grad_phi(φ)
        θ = to_theta(φ)
        g, l = gradfn(θ)
        (l, g ./ scale)
    end

    φ = to_phi(collect(Float64, θ0))
    below_count = zeros(Int, M)
    history = Dict{String,Vector}(
        "iteration" => Int[], "loss" => Float64[],
        "nnz" => Int[], "threshold" => Float64[],
    )

    if verbose
        println("="^64)
        println("SPARSE GENERATOR STLS   (M = $M directions, n_quad = $n_quad)")
        println("="^64)
        @printf("%-6s %-14s %-6s %-12s\n", "iter", "loss", "nnz", "threshold")
        println("-"^64)
    end

    converged  = false
    iterations = 0

    for outer in 1:outer_iter
        iterations = outer
        active = findall(!iszero, φ)
        isempty(active) && break

        # Optimize on the current support only.
        f = φa -> begin
            full = zeros(Float64, M); full[active] = φa
            first(loss_and_grad_phi(full))
        end
        g! = (store, φa) -> begin
            full = zeros(Float64, M); full[active] = φa
            store .= last(loss_and_grad_phi(full))[active]
        end

        res = optimize(f, g!, φ[active], LBFGS(),
                       Optim.Options(iterations=inner_iter, show_trace=false))
        φ = zeros(Float64, M)
        φ[active] = Optim.minimizer(res)
        enforce_positive && (φ .= max.(φ, 0.0))
        loss = Optim.minimum(res)

        # Elimination hysteresis, matching optimize_stls: a coefficient must sit
        # below threshold for `elimination_patience` consecutive outer iterations
        # before it is removed, so a transiently underestimated true term gets a
        # chance to be pulled back first. patience = 1 eliminates immediately.
        threshold = threshold_fraction * maximum(abs, φ)
        below = abs.(φ) .< threshold
        below_count .= ifelse.(below, below_count .+ 1, 0)
        φ_new = copy(φ)
        φ_new[below_count .>= elimination_patience] .= 0.0
        nnz_now = count(!iszero, φ_new)

        push!(history["iteration"], outer)
        push!(history["loss"], loss)
        push!(history["nnz"], nnz_now)
        push!(history["threshold"], threshold)
        verbose && @printf("%-6d %-14.6e %-6d %-12.4e\n", outer, loss, nnz_now, threshold)

        support_stable = count(!iszero, φ) == nnz_now
        φ = φ_new
        if support_stable && outer > 1
            converged = true
            verbose && println("\nsupport stabilized at iteration $outer")
            break
        end
    end

    # Polish on the frozen support with no further thresholding.
    support = findall(!iszero, φ)
    final_loss = if isempty(support)
        first(loss_and_grad_phi(zeros(Float64, M)))
    else
        f = φa -> begin
            full = zeros(Float64, M); full[support] = φa
            first(loss_and_grad_phi(full))
        end
        g! = (store, φa) -> begin
            full = zeros(Float64, M); full[support] = φa
            store .= last(loss_and_grad_phi(full))[support]
        end
        res = optimize(f, g!, φ[support], LBFGS(),
                       Optim.Options(iterations=polish_iter, show_trace=false))
        φ = zeros(Float64, M)
        φ[support] = Optim.minimizer(res)
        enforce_positive && (φ .= max.(φ, 0.0))
        Optim.minimum(res)
    end

    θ_final = to_theta(φ)
    if verbose
        println("-"^64)
        @printf("final loss %.6e   support %d / %d\n", final_loss, length(support), M)
        println("="^64)
    end

    SparseGeneratorResult(
        θ_final, support, final_loss, fisher, history, converged, iterations,
    )
end
