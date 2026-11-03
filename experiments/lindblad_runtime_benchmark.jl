"""
Direct runtime benchmark for the structured Lindblad gradient.

Measures shared-gradient wall time against direct finite differences on the
same Lindblad loss, for both full-state and partial-observation settings.
"""

using Printf, Statistics

include("lindblad_structured.jl")

const RUNTIME_RESULT_PATH = joinpath(@__DIR__, "../paper/figures/lindblad_runtime_benchmark.csv")

function sg_loss_only(A, dirs, states, dts, obs_op, from_initial, mk)
    if !from_initial
        _, loss = shared_gradient(A, [dirs[1]], states, dts; obs_op=obs_op, m_krylov=mk)
        return loss
    end

    loss = 0.0
    p0 = states[1]
    for k in eachindex(dts)
        _, l_k = shared_gradient(A, [dirs[1]], [p0, states[k + 1]], [dts[k]];
                                 obs_op=obs_op, m_krylov=mk)
        loss += l_k
    end
    return loss
end

function fd_gradient_lindblad(A, dirs, states, dts, obs_op, from_initial, mk; eps=1e-5)
    M = length(dirs)
    loss0 = sg_loss_only(A, dirs, states, dts, obs_op, from_initial, mk)
    grad = zeros(M)
    for j in 1:M
        loss_p = sg_loss_only(A + eps * dirs[j], dirs, states, dts, obs_op, from_initial, mk)
        loss_m = sg_loss_only(A - eps * dirs[j], dirs, states, dts, obs_op, from_initial, mk)
        grad[j] = (loss_p - loss_m) / (2eps)
    end
    return grad, loss0
end

function benchmark_theta(c, n)
    θ = true_params(c, n)
    scales = 1 .+ 0.2 .* sin.(collect(1:length(θ)))
    return max.(θ .* scales .+ 0.03, 1e-4)
end

function write_runtime_results(path, rows)
    open(path, "w") do io
        println(io, "n,obs_mode,observation_dim,M,shared_ms,fd_ms,speedup,grad_rel_err")
        for row in rows
            @printf(io, "%d,%s,%d,%d,%.10f,%.10f,%.10f,%.10f\n",
                    row.n, row.obs_mode, row.observation_dim, row.M,
                    row.shared_ms, row.fd_ms, row.speedup, row.grad_rel_err)
        end
    end
end

function run_lindblad_runtime_benchmark(; ns=[2, 3, 4], obs_modes=[:full_state, :pauli_local],
                                        n_trials=7, context=0.5)
    rows = NamedTuple[]
    println("Structured Lindblad runtime benchmark")

    for obs_mode in obs_modes
        println("\nObservation mode: $(obs_mode)")
        for n in ns
            A_const, dirs = build_ising_library(n)
            θ = benchmark_theta(context, n)
            A = A_const + sum(θ[j] * dirs[j] for j in eachindex(θ))
            obs_model = build_observation_model(n; obs_mode=obs_mode)
            full_states, dts_seq = propagate_lindblad(context, n, A_const, dirs)
            states = project_lindblad_states(full_states, obs_model)
            loss_dts = obs_model.from_initial ? collect(DTS_LINDBLAD) : dts_seq
            mk = min(40, 4^n - 1)

            if !obs_model.from_initial
                shared_gradient(A, dirs, states, dts_seq; obs_op=obs_model.obs_op, m_krylov=mk)
            else
                _sg_grad_loss_lindblad(θ, states, loss_dts, A_const, dirs, mk,
                                       obs_model.obs_op, obs_model.from_initial)
            end
            fd_gradient_lindblad(A, dirs, states, loss_dts, obs_model.obs_op, obs_model.from_initial, mk)

            t_shared = Float64[]
            g_shared = zeros(Float64, length(dirs))
            for _ in 1:n_trials
                t0 = time_ns()
                if !obs_model.from_initial
                    g_shared, _ = shared_gradient(A, dirs, states, dts_seq;
                                                 obs_op=obs_model.obs_op, m_krylov=mk)
                else
                    g_shared, _ = _sg_grad_loss_lindblad(θ, states, loss_dts, A_const, dirs, mk,
                                                         obs_model.obs_op, obs_model.from_initial)
                end
                push!(t_shared, (time_ns() - t0) * 1e-6)
            end

            t_fd = Float64[]
            g_fd = zeros(Float64, length(dirs))
            for _ in 1:n_trials
                t0 = time_ns()
                g_fd, _ = fd_gradient_lindblad(A, dirs, states, loss_dts,
                                               obs_model.obs_op, obs_model.from_initial, mk)
                push!(t_fd, (time_ns() - t0) * 1e-6)
            end

            shared_ms = median(t_shared)
            fd_ms = median(t_fd)
            rel_err = norm(g_fd - g_shared) / max(norm(g_shared), norm(g_fd), 1e-12)

            row = (
                n=n,
                obs_mode=string(obs_mode),
                observation_dim=obs_model.obs_dim,
                M=length(dirs),
                shared_ms=shared_ms,
                fd_ms=fd_ms,
                speedup=fd_ms / shared_ms,
                grad_rel_err=rel_err,
            )
            push!(rows, row)
            @printf("  n=%d  obs_dim=%d  shared=%.2f ms  fd=%.2f ms  speedup=%.1fx  grad err=%.2e\n",
                    row.n, row.observation_dim, row.shared_ms, row.fd_ms, row.speedup, row.grad_rel_err)
        end
    end

    write_runtime_results(RUNTIME_RESULT_PATH, rows)
    println("\nSaved Lindblad runtime benchmark → $(RUNTIME_RESULT_PATH)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lindblad_runtime_benchmark()
end
