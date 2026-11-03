using LinearAlgebra, Printf, Random, Statistics
using Flux, Zygote
using ExponentialUtilities: expv

include("../src/shared_krylov.jl")
include("../src/lindblad.jl")

const H_FIELD_LINDBLAD = 0.3
const DTS_LINDBLAD = [0.3, 1.0, 3.0]
const TRAIN_CTXS_LINDBLAD = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
const TEST_CTXS_LINDBLAD  = [0.10, 0.30, 0.50, 0.70, 0.90]

J_of(c, i) = 0.5 * (1 + 0.4 * sin(i * π * c))
γ_of(c, i) = 0.1 * (1 + 0.5 * cos(i * π * c)^2)

true_params(c, n) = vcat([J_of(c, i) for i in 1:n-1], [γ_of(c, i) for i in 1:n])

param_labels(n) = vcat(["J$(i)" for i in 1:n-1], ["gamma$(i)" for i in 1:n])

function build_ising_library(n::Int; sparse_operators=false)
    site_op = sparse_operators ? nsite_op_sparse : nsite_op
    h_superop = sparse_operators ? hamiltonian_superop_sparse : hamiltonian_superop
    d_superop = sparse_operators ? lindblad_superop_sparse : lindblad_superop

    H_field = sum(site_op(σz, i, n) for i in 1:n)
    A_const = H_FIELD_LINDBLAD * h_superop(H_field)

    dirs = AbstractMatrix{ComplexF64}[]
    for i in 1:n-1
        H_i = site_op(σx, i, n) * site_op(σx, i + 1, n) +
              site_op(σy, i, n) * site_op(σy, i + 1, n)
        push!(dirs, h_superop(H_i))
    end
    for i in 1:n
        push!(dirs, d_superop(site_op(σz, i, n)))
    end

    return A_const, dirs
end

function initial_density_vec(n::Int)
    plus = ComplexF64[1, 1] / sqrt(2)
    ψ0 = foldl(kron, [plus for _ in 1:n])
    vec(ψ0 * ψ0')
end

interval_dts(abs_times::AbstractVector) = vcat(abs_times[1], diff(abs_times))

function observation_model_from_ops(n::Int, obs_ops, labels; mode, name, from_initial=true)
    d = 2^n
    C = zeros(ComplexF64, length(obs_ops), d^2)
    for (k, op) in enumerate(obs_ops)
        C[k, :] .= vec(transpose(op))
    end
    C_pinv = pinv(C)

    return (
        mode=mode,
        name=name,
        obs_op=C,
        lift=(y -> C_pinv * y),
        obs_dim=size(C, 1),
        labels=labels,
        from_initial=from_initial,
    )
end

function pauli_observation_model(n::Int; obs_mode=:pauli_local)
    d = 2^n
    obs_ops = Matrix{ComplexF64}[Matrix{ComplexF64}(I, d, d)]
    labels = ["I"]

    if obs_mode == :z_local
        for i in 1:n
            push!(obs_ops, nsite_op(σz, i, n))
            push!(labels, "Z$(i)")
        end
        return observation_model_from_ops(n, obs_ops, labels;
                                          mode=:z_local,
                                          name="identity plus local Z expectations")
    elseif obs_mode == :pauli_single
        for i in 1:n
            push!(obs_ops, nsite_op(σx, i, n)); push!(labels, "X$(i)")
            push!(obs_ops, nsite_op(σy, i, n)); push!(labels, "Y$(i)")
            push!(obs_ops, nsite_op(σz, i, n)); push!(labels, "Z$(i)")
        end
        return observation_model_from_ops(n, obs_ops, labels;
                                          mode=:pauli_single,
                                          name="identity plus all single-site Pauli expectations")
    elseif obs_mode == :pauli_local
        for i in 1:n
            push!(obs_ops, nsite_op(σx, i, n)); push!(labels, "X$(i)")
            push!(obs_ops, nsite_op(σy, i, n)); push!(labels, "Y$(i)")
            push!(obs_ops, nsite_op(σz, i, n)); push!(labels, "Z$(i)")
        end
        for i in 1:n-1
            Xi = nsite_op(σx, i, n); Xj = nsite_op(σx, i + 1, n)
            Yi = nsite_op(σy, i, n); Yj = nsite_op(σy, i + 1, n)
            Zi = nsite_op(σz, i, n); Zj = nsite_op(σz, i + 1, n)
            push!(obs_ops, Xi * Xj); push!(labels, "X$(i)X$(i+1)")
            push!(obs_ops, Yi * Yj); push!(labels, "Y$(i)Y$(i+1)")
            push!(obs_ops, Zi * Zj); push!(labels, "Z$(i)Z$(i+1)")
        end
        return observation_model_from_ops(n, obs_ops, labels;
                                          mode=:pauli_local,
                                          name="local-and-edge Pauli expectations")
    else
        error("Unsupported Pauli obs_mode=$(obs_mode)")
    end
end

function build_observation_model(n::Int; obs_mode=:full_state)
    if obs_mode == :full_state
        return (
            mode=:full_state,
            name="full density matrices",
            obs_op=nothing,
            lift=identity,
            obs_dim=4^n,
            labels=String[],
            from_initial=false,
        )
    elseif obs_mode in (:z_local, :pauli_single, :pauli_local)
        return pauli_observation_model(n; obs_mode=obs_mode)
    else
        error("Unsupported obs_mode=$(obs_mode)")
    end
end

function add_observation_noise(y, obs_model; shots=0, noise_std=0.0, rng=Random.default_rng())
    (shots <= 0 && noise_std <= 0) && return copy(y)

    y_noisy = copy(y)
    for i in eachindex(y_noisy)
        if obs_model.labels[i] == "I"
            y_noisy[i] = one(eltype(y_noisy))
            continue
        end
        μ = clamp(real(y_noisy[i]), -1.0, 1.0)
        σ2 = 0.0
        if shots > 0
            σ2 += max(1 - μ^2, 0.0) / shots
        end
        if noise_std > 0
            σ2 += noise_std^2
        end
        μ̃ = μ + sqrt(σ2) * randn(rng)
        y_noisy[i] = ComplexF64(clamp(μ̃, -1.0, 1.0))
    end
    return y_noisy
end

function project_lindblad_states(states, obs_model; shots=0, noise_std=0.0, rng=Random.default_rng())
    obs_model.obs_op === nothing && return [copy(state) for state in states]
    projected = Vector{Vector{ComplexF64}}(undef, length(states))
    projected[1] = copy(states[1])
    for k in 2:length(states)
        y = obs_model.obs_op * states[k]
        y = add_observation_noise(y, obs_model; shots=shots, noise_std=noise_std, rng=rng)
        projected[k] = obs_model.lift(y)
    end
    return projected
end

function propagate_lindblad(c, n, A_const, dirs; abs_times=DTS_LINDBLAD)
    θ = true_params(c, n)
    L = A_const + sum(θ[j] * dirs[j] for j in eachindex(θ))
    d2 = 4^n
    mk = min(40, d2 - 1)
    dts_seq = interval_dts(abs_times)
    states = Vector{Vector{ComplexF64}}(undef, length(dts_seq) + 1)
    states[1] = initial_density_vec(n)
    for k in eachindex(dts_seq)
        states[k + 1] = expv(dts_seq[k], L, states[k]; m=mk)
    end
    return states, dts_seq
end

# Evaluation helper only — not used in the training graph.
function _sg_loss_lindblad(θ, states, dts, A_const, dirs, mk, obs_op, from_initial)
    L = A_const + sum(θ[j] * dirs[j] for j in eachindex(θ))
    if !from_initial
        _, loss = shared_gradient(L, dirs, states, dts; obs_op=obs_op, m_krylov=mk)
        return loss
    end
    loss = 0.0
    p0 = states[1]
    for k in eachindex(dts)
        _, l_k = shared_gradient(L, dirs, [p0, states[k+1]], [dts[k]];
                                 obs_op=obs_op, m_krylov=mk)
        loss += l_k
    end
    loss
end

param_floor(n) = vcat(fill(0.1, n - 1), fill(0.02, n))

function predict_params(net, c, n)
    raw = vec(net([c]))
    softplus.(raw) .+ param_floor(n)
end

function train_lindblad_model(n::Int; n_epochs=600, hidden=64, seed=42, verbose=true,
                              train_ctxs=TRAIN_CTXS_LINDBLAD, test_ctxs=TEST_CTXS_LINDBLAD,
                              abs_times=DTS_LINDBLAD, obs_mode=:full_state,
                              shots=0, noise_std=0.0, sparse_operators=false)
    Random.seed!(seed)

    d = 2^n
    d2 = d^2
    M = 2n - 1
    A_const, dirs = build_ising_library(n; sparse_operators=sparse_operators)
    obs_model = build_observation_model(n; obs_mode=obs_mode)
    if obs_mode == :full_state && (shots > 0 || noise_std > 0)
        error("Observation noise is only supported for partial-observation Lindblad modes.")
    end
    train_full_states = [propagate_lindblad(c, n, A_const, dirs; abs_times=abs_times)[1] for c in train_ctxs]
    test_full_states  = [propagate_lindblad(c, n, A_const, dirs; abs_times=abs_times)[1] for c in test_ctxs]
    rng = MersenneTwister(seed)
    train_data = [project_lindblad_states(states, obs_model; shots=shots, noise_std=noise_std, rng=rng)
                  for states in train_full_states]
    test_data  = [project_lindblad_states(states, obs_model; shots=shots, noise_std=noise_std, rng=rng)
                  for states in test_full_states]
    dts_seq = interval_dts(abs_times)
    loss_dts = obs_model.from_initial ? collect(abs_times) : dts_seq
    mk = min(40, d2 - 1)

    net = Chain(Dense(1 => hidden, tanh), Dense(hidden => hidden, tanh), Dense(hidden => M)) |> f64
    opt = Flux.setup(Flux.Adam(3e-3), net)

    # SemigroupLayer: A_const is the fixed part (transverse field),
    # dirs are the M = 2n-1 coupling + dissipator directions.
    # Works for complex generators and states without modification.
    sl = SemigroupLayer(A_const, dirs; m_krylov=mk)

    obs_op      = obs_model.obs_op
    from_initial = obs_model.from_initial

    # One semigroup step through the layer; obs_op projected outside so
    # Zygote chains through it automatically.
    function step_loss(sl, ρ_t, ρ_next, θ, dt)
        ρ_pred = sl(ρ_t, θ, dt)
        if obs_op === nothing
            return 0.5 * sum(abs2, ρ_next - ρ_pred)
        else
            return 0.5 * sum(abs2, obs_op * ρ_next - obs_op * ρ_pred)
        end
    end

    function total_loss(model, data, ctxs)
        total = 0.0
        for i in eachindex(ctxs)
            θ      = predict_params(model, ctxs[i], n)
            states = data[i]
            if from_initial
                ρ0 = states[1]
                for k in eachindex(loss_dts)
                    total += step_loss(sl, ρ0, states[k+1], θ, loss_dts[k])
                end
            else
                for t in eachindex(loss_dts)
                    total += step_loss(sl, states[t], states[t+1], θ, loss_dts[t])
                end
            end
        end
        total
    end

    history_train = Float64[]
    history_test = Float64[]

    verbose && println("\n══ Structured Lindblad  n=$n  d²=$d2  M=$M  obs=$(obs_model.mode)  shots=$shots  noise_std=$noise_std ══")
    for epoch in 1:n_epochs
        _, grads = Zygote.withgradient(net) do model
            total_loss(model, train_data, train_ctxs)
        end
        Flux.update!(opt, net, grads[1])

        if verbose && (epoch == 1 || epoch % 200 == 0)
            tr = total_loss(net, train_data, train_ctxs)
            te = total_loss(net, test_data, test_ctxs)
            push!(history_train, tr)
            push!(history_test, te)
            @printf("  epoch %3d  train=%.6f  test=%.6f\n", epoch, tr, te)
        end
    end

    return (
        n=n,
        d=d,
        d2=d2,
        M=M,
        A_const=A_const,
        dirs=dirs,
        net=net,
        mk=mk,
        obs_model=obs_model,
        obs_mode=obs_model.mode,
        shots=shots,
        noise_std=noise_std,
        sparse_operators=sparse_operators,
        abs_times=collect(abs_times),
        dts_seq=dts_seq,
        loss_dts=loss_dts,
        train_ctxs=collect(train_ctxs),
        test_ctxs=collect(test_ctxs),
        train_full_states=train_full_states,
        test_full_states=test_full_states,
        train_data=train_data,
        test_data=test_data,
        history_train=history_train,
        history_test=history_test,
    )
end

function evaluate_lindblad_model(model)
    labels = param_labels(model.n)
    rows = NamedTuple[]
    param_errs = Float64[]
    pred_losses = Float64[]

    for (i, c) in enumerate(model.test_ctxs)
        θ_true = true_params(c, model.n)
        θ_pred = predict_params(model.net, c, model.n)
        push!(pred_losses,
              _sg_loss_lindblad(θ_pred, model.test_data[i], model.loss_dts,
                                model.A_const, model.dirs, model.mk,
                                model.obs_model.obs_op, model.obs_model.from_initial))
        for j in eachindex(θ_true)
            rel_err = abs(θ_pred[j] - θ_true[j]) / max(abs(θ_true[j]), 1e-8)
            push!(param_errs, rel_err)
            push!(rows, (
                n=model.n,
                obs_mode=string(model.obs_mode),
                context=c,
                param_index=j,
                param_label=labels[j],
                theta_true=θ_true[j],
                theta_pred=θ_pred[j],
                rel_err=rel_err,
            ))
        end
    end

    return (
        param_rows=rows,
        mean_param_err=mean(param_errs),
        mean_prediction_loss=mean(pred_losses),
        ambient_dim=Int(model.d^4),
        representation_ratio=model.d^4 / length(model.abs_times),
        observation_dim=model.obs_model.obs_dim,
        snapshot_count=length(model.abs_times),
        observation_count=model.obs_model.obs_dim * length(model.abs_times),
    )
end

function dense_density_curves(model, c_demo; tmax=3.5, n_points=200)
    n = model.n
    θ_true = true_params(c_demo, n)
    θ_pred = predict_params(model.net, c_demo, n)
    L_true = model.A_const + sum(θ_true[j] * model.dirs[j] for j in eachindex(θ_true))
    L_pred = model.A_const + sum(θ_pred[j] * model.dirs[j] for j in eachindex(θ_pred))
    ρ0 = initial_density_vec(n)
    ts = range(0.0, tmax; length=n_points)
    ρ_true = [exp(Matrix(L_true) * t) * ρ0 for t in ts]
    ρ_pred = [exp(Matrix(L_pred) * t) * ρ0 for t in ts]
    return (; ts=collect(ts), ρ_true, ρ_pred, θ_true, θ_pred)
end
