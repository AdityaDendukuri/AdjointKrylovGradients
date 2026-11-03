# AdjointKrylovGradients

Gradients of $u = e^{\Delta t\,A(\theta)} p_0$ with respect to $\theta$, where
$A(\theta) = A_0 + \sum_j \theta_j A_j$ is large and sparse. Code for
[arXiv:2610.15940](https://arxiv.org/abs/2610.15940).

## Use

```julia
using AdjointKrylovGradients, Zygote

layer = SemigroupLayer(A0, dirs)                  # m_krylov = 25, n_quad = 5
u = layer(p0, θ, dt)                              # exp(dt * A(θ)) * p0
g = gradient(θ -> sum(abs2, layer(p0, θ, dt) - u_obs), θ)[1]   # uses the rrule
```

Sparse fit of $\theta$ from snapshots: `sparse_generator_stls(A0, dirs, snapshots, dts, θ0)`.

## Neural rates

The layer composes with Flux, so a network can map a context $c$ to $\theta(c)$ and train
through $e^{\Delta t\,A(\theta(c))} p_0$:

```julia
using AdjointKrylovGradients, Flux

layer = SemigroupLayer(A0, dirs)
net = Chain(Dense(1 => 32, tanh), Dense(32 => length(dirs), softplus)) |> f64   # c -> θ ≥ 0

loss(net) = sum(sum(abs2, layer(p0, net([c]), dt) - u_obs[c]) for c in contexts)

opt = Flux.setup(Adam(1e-2), net)
for epoch in 1:1000
    l, g = Flux.withgradient(loss, net)
    Flux.update!(opt, net, g[1])
end
```

## Run

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=experiments experiments/gradient_accuracy.jl
```
