using AdjointKrylovGradients, Catalyst, JumpProcesses, LinearAlgebra, Printf, Random
Random.seed!(42)
K1,K2,K3,K4 = 0.017, 0.0006, 1.7, 0.2
R = [(1,), (-1,)]; T_END=50.0; DT=0.5
rn = @reaction_network begin
    k1, 2X --> 3X; k2, 3X --> 2X; k3, 0 --> X; k4, X --> 0
end
trajs = [solve(JumpProblem(rn, DiscreteProblem(rn, [:X=>25], (0.0,T_END),
        [:k1=>K1,:k2=>K2,:k3=>K3,:k4=>K4]), Direct()), SSAStepper()) for _ in 1:300]
times = collect(0.0:DT:T_END)
X = build_state_space(trajs, times, R; expansion=3, verbose=false)
P = build_distributions(trajs, times, X)
K = cubic_feature_dimension(1)
dA = build_derivatives(X, R; feature_map=cubic_features, K=K)

θt = Dict(z => zeros(K) for z in R)
θt[(1,)][1]=K3; θt[(1,)][3]=K1; θt[(-1,)][2]=K4; θt[(-1,)][4]=K2
v = flatten_params(θt)

# same proxy stls.jl uses: sigma_j = sqrt(sum_k ||dA_j P_k||^2)
sorted_R = sort(collect(R)); scale = Float64[]
for z in sorted_R, f in 1:K
    s = sum(dot(dA[z][f]*P[k], dA[z][f]*P[k]) for k in 1:length(P)-1)
    push!(scale, sqrt(s))
end
lbl = vcat(["death·f$f" for f in 1:K], ["birth·f$f" for f in 1:K])
println(" idx  term        true θ      Fisher σ      scaled φ=θσ")
for j in eachindex(v)
    v[j] == 0 && continue
    @printf("%4d  %-11s %9.4g  %11.4g  %13.4g\n", j, lbl[j], v[j], scale[j], v[j]*scale[j])
end
φ = v .* scale
act = findall(!iszero, v)
@printf("\nsmallest ACTIVE coefficient in raw θ  : idx %d (%.4g)\n", act[argmin(v[act])], minimum(v[act]))
@printf("smallest ACTIVE coefficient in scaled φ: idx %d (%.4g)\n", act[argmin(φ[act])], minimum(φ[act]))
@printf("per-reaction: birth block φ = %s\n", string(round.(φ[5:8], sigdigits=3)))
