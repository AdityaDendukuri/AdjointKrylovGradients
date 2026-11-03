"""
Structural identifiability of the candidate library, after Stanhope et al.

S. Stanhope, J. E. Rubin and D. Swigon, "Identifiability of linear and
linear-in-parameters dynamical systems from a single trajectory," SIAM J. Appl.
Dyn. Syst. 13(4):1792-1815, 2014.  Theorem 2.6: for the linear-in-parameters
family A(theta) = sum_j theta_j A_j, global identifiability from a single initial
condition p0 requires

    rank G(A) = M,     column j of G(A) = [ A_j p0 ; A_j A p0 ; ... ; A_j A^{n-1} p0 ].

G is a Krylov-type matrix: column j records how direction j acts along the
trajectory's Krylov sequence.  Two directions whose columns are nearly parallel
are nearly indistinguishable from that trajectory no matter how good the
optimizer is, so the resulting confusion is a property of the library and the
excitation rather than a thresholding failure.

The Schlogl audit leaves exactly one such confusion: half the initializer seeds
retain index 8 (birth feature x(x-1)(x-2)/6) alongside the true index 7
(birth feature x(x-1)/2).  This script tests whether that pair is in fact
poorly separated in the Stanhope sense.
"""

using AdjointKrylovGradients, Catalyst, JumpProcesses
using LinearAlgebra, Printf, Random, Statistics

const K1,K2,K3,K4 = 0.017, 0.0006, 1.7, 0.2
const R = [(1,), (-1,)]
const T_END, DT = 50.0, 0.5
const NTRAJ = parse(Int, get(ENV, "STANHOPE_TRAJ", "300"))
const KRYLOV_DEPTH = parse(Int, get(ENV, "STANHOPE_DEPTH", "40"))

"""
    stanhope_matrix(A, dirs, p0; depth)

Columns `A_j A^k p0` for `k = 0 .. depth-1`, stacked.  `depth = n` reproduces
Theorem 2.6 exactly; smaller values give the leading blocks, which is what a
finite observation window actually excites.
"""
function stanhope_matrix(A, dirs, p0; depth::Int)
    n = length(p0); M = length(dirs)
    G = zeros(Float64, n * depth, M)
    v = collect(Float64, p0)
    for k in 0:(depth - 1)
        for j in 1:M
            G[(k * n + 1):((k + 1) * n), j] .= dirs[j] * v
        end
        k < depth - 1 && (v = A * v)
    end
    G
end

Random.seed!(42)
rn = @reaction_network begin
    k1, 2X --> 3X; k2, 3X --> 2X; k3, 0 --> X; k4, X --> 0
end
trajs = [solve(JumpProblem(rn, DiscreteProblem(rn, [:X=>25], (0.0,T_END),
         [:k1=>K1,:k2=>K2,:k3=>K3,:k4=>K4]), Direct()), SSAStepper()) for _ in 1:NTRAJ]
times = collect(0.0:DT:T_END)
X = build_state_space(trajs, times, R; expansion=3, verbose=false)
P = build_distributions(trajs, times, X)
K = cubic_feature_dimension(1)
dA = build_derivatives(X, R; feature_map=cubic_features, K=K)

θt = Dict(z => zeros(K) for z in R)
θt[(1,)][1]=K3; θt[(1,)][3]=K1; θt[(-1,)][2]=K4; θt[(-1,)][4]=K2
A = build_generator(X, R, θt; feature_map=cubic_features)

sorted_R = sort(collect(R))
dirs   = vcat([[dA[z][f] for f in 1:K] for z in sorted_R]...)
labels = vcat([["$(z==(1,) ? "birth" : "death")·f$f" for f in 1:K] for z in sorted_R]...)
p0 = P[1]

@printf("states n = %d, library M = %d, Krylov depth = %d, trajectories = %d\n",
        length(X), length(dirs), KRYLOV_DEPTH, NTRAJ)

G = stanhope_matrix(A, dirs, p0; depth=KRYLOV_DEPTH)
# Column-normalize: rank and angles are scale-free, raw magnitudes are not.
Gn = similar(G)
for j in 1:size(G, 2)
    c = @view G[:, j]
    Gn[:, j] = norm(c) > 0 ? c ./ norm(c) : c
end
sv = svdvals(Gn)

println("\nSingular values of the column-normalized G(A):")
for (i, s) in enumerate(sv); @printf("  σ%-2d = %.6e\n", i, s); end
@printf("\nnumerical rank (tol 1e-10) = %d of %d\n", count(>(1e-10), sv), length(dirs))
@printf("condition number σ_max/σ_min = %.3e\n", sv[1] / sv[end])

function pairwise_report(Gn, labels)
    println("\nPairwise |cos angle| between normalized columns (>0.9 flagged):")
    worst = (0.0, 0, 0)
    M = size(Gn, 2)
    for i in 1:M, j in (i+1):M
        c = abs(dot(Gn[:, i], Gn[:, j]))
        c > worst[1] && (worst = (c, i, j))
        c > 0.9 && @printf("  %-12s vs %-12s : %.6f   <-- poorly separated\n",
                           labels[i], labels[j], c)
    end
    @printf("\nmost collinear pair: %s vs %s, |cos| = %.6f (angle %.3f deg)\n",
            labels[worst[2]], labels[worst[3]], worst[1],
            acosd(clamp(worst[1], -1, 1)))
    worst
end
pairwise_report(Gn, labels)

# The specific confusion the recovery audit reports: indices 7 and 8.
if length(dirs) >= 8
    c78 = abs(dot(Gn[:, 7], Gn[:, 8]))
    @printf("\nthe audited confusion, index 7 (%s) vs index 8 (%s):\n", labels[7], labels[8])
    @printf("   |cos| = %.6f   angle = %.3f degrees\n", c78, acosd(clamp(c78, -1, 1)))
    @printf("   true support keeps 7; half the seeds also retain 8.\n")
end
