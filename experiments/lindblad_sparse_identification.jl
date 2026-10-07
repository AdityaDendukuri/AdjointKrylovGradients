"""
Sparse identification of an open-quantum-system generator.

This replaces the earlier context-encoder experiment.  That version trained a
neural network to map a scalar context to the M = 2n-1 parameters of a known
Ising-plus-dephasing library, so every direction was active and the task was
dense regression through the exponential layer.

Here the task is sparse identification instead, matching the CME experiments:
the candidate library is deliberately overcomplete, containing

  * XX+YY exchange couplings on **all** site pairs (i,j), not only neighbours;
  * dephasing (sigma_z), decay (sigma_-) and bit-flip (sigma_x) dissipators on
    every site,

while the true generator activates only nearest-neighbour couplings and
sigma_z dephasing.  The method must find that support without being told it.

The generator is complex, so this is also the paper's evidence that the
shared-Krylov kernel and the Fisher-preconditioned STLS work unchanged on
complex operators.

Environment knobs:
  LINDBLAD_SPARSE_QUBITS   comma-separated qubit counts   (default 2,3,4)
  LINDBLAD_SPARSE_SEEDS    final seed index               (default 5)
  LINDBLAD_SPARSE_FIRST_SEED first seed index              (default 1)
  LINDBLAD_SPARSE_BLAS_THREADS optional BLAS thread count
  LINDBLAD_SPARSE_NQ       quadrature order               (default 16)
  LINDBLAD_SPARSE_OUT      output CSV path
"""

using AdjointKrylovGradients
using LinearAlgebra, SparseArrays, Random, Printf, Statistics
using CSV, DataFrames

const QUBITS = parse.(Int, split(get(ENV, "LINDBLAD_SPARSE_QUBITS", "2,3,4"), ","))
const NSEEDS = parse(Int, get(ENV, "LINDBLAD_SPARSE_SEEDS", "5"))
const FIRST_SEED = parse(Int, get(ENV, "LINDBLAD_SPARSE_FIRST_SEED", "1"))
haskey(ENV, "LINDBLAD_SPARSE_BLAS_THREADS") && BLAS.set_num_threads(parse(Int, ENV["LINDBLAD_SPARSE_BLAS_THREADS"]))
const NQ     = parse(Int, get(ENV, "LINDBLAD_SPARSE_NQ", "16"))
const OUT    = get(ENV, "LINDBLAD_SPARSE_OUT",
                   joinpath(@__DIR__, "..", "benchmark_results",
                            "lindblad_sparse_identification.csv"))

const H_FIELD = 0.3

"""
Overcomplete Lindblad library for `n` qubits.

Returns `(A_const, dirs, labels, true_index)` where `true_index` maps each
physically active mechanism to its position in `dirs`.
"""
function overcomplete_lindblad_library(n::Int)
    site  = (op, i) -> nsite_op_sparse(op, i, n)
    hsup  = hamiltonian_superop_sparse
    dsup  = lindblad_superop_sparse

    # Fixed transverse field, not part of the identification problem.
    H_field = sum(site(σz, i) for i in 1:n)
    A_const = H_FIELD * hsup(H_field)

    dirs   = SparseMatrixCSC{ComplexF64,Int}[]
    labels = String[]
    neighbour = Int[]

    # All pairwise exchange couplings; only |i-j| == 1 is physically active.
    for i in 1:n-1, j in i+1:n
        H_ij = site(σx, i) * site(σx, j) + site(σy, i) * site(σy, j)
        push!(dirs, hsup(H_ij))
        push!(labels, "J($i,$j)")
        j - i == 1 && push!(neighbour, length(dirs))
    end

    # Dissipators in three channels per site; only sigma_z is active.
    dephasing = Int[]
    for (name, op) in (("z", σz), ("m", σm), ("x", σx))
        for i in 1:n
            push!(dirs, dsup(site(op, i)))
            push!(labels, "D$name($i)")
            name == "z" && push!(dephasing, length(dirs))
        end
    end

    (A_const, dirs, labels, vcat(neighbour, dephasing))
end

initial_density_vec(n) = begin
    plus = ComplexF64[1, 1] / sqrt(2)
    ψ = foldl(kron, [plus for _ in 1:n])
    vec(ψ * ψ')
end

rows = NamedTuple[]

for n in QUBITS
    A_const, dirs, labels, true_idx = overcomplete_lindblad_library(n)
    M  = length(dirs)
    d2 = 4^n
    mk = min(40, d2 - 1)
    dts = fill(0.4, 6)

    @printf("\n=== %d qubits: Liouville dim %d, library M = %d, true support %d ===\n",
            n, d2, M, length(true_idx))

    for seed in FIRST_SEED:NSEEDS
        rng = MersenneTwister(1000 + seed)

        # True sparse parameters: couplings ~ 0.5, dephasing ~ 0.1, jittered.
        θ_true = zeros(M)
        for j in true_idx
            base = startswith(labels[j], "J") ? 0.5 : 0.1
            θ_true[j] = base * (1 + 0.3 * (2 * rand(rng) - 1))
        end

        A_true = assemble_generator(A_const, dirs, θ_true)
        ρ0 = initial_density_vec(n)
        snaps = [ρ0]
        for dt in dts
            push!(snaps, exp(Matrix(A_true) * dt) * snaps[end])
        end

        # Uninformative start: every candidate active at a common small value.
        θ0 = fill(0.15, M)

        res = sparse_generator_stls(
            A_const, dirs, snaps, dts, θ0;
            threshold_fraction = 0.10,
            outer_iter = 15, inner_iter = 12, polish_iter = 120,
            m_krylov = mk, n_quad = NQ,
            fisher_precond = true, enforce_positive = true,
            verbose = false,
        )

        recovered = Set(res.support)
        truth     = Set(true_idx)
        tp = length(intersect(recovered, truth))
        fp = length(setdiff(recovered, truth))
        fn = length(setdiff(truth, recovered))
        exact = recovered == truth
        rate_err = norm(res.theta[true_idx] - θ_true[true_idx]) / norm(θ_true[true_idx])

        push!(rows, (
            qubits = n, liouville_dim = d2, library_size = M,
            true_support = length(true_idx), seed = seed, n_quad = NQ,
            m_krylov = mk, recovered_support = length(res.support),
            true_positives = tp, false_positives = fp, false_negatives = fn,
            exact_support = exact, rate_relerr = rate_err,
            final_loss = res.loss,
            converged = res.converged, outer_iterations = res.iterations,
        ))

        @printf("  seed %d: support %d/%d  fp=%d fn=%d  exact=%-5s  rate err %.3e  loss %.2e\n",
                seed, tp, length(true_idx), fp, fn, exact, rate_err, res.loss)
    end
end

df = DataFrame(rows)
mkpath(dirname(OUT))
CSV.write(OUT, df)

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
for n in QUBITS
    sub = df[df.qubits .== n, :]
    nrow(sub) == 0 && continue
    @printf("%d qubits (M=%d): exact support %d/%d seeds, mean rate error %.3e (median %.3e)\n",
            n, sub.library_size[1], count(sub.exact_support), nrow(sub),
            mean(sub.rate_relerr), median(sub.rate_relerr))
end
println("\nSaved $(OUT)")
