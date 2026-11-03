"""Large explicit-sparse Lindblad benchmark for a memory-limited laptop.

Run one qubit count per process so `/usr/bin/time -l` can report peak resident
memory.  The reported gradient time includes both Arnoldi factorizations,
reduced path evaluation, and all library contractions.

Environment variables:
  LINDBLAD_MAC_Q          qubit count (default 10)
  LINDBLAD_MAC_M          Krylov depth (default 40)
  LINDBLAD_MAC_NQ         Gauss--Legendre nodes (default 8)
  LINDBLAD_MAC_REPEATS    timed repetitions (default 3)
  LINDBLAD_MAC_CHECK      run the depth/node refinement check (default true)
  LINDBLAD_MAC_OUT        output CSV path
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, SparseDirectionCache,
    arnoldi_basis!, _eval_krylov_batch, real_scalar_type
using CSV, DataFrames, ExponentialUtilities, LinearAlgebra, Printf, SparseArrays,
    Statistics

include(joinpath(@__DIR__, "optimized_shared_krylov_utils.jl"))

const Q = parse(Int, get(ENV, "LINDBLAD_MAC_Q", "10"))
const MK = parse(Int, get(ENV, "LINDBLAD_MAC_M", "40"))
const NQ = parse(Int, get(ENV, "LINDBLAD_MAC_NQ", "8"))
const REPEATS = parse(Int, get(ENV, "LINDBLAD_MAC_REPEATS", "3"))
const CHECK = lowercase(get(ENV, "LINDBLAD_MAC_CHECK", "true")) == "true"
const OUT = get(ENV, "LINDBLAD_MAC_OUT",
    joinpath(@__DIR__, "..", "benchmark_results", "lindblad_macbook_scaling_q$(Q).csv"))

site(op, index, q) = nsite_op_sparse(op, index, q)

function generator(q)
    H = 0.3 * sum(site(σz, index, q) for index in 1:q)
    for index in 1:(q - 1)
        H += 0.45 * (site(σx, index, q) * site(σx, index + 1, q) +
                     site(σy, index, q) * site(σy, index + 1, q))
    end
    A = hamiltonian_superop_sparse(H)
    for index in 1:q
        A += 0.08 * lindblad_superop_sparse(site(σz, index, q))
        A += 0.03 * lindblad_superop_sparse(site(σm, index, q))
    end
    A
end

function local_library(q)
    directions = SparseMatrixCSC{ComplexF64,Int}[]
    for index in 1:q
        push!(directions, lindblad_superop_sparse(site(σz, index, q)))
    end
    for index in 1:(q - 1)
        bond = site(σx, index, q) * site(σx, index + 1, q) +
               site(σy, index, q) * site(σy, index + 1, q)
        push!(directions, hamiltonian_superop_sparse(bond))
    end
    directions
end

function initial_state(q)
    zero_state = ComplexF64[1, 0]
    plus_state = ComplexF64[1, 1] / sqrt(2)
    psi = foldl(kron, [isodd(index) ? plus_state : zero_state for index in 1:q])
    kron(conj(psi), psi)
end

function cotangent(q)
    observable = sum((0.7 / index) * site(σz, index, q) +
                     (0.3 / index) * site(σx, index, q) for index in 1:q)
    vec(Matrix(observable))
end

median_ms(f) = (f(); median((@elapsed f()) * 1e3 for _ in 1:REPEATS))
megabytes(value) = Base.summarysize(value) / 2.0^20

function evaluate(A, directions, state, upstream, dt, depth, order)
    nodes, weights = gauss_legendre_01(order)
    workspace = OptimizedQuadratureWorkspace(directions, length(state), depth, order)
    result = optimized_linear_loss_and_vjp!(
        workspace, A, state, upstream, dt, nodes, weights; m_krylov=depth,
    )
    workspace, result, nodes, weights
end

function main()
    dt = 0.4
    @printf("Constructing q=%d explicit-sparse Lindblad problem\n", Q)
    A = generator(Q)
    directions = local_library(Q)
    state = initial_state(Q)
    upstream = cotangent(Q)
    n = length(state)
    operator_mb = megabytes(A) + sum(megabytes, directions)

    workspace, result, nodes, weights = evaluate(
        A, directions, state, upstream, dt, MK, NQ,
    )
    gradient = copy(result.parameter_vjp)
    prediction = copy(result.prediction)
    gradient_call = () -> optimized_linear_loss_and_vjp!(
        workspace, A, state, upstream, dt, nodes, weights; m_krylov=MK,
    )
    gradient_ms = median_ms(gradient_call)
    forward_call = () -> expv(dt, A, state; m=MK, tol=1e-12)
    forward_ms = median_ms(forward_call)
    workspace_mb = megabytes(workspace)

    # Retain only the numerical results before the refinement run.  Running
    # one workspace at a time keeps the large-system audit within laptop RAM.
    gradient_call = nothing
    result = nothing
    workspace = nothing
    GC.gc()

    forward_change = NaN
    gradient_change = NaN
    if CHECK
        reference_workspace, reference, _, _ = evaluate(
            A, directions, state, upstream, dt, MK + 8, NQ + 4,
        )
        forward_change = norm(prediction - reference.prediction) /
                         max(norm(reference.prediction), eps())
        reference_gradient = copy(reference.parameter_vjp)
        gradient_change = norm(gradient - reference_gradient) /
                          max(norm(reference_gradient), eps())
        reference_workspace = nothing
        reference = nothing
        GC.gc()
    end

    row = DataFrame((
        qubits=[Q], liouville_dimension=[n], directions=[length(directions)],
        krylov_depth=[MK], quadrature_nodes=[NQ], repeats=[REPEATS],
        generator_nnz=[nnz(A)], operator_storage_mb=[operator_mb],
        persistent_workspace_mb=[workspace_mb], forward_ms=[forward_ms],
        gradient_ms=[gradient_ms], refined_depth=[CHECK ? MK + 8 : 0],
        refined_nodes=[CHECK ? NQ + 4 : 0],
        forward_relative_change=[forward_change],
        gradient_relative_change=[gradient_change],
    ))
    mkpath(dirname(OUT))
    CSV.write(OUT, row)
    @printf("q=%d n=%d M=%d nnz(A)=%d\n", Q, n, length(directions), nnz(A))
    @printf("operator=%.1f MB workspace=%.1f MB forward=%.1f ms gradient=%.1f ms\n",
        operator_mb, workspace_mb, forward_ms, gradient_ms)
    @printf("refinement changes: forward=%.3e gradient=%.3e\n",
        forward_change, gradient_change)
    println("Saved $OUT")
end

main()
