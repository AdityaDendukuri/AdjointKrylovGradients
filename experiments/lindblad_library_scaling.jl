"""
Library-size scaling for adjoint Krylov gradients of a Lindblad propagator.

The benchmark holds the four-qubit generator, state, cotangent, Krylov depth,
and quadrature rule fixed while increasing the number of structured Pauli
directions.  It compares one shared forward/adjoint construction with a
per-direction augmented-matrix Fréchet action.  A separate two-qubit audit
starts at an exactly stationary amplitude-damping state: forward Arnoldi
terminates in one step, but a Hamiltonian perturbation has nonzero sensitivity.

Environment knobs:
  LINDBLAD_SCALE_QUBITS       scaling qubit count          (default 4)
  LINDBLAD_SCALE_SIZES        comma-separated library sizes (default 16,64,256,510)
  LINDBLAD_SCALE_M            Krylov depth                 (default 32)
  LINDBLAD_SCALE_NQ           quadrature order             (default 16)
  LINDBLAD_SCALE_REPEATS      timing repeats               (default 5)
  LINDBLAD_SCALE_OUT          output CSV path
  LINDBLAD_STATIONARY_OUT     stationary-audit CSV path
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, SparseDirectionCache,
    arnoldi_basis!, _eval_krylov_batch, real_scalar_type
using LinearAlgebra, SparseArrays, Random, Printf, Statistics
using ExponentialUtilities: expv
using CSV, DataFrames

include(joinpath(@__DIR__, "optimized_shared_krylov_utils.jl"))

const Q = parse(Int, get(ENV, "LINDBLAD_SCALE_QUBITS", "4"))
const SIZES = parse.(Int, split(get(ENV, "LINDBLAD_SCALE_SIZES", "16,64,256,510"), ","))
const MK = parse(Int, get(ENV, "LINDBLAD_SCALE_M", "40"))
const NQ = parse(Int, get(ENV, "LINDBLAD_SCALE_NQ", "8"))
const CORE_RTOL = parse(Float64, get(ENV, "LINDBLAD_CORE_RTOL", "0"))
const REPEATS = parse(Int, get(ENV, "LINDBLAD_SCALE_REPEATS", "5"))
const OUT = get(ENV, "LINDBLAD_SCALE_OUT",
    joinpath(@__DIR__, "..", "benchmark_results", "lindblad_library_scaling.csv"))
const STATIONARY_OUT = get(ENV, "LINDBLAD_STATIONARY_OUT",
    joinpath(@__DIR__, "..", "benchmark_results", "lindblad_stationary_audit.csv"))

const PAULIS = (sparse(σx), sparse(σy), sparse(σz))

function pauli_strings(q::Int)
    identity = sparse(ComplexF64[1 0; 0 1])
    strings = SparseMatrixCSC{ComplexF64,Int}[]
    for code in 1:(4^q - 1)
        pauli_digits = digits(code; base=4, pad=q)
        factors = [digit == 0 ? identity : PAULIS[digit] for digit in pauli_digits]
        push!(strings, foldl(kron, factors))
    end
    strings
end

function pauli_lindblad_library(q::Int)
    strings = pauli_strings(q)
    hamiltonian = [hamiltonian_superop_sparse(P) for P in strings]
    dissipative = [lindblad_superop_sparse(P) for P in strings]
    vcat(hamiltonian, dissipative)
end

function benchmark_generator(q::Int)
    site = (op, i) -> nsite_op_sparse(op, i, q)
    H = 0.3 * sum(site(σz, i) for i in 1:q)
    for i in 1:q-1
        H += 0.45 * (site(σx, i) * site(σx, i + 1) +
                     site(σy, i) * site(σy, i + 1))
    end
    A = hamiltonian_superop_sparse(H)
    for i in 1:q
        A += 0.08 * lindblad_superop_sparse(site(σz, i))
        A += 0.03 * lindblad_superop_sparse(site(σm, i))
    end
    A
end

function initial_state(q::Int)
    zero = ComplexF64[1, 0]
    plus = ComplexF64[1, 1] / sqrt(2)
    factors = [isodd(i) ? plus : zero for i in 1:q]
    ψ = foldl(kron, factors)
    vec(ψ * ψ')
end

function observable_cotangent(q::Int)
    site = (op, i) -> nsite_op_sparse(op, i, q)
    O = sum((0.7 / i) * site(σz, i) + (0.3 / i) * site(σx, i) for i in 1:q)
    vec(Matrix(O))
end

function median_time_ms(f)
    f()
    median((@elapsed f()) * 1000 for _ in 1:REPEATS)
end

function frechet_block(A, direction)
    n = size(A, 1)
    [A direction; spzeros(ComplexF64, n, n) A]
end

function directional_gradient(block, state, cotangent, dt; m=MK, tol=1e-13)
    n = length(state)
    seed = vcat(zeros(ComplexF64, n), state)
    real(dot(cotangent, view(expv(dt, block, seed; m=m, tol=tol), 1:n)))
end

function run_scaling()
    dirs_all = pauli_lindblad_library(Q)
    maximum(SIZES) <= length(dirs_all) || error(
        "requested $(maximum(SIZES)) directions, but the $Q-qubit Pauli library has $(length(dirs_all))",
    )
    A = benchmark_generator(Q)
    state = initial_state(Q)
    cotangent = observable_cotangent(Q)
    dt = 0.4
    nodes, weights = gauss_legendre_01(NQ)
    rows = NamedTuple[]
    blocks_all = [frechet_block(A, D) for D in dirs_all[1:maximum(SIZES)]]
    reference_depth = min(2 * MK, length(state))
    reference_all = [directional_gradient(B, state, cotangent, dt;
        m=reference_depth, tol=1e-15) for B in blocks_all]

    @printf("Lindblad scaling: q=%d, Liouville dimension=%d, available directions=%d\n",
        Q, length(state), length(dirs_all))
    for M in SIZES
        dirs = dirs_all[1:M]
        workspace = OptimizedQuadratureWorkspace(dirs, length(state), MK, NQ)
        shared = () -> optimized_linear_loss_and_vjp!(
            workspace, A, state, cotangent, dt, nodes, weights; m_krylov=MK,
        )
        shared_value = copy(shared().parameter_vjp)
        shared_ms = median_time_ms(shared)

        core_cache = SparseDirectionCache(dirs)
        core_forward = ArnoldiWorkspace(ComplexF64, length(state), MK)
        core_adjoint = ArnoldiWorkspace(ComplexF64, length(state), MK)
        core = () -> compressed_core_frechet_vjp(
            A, dirs, state, dt, cotangent;
            m_krylov=MK, core_rtol=CORE_RTOL, core_atol=0.0,
            ws_f=core_forward, ws_a=core_adjoint,
            direction_cache=core_cache, return_diagnostics=true,
        )
        core_result = core()
        core_value = core_result.gradient
        core_ms = median_time_ms(core)

        blocks = blocks_all[1:M]
        directional = () -> [directional_gradient(B, state, cotangent, dt) for B in blocks]
        directional_value = directional()
        directional_ms = median_time_ms(directional)
        reference = reference_all[1:M]
        relative_difference = norm(shared_value - directional_value) /
                              max(norm(directional_value), eps())
        shared_error = norm(shared_value - reference) / max(norm(reference), eps())
        core_error = norm(core_value - reference) / max(norm(reference), eps())
        directional_error = norm(directional_value - reference) / max(norm(reference), eps())
        speedup = directional_ms / shared_ms
        push!(rows, (
            qubits=Q, liouville_dim=length(state), library_size=M,
            krylov_depth=MK, quadrature_order=NQ, repeats=REPEATS,
            shared_ms=shared_ms, directional_ms=directional_ms,
            speedup=speedup, relative_difference=relative_difference,
            shared_reference_error=shared_error,
            directional_reference_error=directional_error,
            core_ms=core_ms, core_reference_error=core_error,
            core_rank=core_result.core_rank, core_rtol=CORE_RTOL,
        ))
        @printf("  M=%3d  quadrature=%7.2f ms  core=%7.2f ms  directional=%8.2f ms  speedup=%6.1fx  errors=(%.2e, %.2e, %.2e)\n",
            M, shared_ms, core_ms, directional_ms, speedup,
            shared_error, core_error, directional_error)
    end
    mkpath(dirname(OUT))
    CSV.write(OUT, DataFrame(rows))
end

function run_stationary_audit()
    q = 2
    site = (op, i) -> nsite_op_sparse(op, i, q)
    A = 0.4 * hamiltonian_superop_sparse(site(σz, 1) + site(σz, 2))
    for i in 1:q
        A += 0.3 * lindblad_superop_sparse(site(σm, i))
    end
    direction = hamiltonian_superop_sparse(site(σx, 1))
    # With the package convention sigma_- = |1><0|, |1> is the dark state.
    ground = ComplexF64[0, 1]
    ψ = kron(ground, ground)
    state = vec(ψ * ψ')
    cotangent = vec(Matrix(site(σy, 1)))
    dt = 0.7
    nodes, weights = gauss_legendre_01(24)
    workspace = OptimizedQuadratureWorkspace([direction], length(state), 8, 24)
    result = optimized_linear_loss_and_vjp!(
        workspace, A, state, cotangent, dt, nodes, weights; m_krylov=8,
    )
    shared = only(result.parameter_vjp)
    reference = directional_gradient(frechet_block(A, direction), state, cotangent, dt)
    forward_residual = norm(A * state)
    relative_error = abs(shared - reference) / max(abs(reference), eps())
    row = DataFrame((
        qubits=[q], liouville_dim=[length(state)], forward_dimension=[result.forward_dimension],
        forward_residual=[forward_residual], shared_gradient=[shared],
        block_frechet_gradient=[reference], relative_error=[relative_error],
    ))
    mkpath(dirname(STATIONARY_OUT))
    CSV.write(STATIONARY_OUT, row)
    @printf("Stationary audit: forward dimension=%d, ||A rho||=%.2e, gradient=%.8e, rel.err=%.2e\n",
        result.forward_dimension, forward_residual, shared, relative_error)
end

run_scaling()
run_stationary_audit()
println("Saved $OUT")
println("Saved $STATIONARY_OUT")
