"""
Regime study for two evaluations of the same projected Frechet derivative.

For every Krylov depth, quadrature order, and library size, compare the
streaming Gauss--Legendre contraction with both the full and truncated
projected cores of Kandolf et al. (Algorithm 2).  Both paths rebuild the same
forward and adjoint Arnoldi spaces.  The truncated core uses the smallest rank
found by a singular-value-tolerance sweep whose library-gradient error is no
larger than the quadrature error.  Timings include the core SVD and are
recorded after warmup with rotating method order.

Environment knobs:
  LINDBLAD_REGIME_QUBITS    qubit count                         (default 5)
  LINDBLAD_REGIME_SIZES     comma-separated library sizes       (default 64,256,1024)
  LINDBLAD_REGIME_DEPTHS    comma-separated Krylov depths        (default 16,32,48,64)
  LINDBLAD_REGIME_ORDERS    comma-separated quadrature orders    (default 4,8,12,16)
  LINDBLAD_REGIME_REPEATS   timing repeats                      (default 5)
  LINDBLAD_REGIME_OUT       output CSV path
"""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, SparseDirectionCache,
    arnoldi_basis!, _eval_krylov_batch, _eval_reduced_exponential_batch,
    real_scalar_type
using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames

include(joinpath(@__DIR__, "optimized_shared_krylov_utils.jl"))

const Q = parse(Int, get(ENV, "LINDBLAD_REGIME_QUBITS", "5"))
const SIZES = parse.(Int, split(get(ENV, "LINDBLAD_REGIME_SIZES", "64,256,1024"), ","))
const DEPTHS = parse.(Int, split(get(ENV, "LINDBLAD_REGIME_DEPTHS", "16,32,48,64"), ","))
const ORDERS = parse.(Int, split(get(ENV, "LINDBLAD_REGIME_ORDERS", "4,8,12,16"), ","))
const REPEATS = parse(Int, get(ENV, "LINDBLAD_REGIME_REPEATS", "5"))
const OUT = get(ENV, "LINDBLAD_REGIME_OUT",
    joinpath(@__DIR__, "..", "benchmark_results", "lindblad_quadrature_core_regimes.csv"))
const DEPTH_OUT = get(ENV, "LINDBLAD_REGIME_DEPTH_OUT",
    joinpath(@__DIR__, "..", "benchmark_results", "lindblad_quadrature_core_depth.csv"))

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

function pauli_lindblad_library(q::Int, count::Int)
    strings = pauli_strings(q)
    count <= 2length(strings) || error("requested $count directions")
    directions = SparseMatrixCSC{ComplexF64,Int}[]
    for P in strings
        length(directions) == count && break
        push!(directions, hamiltonian_superop_sparse(P))
    end
    for P in strings
        length(directions) == count && break
        push!(directions, lindblad_superop_sparse(P))
    end
    directions
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
    ψ = foldl(kron, [isodd(i) ? plus : zero for i in 1:q])
    vec(ψ * ψ')
end

function observable_cotangent(q::Int)
    site = (op, i) -> nsite_op_sparse(op, i, q)
    O = sum((0.7 / i) * site(σz, i) + (0.3 / i) * site(σx, i) for i in 1:q)
    vec(Matrix(O))
end

function paired_median_times_ms(first, second)
    first()
    second()
    first_times = Float64[]
    second_times = Float64[]
    for repeat in 1:REPEATS
        if isodd(repeat)
            push!(first_times, (@elapsed first()) * 1000)
            push!(second_times, (@elapsed second()) * 1000)
        else
            push!(second_times, (@elapsed second()) * 1000)
            push!(first_times, (@elapsed first()) * 1000)
        end
    end
    median(first_times), median(second_times)
end

function rotating_median_times_ms(functions)
    foreach(f -> f(), functions)
    times = [Float64[] for _ in functions]
    for repeat in 1:REPEATS
        first_index = mod1(repeat, length(functions))
        order = vcat(first_index:length(functions), 1:(first_index - 1))
        for index in order
            push!(times[index], (@elapsed functions[index]()) * 1000)
        end
    end
    Tuple(median(values) for values in times)
end

function allocated_bytes(f)
    f()
    @allocated f()
end

function matched_core_atol(A, directions, cache, state, dt, cotangent, depth,
                           reference, target_error)
    n = length(state)
    fw = ArnoldiWorkspace(ComplexF64, n, depth)
    aw = ArnoldiWorkspace(ComplexF64, n, depth)
    fb = arnoldi_basis!(fw, A, state; m=depth)
    ab = arnoldi_basis!(aw, A', cotangent; m=depth)
    mf, ma = fb.m, ab.m
    block = zeros(ComplexF64, ma + mf, ma + mf)
    block[1:ma, 1:ma] .= ab.H
    block[(ma + 1):end, (ma + 1):end] .= fb.H'
    block[1, ma + 1] = ab.beta * fb.beta
    core = exp(dt .* block)[1:ma, (ma + 1):end]
    decomposition = svd(core)
    singular_values = decomposition.S
    left = view(ab.V, :, 1:ma) * decomposition.U
    right = view(fb.V, :, 1:mf) * decomposition.V
    kernel = zeros(ComplexF64, length(cache.rows))
    reference_norm = max(norm(reference), eps())
    selected_rank = length(singular_values)
    selected_error = 0.0
    for rank in eachindex(singular_values)
        sigma = singular_values[rank]
        for coordinate in eachindex(kernel)
            row = cache.rows[coordinate]
            column = cache.cols[coordinate]
            kernel[coordinate] += sigma * conj(left[row, rank]) *
                                  right[column, rank]
        end
        gradient = real.(transpose(cache.coefficients) * kernel)
        error = norm(gradient - reference) / reference_norm
        if error <= target_error
            selected_rank = rank
            selected_error = error
            break
        end
    end
    if selected_rank == length(singular_values)
        return 0.0, selected_rank, selected_error
    end
    tail_after = norm(view(singular_values, (selected_rank + 1):length(singular_values)))
    tail_before = hypot(tail_after, singular_values[selected_rank])
    atol = (tail_after + tail_before) / 2
    atol, selected_rank, selected_error
end

function run_regimes()
    max_size = maximum(SIZES)
    directions_all = pauli_lindblad_library(Q, max_size)
    A = benchmark_generator(Q)
    state = initial_state(Q)
    cotangent = observable_cotangent(Q)
    n = length(state)
    dt = 0.4
    rows = NamedTuple[]

    @printf("Lindblad quadrature/core regimes: q=%d, n=%d, Mmax=%d\n", Q, n, max_size)
    for depth in DEPTHS, library_size in SIZES
        directions = directions_all[1:library_size]
        cache = SparseDirectionCache(directions)
        core_forward = ArnoldiWorkspace(ComplexF64, n, depth)
        core_adjoint = ArnoldiWorkspace(ComplexF64, n, depth)
        core = () -> compressed_core_frechet_vjp(
            A, directions, state, dt, cotangent;
            m_krylov=depth, core_rtol=0.0, core_atol=0.0,
            ws_f=core_forward, ws_a=core_adjoint,
            direction_cache=cache, return_diagnostics=true,
        )
        core_result = core()
        core_value = copy(core_result.gradient)
        core_allocated = allocated_bytes(core)

        for order in ORDERS
            nodes, weights = gauss_legendre_01(order)
            workspace = OptimizedQuadratureWorkspace(
                directions, n, depth, order,
            )
            quadrature = () -> optimized_linear_loss_and_vjp!(
                workspace, A, state, cotangent, dt, nodes, weights;
                m_krylov=depth,
            )
            quadrature_value = copy(quadrature().parameter_vjp)
            discrepancy = norm(quadrature_value - core_value) /
                          max(norm(core_value), eps())

            compressed_quadrature_workspace = OptimizedQuadratureWorkspace(
                directions, n, depth, order,
            )
            compressed_quadrature_rank = order
            compressed_quadrature_error = discrepancy
            match_target = max(1e-14, discrepancy)
            for candidate_rank in 1:order
                candidate = compressed_quadrature_linear_loss_and_vjp!(
                    compressed_quadrature_workspace,
                    A, state, cotangent, dt, nodes, weights;
                    m_krylov=depth,
                    retained_rank=candidate_rank,
                )
                candidate_error = norm(candidate.parameter_vjp - core_value) /
                                  max(norm(core_value), eps())
                if candidate_error <= match_target
                    compressed_quadrature_rank = candidate_rank
                    compressed_quadrature_error = candidate_error
                    break
                end
            end
            compressed_quadrature = () ->
                compressed_quadrature_linear_loss_and_vjp!(
                    compressed_quadrature_workspace,
                    A, state, cotangent, dt, nodes, weights;
                    m_krylov=depth,
                    retained_rank=compressed_quadrature_rank,
                )

            selected_atol, selected_rank, selected_error = matched_core_atol(
                A, directions, cache, state, dt, cotangent, depth,
                core_value, discrepancy,
            )
            compressed_forward = ArnoldiWorkspace(ComplexF64, n, depth)
            compressed_adjoint = ArnoldiWorkspace(ComplexF64, n, depth)
            compressed = () -> compressed_core_frechet_vjp(
                A, directions, state, dt, cotangent;
                m_krylov=depth, core_rtol=0.0, core_atol=selected_atol,
                ws_f=compressed_forward, ws_a=compressed_adjoint,
                direction_cache=cache, return_diagnostics=true,
            )
            selected_result = compressed()
            selected_result.core_rank == selected_rank || error(
                "rank-selection mismatch: expected $selected_rank, got " *
                "$(selected_result.core_rank), atol=$selected_atol",
            )
            selected_error = norm(selected_result.gradient - core_value) /
                             max(norm(core_value), eps())
            quadrature_ms, compressed_quadrature_ms, compressed_ms, core_ms =
                rotating_median_times_ms(
                    (quadrature, compressed_quadrature, compressed, core),
                )
            quadrature_allocated = allocated_bytes(quadrature)
            compressed_quadrature_allocated =
                allocated_bytes(compressed_quadrature)
            compressed_allocated = allocated_bytes(compressed)
            # Evaluation-specific storage beyond the common Arnoldi bases.
            # ComplexF64 values occupy 16 bytes.  Retained storage counts path
            # or low-rank factors.  Peak workspace also counts six m-by-m
            # arrays for the reduced block exponential, core, and SVD.
            quadrature_storage = 16 * 2n * (order + 1)
            compressed_quadrature_retained_storage = 16 * (
                2n * compressed_quadrature_rank + compressed_quadrature_rank
            )
            compressed_quadrature_peak_storage = 16 * (
                2n * compressed_quadrature_rank +
                6depth * order + 4order^2
            )
            core_storage = 16 * (2n * depth + 6depth^2)
            compressed_storage = 16 * (
                2n * selected_result.core_rank + 6depth^2
            )
            full_retained_storage = 16 * (2n * depth + depth)
            compressed_retained_storage = 16 * (
                2n * selected_result.core_rank + selected_result.core_rank
            )
            push!(rows, (
                qubits=Q, liouville_dim=n, library_size=library_size,
                krylov_depth=depth, quadrature_order=order,
                repeats=REPEATS, quadrature_ms=quadrature_ms,
                compressed_quadrature_ms=compressed_quadrature_ms,
                compressed_core_ms=compressed_ms,
                core_ms=core_ms, core_over_quadrature=core_ms / quadrature_ms,
                compressed_over_quadrature=compressed_ms / quadrature_ms,
                compressed_quadrature_over_quadrature=
                    compressed_quadrature_ms / quadrature_ms,
                quadrature_relative_error=discrepancy,
                compressed_quadrature_relative_error=
                    compressed_quadrature_error,
                compressed_quadrature_rank=compressed_quadrature_rank,
                compressed_relative_error=selected_error,
                selected_core_atol=selected_atol,
                selected_core_rank=selected_result.core_rank,
                quadrature_allocated_bytes=quadrature_allocated,
                compressed_quadrature_allocated_bytes=
                    compressed_quadrature_allocated,
                compressed_core_allocated_bytes=compressed_allocated,
                core_allocated_bytes=core_allocated,
                allocation_ratio=core_allocated / max(quadrature_allocated, 1),
                quadrature_path_storage_bytes=quadrature_storage,
                compressed_quadrature_peak_storage_bytes=
                    compressed_quadrature_peak_storage,
                compressed_quadrature_retained_storage_bytes=
                    compressed_quadrature_retained_storage,
                compressed_peak_storage_bytes=compressed_storage,
                core_factor_storage_bytes=core_storage,
                compressed_over_quadrature_peak_storage=
                    compressed_storage / quadrature_storage,
                storage_ratio=core_storage / quadrature_storage,
                compressed_retained_storage_bytes=compressed_retained_storage,
                full_retained_storage_bytes=full_retained_storage,
                compressed_over_quadrature_retained_storage=
                    compressed_retained_storage / quadrature_storage,
                core_rank=core_result.core_rank,
            ))
            @printf("  m=%2d M=%4d nq=%2d rq=%2d rc=%2d  quad=%7.2f ms qcomp=%7.2f ms corecomp=%7.2f ms full=%7.2f ms  err(q/qc/cc)=%.2e/%.2e/%.2e\n",
                depth, library_size, order, compressed_quadrature_rank,
                selected_result.core_rank, quadrature_ms,
                compressed_quadrature_ms, compressed_ms, core_ms,
                discrepancy, compressed_quadrature_error, selected_error)
            @printf("      retained(qc/q)=%4.2fx peak(qc/q)=%4.2fx retained(cc/q)=%4.2fx peak(cc/q)=%4.2fx\n",
                compressed_quadrature_retained_storage / quadrature_storage,
                compressed_quadrature_peak_storage / quadrature_storage,
                compressed_retained_storage / quadrature_storage,
                compressed_storage / quadrature_storage)
        end
    end
    mkpath(dirname(OUT))
    results = DataFrame(rows)
    CSV.write(OUT, results)
    depth_results = filter(row ->
        row.library_size == maximum(SIZES) && row.quadrature_order == 8,
        results,
    )
    CSV.write(DEPTH_OUT, depth_results)
    println("Saved $OUT")
    println("Saved $DEPTH_OUT")
end

run_regimes()
