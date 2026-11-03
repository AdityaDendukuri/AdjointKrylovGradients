"""Matched-basis timing of streaming quadrature and Kandolf Algorithm 2."""

using AdjointKrylovGradients
using AdjointKrylovGradients: ArnoldiWorkspace, SparseDirectionCache,
    arnoldi_basis!, _eval_krylov_batch, real_scalar_type
using LinearAlgebra, SparseArrays, Statistics, Printf
using CSV, DataFrames
using FastGaussQuadrature: gausslobatto

include("optimized_shared_krylov_utils.jl")
include("../src/fpe.jl")
include("brusselator_error_control.jl")

const REPEATS_CROSS = parse(Int, get(ENV, "CROSS_REPEATS", "9"))
const ORDERS_CROSS = parse.(Int, split(get(ENV, "CROSS_ORDERS", "2,3,4,5,6,8,12,16,24"), ","))
const OUT_CROSS = get(ENV, "CROSS_OUT",
    joinpath(@__DIR__, "..", "benchmark_results", "core_quadrature_cross_system.csv"))

function median_ms(f)
    f()
    median((@elapsed f()) * 1000 for _ in 1:REPEATS_CROSS)
end

function benchmark_case(name, A, dirs, state, cotangent, dt, m, nq)
    n = length(state)
    cache = SparseDirectionCache(dirs)
    nodes, weights = gauss_legendre_01(nq)
    qworkspace = OptimizedQuadratureWorkspace(dirs, n, m, nq)
    quadrature = () -> optimized_linear_loss_and_vjp!(
        qworkspace, A, state, cotangent, dt, nodes, weights; m_krylov=m,
    ).parameter_vjp
    fworkspace = ArnoldiWorkspace(eltype(state), n, m)
    aworkspace = ArnoldiWorkspace(eltype(state), n, m)
    core = () -> compressed_core_frechet_vjp(
        A, dirs, state, dt, cotangent;
        m_krylov=m, core_rtol=0.0, core_atol=0.0,
        ws_f=fworkspace, ws_a=aworkspace, direction_cache=cache,
    )
    qvalue, cvalue = copy(quadrature()), copy(core())
    component_scale = max(norm(cvalue)/sqrt(length(cvalue)), eps())
    component_errors = abs.(qvalue-cvalue) ./ max.(abs.(cvalue), component_scale)
    qtime, ctime = median_ms(quadrature), median_ms(core)
    scalar_bytes = sizeof(eltype(state))
    qstorage = scalar_bytes * 2n * (nq + 1)
    cstorage = scalar_bytes * (2n * m + 6m^2)
    common_basis_storage = scalar_bytes * 2n * m
    row = (
        system=name, n=n, M=length(dirs), dt=dt, krylov_depth=m,
        quadrature_rule="Gauss-Legendre", quadrature_order=nq, repeats=REPEATS_CROSS,
        quadrature_ms=qtime, core_ms=ctime,
        core_over_quadrature=ctime/qtime,
        relative_difference=norm(qvalue-cvalue)/max(norm(cvalue), eps()),
        component_fraction_below_1e6=mean(component_errors .<= 1e-6),
        component_fraction_below_1e8=mean(component_errors .<= 1e-8),
        component_error_p95=quantile(component_errors, 0.95),
        quadrature_extra_storage_bytes=qstorage,
        core_extra_storage_bytes=cstorage,
        core_over_quadrature_storage=cstorage/qstorage,
        common_basis_storage_bytes=common_basis_storage,
        core_over_quadrature_total_storage=
            (common_basis_storage+cstorage)/(common_basis_storage+qstorage),
    )
    @printf("%-12s n=%4d M=%3d m=%2d nq=%2d quad=%8.3f ms core=%8.3f ms ratio=%5.2f storage=%5.2f err=%.2e\n",
        name, n, length(dirs), m, nq, qtime, ctime, ctime/qtime,
        (common_basis_storage+cstorage)/(common_basis_storage+qstorage),
        row.relative_difference)
    row
end

function lobatto_vjp!(workspace, A, state, cotangent, dt, nodes, weights; m_krylov)
    interior = length(nodes) - 2
    workspace.forward_times[1] = dt
    workspace.adjoint_times[1] = dt
    for k in 1:interior
        workspace.forward_times[k+1] = dt * nodes[k+1]
        workspace.adjoint_times[k+1] = dt * (1-nodes[k+1])
    end
    fb = arnoldi_basis!(workspace.forward, A, state; m=m_krylov)
    ab = arnoldi_basis!(workspace.adjoint, A', cotangent; m=m_krylov)
    fv = _eval_krylov_batch(fb, workspace.forward_times, fb.beta)
    av = _eval_krylov_batch(ab, workspace.adjoint_times, ab.beta)
    cache, kernel = workspace.direction_cache, workspace.kernel_values
    fill!(kernel, zero(eltype(kernel)))
    for coordinate in eachindex(kernel)
        row, column = cache.rows[coordinate], cache.cols[coordinate]
        kernel[coordinate] += weights[1] * conj(av[row,1]) * state[column]
        for k in 1:interior
            kernel[coordinate] += weights[k+1] * conj(av[row,k+1]) * fv[column,k+1]
        end
        kernel[coordinate] += weights[end] * conj(cotangent[row]) * fv[column,1]
    end
    mul!(workspace.projected_values, transpose(cache.coefficients), kernel)
    for j in eachindex(workspace.gradient)
        workspace.gradient[j] = real(dt * workspace.projected_values[j])
    end
    workspace.gradient
end

function benchmark_lobatto_case(name, A, dirs, state, cotangent, dt, m, nq)
    x, weights = gausslobatto(nq)
    nodes = (x .+ 1) ./ 2
    weights ./= 2
    n = length(state)
    cache = SparseDirectionCache(dirs)
    workspace = OptimizedQuadratureWorkspace(dirs, n, m, nq-2)
    quadrature = () -> lobatto_vjp!(workspace, A, state, cotangent, dt,
        nodes, weights; m_krylov=m)
    fworkspace = ArnoldiWorkspace(eltype(state), n, m)
    aworkspace = ArnoldiWorkspace(eltype(state), n, m)
    core = () -> compressed_core_frechet_vjp(A, dirs, state, dt, cotangent;
        m_krylov=m, core_rtol=0.0, core_atol=0.0,
        ws_f=fworkspace, ws_a=aworkspace, direction_cache=cache)
    qvalue, cvalue = copy(quadrature()), copy(core())
    component_scale = max(norm(cvalue)/sqrt(length(cvalue)), eps())
    component_errors = abs.(qvalue-cvalue) ./ max.(abs.(cvalue), component_scale)
    qtime, ctime = median_ms(quadrature), median_ms(core)
    scalar_bytes = sizeof(eltype(state))
    qstorage = scalar_bytes * 2n * max(nq-1, 1)
    cstorage = scalar_bytes * (2n*m + 6m^2)
    common = scalar_bytes * 2n*m
    row = (system=name, n=n, M=length(dirs), dt=dt, krylov_depth=m,
        quadrature_rule="Gauss-Lobatto", quadrature_order=nq,
        repeats=REPEATS_CROSS, quadrature_ms=qtime, core_ms=ctime,
        core_over_quadrature=ctime/qtime,
        relative_difference=norm(qvalue-cvalue)/max(norm(cvalue),eps()),
        component_fraction_below_1e6=mean(component_errors .<= 1e-6),
        component_fraction_below_1e8=mean(component_errors .<= 1e-8),
        component_error_p95=quantile(component_errors, 0.95),
        quadrature_extra_storage_bytes=qstorage,
        core_extra_storage_bytes=cstorage,
        core_over_quadrature_storage=cstorage/qstorage,
        common_basis_storage_bytes=common,
        core_over_quadrature_total_storage=(common+cstorage)/(common+qstorage))
    @printf("%-12s Lobatto n=%2d quad=%8.3f ms core=%8.3f ms ratio=%5.2f err=%.2e\n",
        name, nq, qtime, ctime, ctime/qtime, row.relative_difference)
    row
end

function benchmark_commuting_case(name, A, dirs, state, cotangent, dt, m)
    n = length(state)
    fworkspace = ArnoldiWorkspace(eltype(state), n, m)
    buffer = similar(state)
    gradient = zeros(real_scalar_type(eltype(state)), length(dirs))
    shortcut = () -> begin
        fb = arnoldi_basis!(fworkspace, A, state; m=m)
        endpoint = _eval_krylov_batch(fb, [dt], fb.beta)
        for j in eachindex(dirs)
            mul!(buffer, dirs[j], view(endpoint,:,1))
            gradient[j] = real(dt * dot(cotangent, buffer))
        end
        gradient
    end
    cache = SparseDirectionCache(dirs)
    cf = ArnoldiWorkspace(eltype(state), n, m)
    ca = ArnoldiWorkspace(eltype(state), n, m)
    core = () -> compressed_core_frechet_vjp(A, dirs, state, dt, cotangent;
        m_krylov=m, core_rtol=0.0, core_atol=0.0,
        ws_f=cf, ws_a=ca, direction_cache=cache)
    qvalue, cvalue = copy(shortcut()), copy(core())
    component_scale = max(norm(cvalue)/sqrt(length(cvalue)), eps())
    component_errors = abs.(qvalue-cvalue) ./ max.(abs.(cvalue), component_scale)
    qtime, ctime = median_ms(shortcut), median_ms(core)
    scalar_bytes = sizeof(eltype(state))
    qstorage = scalar_bytes*n*m
    cstorage = scalar_bytes*(2n*m+6m^2)
    row = (system=name, n=n, M=length(dirs), dt=dt, krylov_depth=m,
        quadrature_rule="commuting-exact", quadrature_order=0,
        repeats=REPEATS_CROSS, quadrature_ms=qtime, core_ms=ctime,
        core_over_quadrature=ctime/qtime,
        relative_difference=norm(qvalue-cvalue)/max(norm(cvalue),eps()),
        component_fraction_below_1e6=mean(component_errors .<= 1e-6),
        component_fraction_below_1e8=mean(component_errors .<= 1e-8),
        component_error_p95=quantile(component_errors, 0.95),
        quadrature_extra_storage_bytes=qstorage,
        core_extra_storage_bytes=cstorage,
        core_over_quadrature_storage=cstorage/qstorage,
        common_basis_storage_bytes=0,
        core_over_quadrature_total_storage=cstorage/qstorage)
    @printf("%-12s commuting exact=%8.3f ms core=%8.3f ms ratio=%5.2f err=%.2e\n",
        name, qtime, ctime, ctime/qtime, row.relative_difference)
    row
end

rows = NamedTuple[]

function pauli_strings_cross(q)
    identity = sparse(ComplexF64[1 0; 0 1])
    paulis = (sparse(σx), sparse(σy), sparse(σz))
    [foldl(kron, [d == 0 ? identity : paulis[d]
                  for d in digits(code; base=4, pad=q)])
     for code in 1:(4^q-1)]
end

function lindblad_cross_case(q, count)
    strings = pauli_strings_cross(q)
    dirs = SparseMatrixCSC{ComplexF64,Int}[]
    for P in strings
        length(dirs) == count && break
        push!(dirs, hamiltonian_superop_sparse(P))
    end
    for P in strings
        length(dirs) == count && break
        push!(dirs, lindblad_superop_sparse(P))
    end
    site = (op, i) -> nsite_op_sparse(op, i, q)
    H = 0.3 * sum(site(σz, i) for i in 1:q)
    for i in 1:q-1
        H += 0.45 * (site(σx,i)*site(σx,i+1) + site(σy,i)*site(σy,i+1))
    end
    A = hamiltonian_superop_sparse(H)
    for i in 1:q
        A += 0.08*lindblad_superop_sparse(site(σz,i))
        A += 0.03*lindblad_superop_sparse(site(σm,i))
    end
    zero = ComplexF64[1,0]; plus = ComplexF64[1,1]/sqrt(2)
    ψ = foldl(kron, [isodd(i) ? plus : zero for i in 1:q])
    state = vec(ψ*ψ')
    O = sum((0.7/i)*site(σz,i) + (0.3/i)*site(σx,i) for i in 1:q)
    A, dirs, state, vec(Matrix(O))
end

# Stiff Brusselator snapshot used by the error-control audit.
let
    dirs, _ = build_library()
    θtrue = rates()
    θeval = θtrue .* [j == 1 ? 0.78 : j == 13 ? 1.22 : j == 23 ? 0.74 : j == 39 ? 1.28 : 1.0 for j in eachindex(θtrue)]
    Atrue = sum(θtrue[j] .* dirs[j] for j in eachindex(dirs))
    A = sum(θeval[j] .* dirs[j] for j in eachindex(dirs))
    state0 = zeros(size(A, 1)); state0[lin(3, 2)] = 1.0
    state = deep_action(Atrue, state0, 0.25, 420)
    target = deep_action(Atrue, state, 0.5, 420)
    prediction = deep_action(A, state, 0.5, 420)
    for nq in ORDERS_CROSS
        push!(rows, benchmark_case("Brusselator", A, dirs, state,
            prediction-target, 0.5, 60, nq))
    end
    for nq in filter(<=(12), ORDERS_CROSS)
        push!(rows, benchmark_lobatto_case("Brusselator", A, dirs, state,
            prediction-target, 0.5, 60, nq))
    end
end

# Five-qubit open-system case with the 1024-direction Pauli library.
let
    A, dirs, state, cotangent = lindblad_cross_case(5, 1024)
    for nq in ORDERS_CROSS
        push!(rows, benchmark_case("Lindblad", A, dirs, state, cotangent, 0.4, 64, nq))
    end
    for nq in filter(<=(12), ORDERS_CROSS)
        push!(rows, benchmark_lobatto_case("Lindblad", A, dirs, state, cotangent, 0.4, 64, nq))
    end
end

# Stiff nonnormal Fokker--Planck case from the accuracy audit.
let
    n, M, L, D = 80, 10, 2.0, 0.5
    xs = collect(range(-L, L; length=n))
    drift = [let k=j, L=L; x -> (x/L)^k end for j in 1:M]
    A0, dirs = fpe_library_1d(xs, drift; D=D)
    θ = zeros(M); θ[1] = -4.08; θ[3] = 3.96
    A = A0 + sum(θ[j] .* dirs[j] for j in 1:M)
    state = exp.(-0.5 .* xs.^2); state ./= sum(state)
    cotangent = sin.(range(0, 3π; length=n))
    for nq in ORDERS_CROSS
        push!(rows, benchmark_case("FPE", A, dirs, state, cotangent, 1.0, 60, nq))
    end
    for nq in ORDERS_CROSS
        push!(rows, benchmark_lobatto_case("FPE", A, dirs, state, cotangent, 1.0, 60, nq))
    end
end

# The linear exponential inside one KS integrating-factor step.
let
    n, length_domain = 64, 22.0
    dx = length_domain/n
    second = spdiagm(-1=>fill(1/dx^2,n-1), 0=>fill(-2/dx^2,n), 1=>fill(1/dx^2,n-1))
    second[1,n] = second[n,1] = 1/dx^2
    dirs = SparseMatrixCSC{Float64,Int}[sparse(-second), sparse(-(second*second))]
    A = 0.72 .* dirs[1] + 1.28 .* dirs[2]
    x = collect(range(0, length_domain; length=n+1))[1:end-1]
    state = 0.72 .* cos.(4π.*x./length_domain) + 0.22 .* sin.(6π.*x./length_domain)
    # A squared-error pullback has a state-shaped cotangent; use a nearby
    # smooth residual with overlapping Fourier support.
    cotangent = state .+ 0.05 .* cos.(10π.*x./length_domain)
    for nq in ORDERS_CROSS
        push!(rows, benchmark_case("KS", A, dirs, state, cotangent, 0.0125, 40, nq))
    end
    for nq in filter(<=(12), ORDERS_CROSS)
        push!(rows, benchmark_lobatto_case("KS", A, dirs, state, cotangent, 0.0125, 40, nq))
    end
    push!(rows, benchmark_commuting_case("KS", A, dirs, state, cotangent, 0.0125, 40))
end

mkpath(dirname(OUT_CROSS))
CSV.write(OUT_CROSS, DataFrame(rows))
println("Saved ", OUT_CROSS)
