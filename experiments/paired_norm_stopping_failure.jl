"""
Demonstrate delayed convergence of a successive-depth Fréchet--Krylov stopping
rule and the paired-norm error bound on a nonnormal pure-birth CTMC.

The observable is the probability of state `TARGET` after a finite interval.
At shallow depths, consecutive projected gradients agree at zero even though
the exact sensitivity is nonzero.  The contractive 1/infinity-norm bound
remains an upper bound.

Run from the repository root:

    julia --project=experiments experiments/paired_norm_stopping_failure.jl
"""

using LinearAlgebra, SparseArrays, Printf
using CSV, DataFrames

const N = parse(Int, get(ENV, "CERT_N", "80"))
const TARGET = parse(Int, get(ENV, "CERT_TARGET", "15"))
const RATE = parse(Float64, get(ENV, "CERT_RATE", "1.0"))
const FINAL_TIME = parse(Float64, get(ENV, "CERT_TIME", "20.0"))
const MAX_DEPTH = parse(Int, get(ENV, "CERT_MAX_DEPTH", "50"))
const OUTPUT = get(
    ENV,
    "CERT_OUTPUT",
    joinpath(@__DIR__, "..", "paper", "figures", "paired_norm_stopping_failure.csv"),
)

function pure_birth_direction(n)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for column in 1:(n - 1)
        push!(rows, column, column + 1)
        push!(cols, column, column)
        push!(vals, -1.0, 1.0)
    end
    sparse(rows, cols, vals, n, n)
end

function arnoldi_with_residual(A, start, depth; tolerance=1e-14)
    n = length(start)
    requested = min(depth, n - 1)
    V = zeros(eltype(start), n, requested + 1)
    H = zeros(eltype(start), requested + 1, requested)
    beta = norm(start)
    beta > tolerance || error("Arnoldi start must be nonzero")
    V[:, 1] .= start ./ beta
    used = requested
    for column in 1:requested
        work = A * view(V, :, column)
        for row in 1:column
            H[row, column] = dot(view(V, :, row), work)
            work .-= H[row, column] .* view(V, :, row)
        end
        H[column + 1, column] = norm(work)
        if H[column + 1, column] <= tolerance
            used = column
            break
        end
        V[:, column + 1] .= work ./ H[column + 1, column]
    end
    (
        V=V[:, 1:used],
        H=H[1:used, 1:used],
        beta=beta,
        residual_coefficient=H[used + 1, used],
        residual_vector=V[:, used + 1],
        depth=used,
    )
end

function projected_core(forward, adjoint, time)
    mf = forward.depth
    ma = adjoint.depth
    block = zeros(Float64, ma + mf, ma + mf)
    block[1:ma, 1:ma] .= adjoint.H
    block[(ma + 1):end, (ma + 1):end] .= forward.H'
    block[1, ma + 1] = adjoint.beta * forward.beta
    exp(time .* block)[1:ma, (ma + 1):end]
end

function projected_matrix(forward, adjoint, time)
    adjoint.V * projected_core(forward, adjoint, time) * forward.V'
end

function integrated_residual(arnoldi, time, vector_norm)
    m = arnoldi.depth
    e1 = zeros(m)
    e1[1] = 1.0
    integral = (time .* arnoldi.H) \ ((exp(time .* arnoldi.H) - I) * e1)
    abs(time * arnoldi.beta * arnoldi.residual_coefficient * integral[end]) *
        vector_norm(arnoldi.residual_vector)
end

function main()
2 <= TARGET < N || error("CERT_TARGET must lie in 2:(CERT_N-1)")
2 <= MAX_DEPTH < N || error("CERT_MAX_DEPTH must lie in 2:(CERT_N-1)")

direction = pure_birth_direction(N)
generator = RATE .* direction
initial = zeros(N)
initial[1] = 1.0
terminal = zeros(N)
terminal[TARGET + 1] = 1.0

adjoint_block = FINAL_TIME .* Matrix(generator')
forcing = FINAL_TIME .* (terminal * initial')
exact_matrix = exp([
    adjoint_block forcing
    zeros(N, N) adjoint_block
])[1:N, (N + 1):(2N)]
exact_gradient = real(dot(exact_matrix, Matrix(direction)))

rows = NamedTuple[]
previous_gradient = NaN
previous_matrix = nothing
for depth in 2:MAX_DEPTH
    forward = arnoldi_with_residual(generator, initial, depth)
    adjoint = arnoldi_with_residual(generator', terminal, depth)
    approximation = projected_matrix(forward, adjoint, FINAL_TIME)
    gradient = real(dot(approximation, Matrix(direction)))
    gradient_error = abs(gradient - exact_gradient)
    successive_gradient = isfinite(previous_gradient) ?
        abs(gradient - previous_gradient) : NaN
    matrix_error = norm(approximation - exact_matrix)
    successive_matrix = previous_matrix === nothing ? NaN :
        norm(approximation - previous_matrix)

    eta_forward = integrated_residual(forward, FINAL_TIME, x -> norm(x, 1))
    eta_adjoint = integrated_residual(adjoint, FINAL_TIME, x -> norm(x, Inf))
    direction_bound = opnorm(Matrix(direction), 1)
    error_bound = FINAL_TIME * direction_bound * (
        norm(initial, 1) * eta_adjoint +
        norm(terminal, Inf) * eta_forward +
        eta_forward * eta_adjoint
    )

    push!(rows, (
        depth=depth,
        exact_gradient=exact_gradient,
        projected_gradient=gradient,
        gradient_error=gradient_error,
        successive_gradient_estimate=successive_gradient,
        paired_norm_error_bound=error_bound,
        matrix_error_frobenius=matrix_error,
        successive_matrix_estimate=successive_matrix,
        forward_residual_integral=eta_forward,
        adjoint_residual_integral=eta_adjoint,
    ))
    @printf(
        "m=%2d  error=%9.2e  successive=%9.2e  bound=%9.2e\n",
        depth,
        gradient_error,
        successive_gradient,
        error_bound,
    )
    previous_gradient = gradient
    previous_matrix = approximation
end

data = DataFrame(rows)
@assert all(
    data.paired_norm_error_bound .+
    100eps(Float64) .>= data.gradient_error
)
stop_index = findfirst(
    isfinite.(data.successive_matrix_estimate) .&
    (data.successive_matrix_estimate .<= 1e-4)
)
@assert stop_index !== nothing
@assert data.depth[stop_index] == 3
@assert data.matrix_error_frobenius[stop_index] >= 1.0

mkpath(dirname(OUTPUT))
CSV.write(OUTPUT, data)
@printf("exact gradient: %.12e\n", exact_gradient)
println("saved ", OUTPUT)
end

main()
