"""Nested Clenshaw--Curtis versus Gauss--Legendre for the stiff FPE gradient.

This experiment isolates quadrature error by evaluating the forward and
adjoint paths with dense exponentials.  Clenshaw--Curtis node counts follow
3, 5, 9, 17, 33, so every refinement retains the preceding nodes.
"""

using LinearAlgebra, SparseArrays, Statistics, Printf
using FastGaussQuadrature: gausslegendre
using CSV, DataFrames

include("../src/fpe.jl")

const NODE_COUNTS = (3, 5, 9, 17, 33, 65)
const OUTPUT = get(ENV, "CC_FPE_OUTPUT",
    joinpath(@__DIR__, "..", "benchmark_results", "nested_clenshaw_curtis_fpe.csv"))

gauss01(n) = let (x, w) = gausslegendre(n); ((x .+ 1) ./ 2, w ./ 2) end

"""Clenshaw--Curtis nodes and weights on [0,1]."""
function clenshaw_curtis(node_count)
    node_count >= 2 || throw(ArgumentError("at least two nodes are required"))
    degree = node_count - 1
    theta = pi .* collect(0:degree) ./ degree
    nodes = (1 .+ cos.(theta)) ./ 2
    weights = zeros(Float64, node_count)
    if degree == 1
        weights .= 0.5
        return nodes, weights
    end
    interior = 2:degree
    values = ones(Float64, degree - 1)
    if iseven(degree)
        weights[1] = weights[end] = 1 / (2 * (degree^2 - 1))
        for k in 1:(degree ÷ 2 - 1)
            values .-= 2 .* cos.(2k .* theta[interior]) ./ (4k^2 - 1)
        end
        values .-= cos.(degree .* theta[interior]) ./ (degree^2 - 1)
    else
        weights[1] = weights[end] = 1 / (2 * degree^2)
        for k in 1:((degree - 1) ÷ 2)
            values .-= 2 .* cos.(2k .* theta[interior]) ./ (4k^2 - 1)
        end
    end
    weights[interior] .= values ./ degree
    nodes, weights
end

function gradient_rule(A, directions, state, cotangent, dt, nodes, weights)
    gradient = zeros(length(directions))
    for q in eachindex(nodes)
        forward = exp(nodes[q] * dt * A) * state
        adjoint = exp((1 - nodes[q]) * dt * A') * cotangent
        for j in eachindex(directions)
            gradient[j] += dt * weights[q] * dot(adjoint, directions[j] * forward)
        end
    end
    gradient
end

function dense_reference(A, directions, state, cotangent, dt)
    n = length(state)
    block = zeros(Float64, 2n, 2n)
    block[1:n, 1:n] .= dt .* A'
    block[n+1:2n, n+1:2n] .= dt .* A'
    block[1:n, n+1:2n] .= dt .* (cotangent * state')
    frechet = exp(block)[1:n, n+1:2n]
    [dot(frechet, Matrix(direction)) for direction in directions]
end

n, count, length_domain, diffusion = 80, 10, 2.0, 0.5
xs = collect(range(-length_domain, length_domain; length=n))
drifts = [let k=j, L=length_domain; x -> (x/L)^k end for j in 1:count]
A0, directions = fpe_library_1d(xs, drifts; D=diffusion)
theta_true = zeros(count); theta_true[1] = -4.0; theta_true[3] = 4.0
theta_eval = theta_true + 0.08 .* collect(range(-1, 1; length=count))
Atrue = Matrix(A0 + sum(theta_true[j] .* directions[j] for j in 1:count))
Aeval = Matrix(A0 + sum(theta_eval[j] .* directions[j] for j in 1:count))
state = exp.(-0.5 .* xs.^2); state ./= sum(state)
dt = 1.0
target = exp(dt .* Atrue) * state
prediction = exp(dt .* Aeval) * state
cotangent = prediction - target
reference = dense_reference(Aeval, directions, state, cotangent, dt)
scale = max(norm(reference), eps())

rows = NamedTuple[]
values = Dict{Tuple{String,Int},Vector{Float64}}()
times_ms = Dict{Tuple{String,Int},Float64}()
for node_count in NODE_COUNTS
    for (rule, constructor) in (("Gauss-Legendre", gauss01),
                                ("Clenshaw-Curtis", clenshaw_curtis))
        nodes, weights = constructor(node_count)
        evaluate = () -> gradient_rule(
            Aeval, directions, state, cotangent, dt, nodes, weights,
        )
        value = evaluate()
        elapsed = median((@elapsed evaluate()) for _ in 1:5)
        error = norm(value - reference) / scale
        values[(rule, node_count)] = value
        times_ms[(rule, node_count)] = 1000elapsed
        push!(rows, (rule=rule, nodes=node_count, relative_error=error,
            elapsed_ms=1000elapsed))
        @printf("%-17s nodes=%2d error=%9.2e time=%8.2f ms\n",
            rule, node_count, error, 1000elapsed)
    end
end

for tolerance in (1e-6, 1e-8, 1e-10)
    println("\nTolerance ", tolerance)
    for rule in ("Gauss-Legendre", "Clenshaw-Curtis")
        selected = filter(row -> row.rule == rule && row.relative_error <= tolerance, rows)
        if isempty(selected)
            println("  ", rule, ": not reached")
            continue
        end
        final_nodes = first(selected).nodes
        if rule == "Clenshaw-Curtis"
            adaptive_nodes = final_nodes
        else
            adaptive_nodes = sum(n for n in NODE_COUNTS if n <= final_nodes)
        end
        println("  ", rule, ": final nodes=", final_nodes,
            ", adaptive node evaluations=", adaptive_nodes)
    end
end

println("\nSuccessive-rule stopping")
for tolerance in (1e-4, 1e-6, 1e-8)
    println("  tolerance ", tolerance)
    for rule in ("Gauss-Legendre", "Clenshaw-Curtis")
        previous = nothing
        stopped = false
        for node_count in NODE_COUNTS
            value = values[(rule, node_count)]
            if previous !== nothing
                indicator = norm(value - previous) / max(norm(value), eps())
                if indicator <= tolerance
                    true_error = norm(value - reference) / scale
                    work_nodes = rule == "Clenshaw-Curtis" ? node_count :
                        sum(n for n in NODE_COUNTS if n <= node_count)
                    estimated_ms = rule == "Clenshaw-Curtis" ?
                        times_ms[(rule, node_count)] :
                        sum(times_ms[(rule, n)] for n in NODE_COUNTS if n <= node_count)
                    @printf("    %-17s stop=%2d indicator=%8.2e true=%8.2e nodes=%3d time=%7.2f ms\n",
                        rule, node_count, indicator, true_error, work_nodes,
                        estimated_ms)
                    stopped = true
                    break
                end
            end
            previous = value
        end
        stopped || println("    ", rule, ": not stopped")
    end
end

mkpath(dirname(OUTPUT))
CSV.write(OUTPUT, DataFrame(rows))
println("Saved ", OUTPUT)
