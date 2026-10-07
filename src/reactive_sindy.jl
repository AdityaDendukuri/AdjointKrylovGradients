"""
    reactive_sindy_regression_stats(X, y)

Compute standardized Gram-matrix sufficient statistics for nonnegative
elastic-net regression. Standardization is internal; returned coefficients from
`nonnegative_elastic_net` are always mapped back to the physical units of `X`.
"""
function reactive_sindy_regression_stats(X, y)
    n = size(X, 1)
    gram_raw = Matrix(X' * X) / n
    cross_raw = vec(X' * y) / n
    scales = max.(sqrt.(max.(diag(gram_raw), 0.0)), 1e-12)
    gram = gram_raw ./ (scales * scales')
    cross = cross_raw ./ scales
    (; gram, cross, scales)
end

"""
    nonnegative_elastic_net(stats; alpha, l1_ratio, ...)

Solve

    min_{β≥0} 1/(2n)‖Xβ-y‖²
              + alpha*l1_ratio*‖γ‖₁
              + alpha*(1-l1_ratio)/2*‖γ‖²,

where `γ` denotes the standardized coefficient vector. Cyclic coordinate
descent operates on the Gram matrix and remains bounded for rank-deficient
reaction libraries containing opposite stoichiometric directions.
"""
function nonnegative_elastic_net(
    stats;
    alpha,
    l1_ratio,
    max_sweeps=20_000,
    tol=1e-10,
)
    0 <= l1_ratio <= 1 || throw(ArgumentError("l1_ratio must lie in [0, 1]"))
    alpha >= 0 || throw(ArgumentError("alpha must be nonnegative"))

    G, c, scales = stats.gram, stats.cross, stats.scales
    p = length(c)
    γ = zeros(p)
    ridge = alpha * (1 - l1_ratio)
    lasso = alpha * l1_ratio

    for _ in 1:max_sweeps
        max_change = 0.0
        for j in 1:p
            partial = c[j] - dot(view(G, j, :), γ) + G[j, j] * γ[j]
            denominator = G[j, j] + ridge
            γ_new = denominator > 0 ?
                    max(0.0, (partial - lasso) / denominator) : 0.0
            max_change = max(max_change, abs(γ_new - γ[j]))
            γ[j] = γ_new
        end
        max_change < tol && break
    end
    γ ./ scales
end
