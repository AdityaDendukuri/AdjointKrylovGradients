"""Measure redundant versus shared Arnoldi construction in one reverse pass."""

using AdjointKrylovGradients
using AdjointKrylovGradients: arnoldi_basis!, _eval_krylov_batch,
    _quadrature_vjp_from_bases
using ExponentialUtilities: expv
using LinearAlgebra, Statistics, Printf

include("brusselator_error_control.jl")

const REPEATS_REUSE = parse(Int, get(ENV, "REUSE_REPEATS", "11"))

function median_ms(f)
    f()
    median((@elapsed f())*1000 for _ in 1:REPEATS_REUSE)
end

function arnoldi_cgs2!(ws, A, v; m)
    n = length(v); m = min(m, n, size(ws.V,2))
    V, H, w = ws.V, ws.H, ws.w
    fill!(V, 0); fill!(H, 0)
    beta = norm(v); V[:,1] .= v ./ beta
    correction = zeros(eltype(v), m)
    used = m
    for j in 1:m
        mul!(w, A, view(V,:,j))
        basis = view(V,:,1:j)
        coefficients = view(H,1:j,j)
        mul!(coefficients, basis', w)
        mul!(w, basis, coefficients, -1, 1)
        correction_j = view(correction,1:j)
        mul!(correction_j, basis', w)
        mul!(w, basis, correction_j, -1, 1)
        coefficients .+= correction_j
        residual = norm(w)
        if j < m
            if residual < 1e-12; used=j; break; end
            H[j+1,j] = residual
            V[:,j+1] .= w ./ residual
        end
    end
    (V=view(V,:,1:used), H=view(H,1:used,1:used), beta=beta, m=used)
end

function arnoldi_cgs1!(ws, A, v; m)
    n = length(v); m = min(m, n, size(ws.V,2))
    V, H, w = ws.V, ws.H, ws.w
    fill!(V, 0); fill!(H, 0)
    beta = norm(v); V[:,1] .= v ./ beta
    used = m
    for j in 1:m
        mul!(w, A, view(V,:,j))
        basis = view(V,:,1:j)
        coefficients = view(H,1:j,j)
        mul!(coefficients, basis', w)
        mul!(w, basis, coefficients, -1, 1)
        residual = norm(w)
        if j < m
            if residual < 1e-12; used=j; break; end
            H[j+1,j] = residual
            V[:,j+1] .= w ./ residual
        end
    end
    (V=view(V,:,1:used), H=view(H,1:used,1:used), beta=beta, m=used)
end

function arnoldi_iom!(ws, A, v; m, window)
    n = length(v); m = min(m, n, size(ws.V,2))
    V, H, w = ws.V, ws.H, ws.w
    fill!(V, 0); fill!(H, 0)
    beta = norm(v); V[:,1] .= v ./ beta
    used = m
    for j in 1:m
        mul!(w, A, view(V,:,j))
        first = max(1, j-window+1)
        for i in first:j
            coefficient = dot(view(V,:,i), w)
            H[i,j] = coefficient
            @inbounds @simd for l in 1:n
                w[l] -= coefficient*V[l,i]
            end
        end
        residual = norm(w)
        if j < m
            if residual < 1e-12; used=j; break; end
            H[j+1,j] = residual
            V[:,j+1] .= w ./ residual
        end
    end
    (V=view(V,:,1:used), H=view(H,1:used,1:used), beta=beta, m=used)
end

function legacy_reverse(A, dirs, state, cotangent, dt, m, nq)
    prediction = expv(dt, A, state; m=m)
    gradient = krylov_frechet_vjp(A, dirs, state, dt, cotangent;
        m_krylov=m, n_quad=nq)
    state_vjp = expv(dt, A', cotangent; m=m)
    prediction, gradient, state_vjp
end

function shared_reverse(A, dirs, state, cotangent, dt, m, nq, fw, aw, cache)
    fb = arnoldi_basis!(fw, A, state; m=m)
    prediction = _eval_krylov_batch(fb, [dt], fb.beta)[:,1]
    ab = arnoldi_basis!(aw, A', cotangent; m=m)
    gradient = _quadrature_vjp_from_bases(
        fb, ab, dirs, dt, nq; direction_cache=cache,
    )
    state_vjp = _eval_krylov_batch(ab, [dt], ab.beta)[:,1]
    prediction, gradient, state_vjp
end

dirs, _ = build_library()
θtrue = rates()
θeval = θtrue .* [j == 1 ? 0.78 : j == 13 ? 1.22 : j == 23 ? 0.74 : j == 39 ? 1.28 : 1.0 for j in eachindex(θtrue)]
Atrue = sum(θtrue[j].*dirs[j] for j in eachindex(dirs))
A = sum(θeval[j].*dirs[j] for j in eachindex(dirs))
state0 = zeros(size(A,1)); state0[lin(3,2)] = 1.0
state = deep_action(Atrue, state0, 0.25, 420)
target = deep_action(Atrue, state, 0.5, 420)
prediction = deep_action(A, state, 0.5, 420)
cotangent = prediction-target
m, nq, dt = 60, 5, 0.5
fw = ArnoldiWorkspace(Float64, length(state), m)
aw = ArnoldiWorkspace(Float64, length(state), m)
cache = SparseDirectionCache(dirs)

old_value = legacy_reverse(A, dirs, state, cotangent, dt, m, nq)
new_value = shared_reverse(A, dirs, state, cotangent, dt, m, nq, fw, aw, cache)
agreement = maximum(norm(old_value[k]-new_value[k])/max(norm(old_value[k]),eps()) for k in 1:3)
old_ms = median_ms(() -> legacy_reverse(A, dirs, state, cotangent, dt, m, nq))
new_ms = median_ms(() -> shared_reverse(A, dirs, state, cotangent, dt, m, nq, fw, aw, cache))
@printf("Brusselator reverse pass: legacy %.3f ms, shared %.3f ms, speedup %.2fx, relative difference %.3e\n",
    old_ms, new_ms, old_ms/new_ms, agreement)

forward_ms = median_ms(() -> arnoldi_basis!(fw, A, state; m=m))
adjoint_ms = median_ms(() -> arnoldi_basis!(aw, A', cotangent; m=m))
fb = arnoldi_basis!(fw, A, state; m=m)
ab = arnoldi_basis!(aw, A', cotangent; m=m)
parameter_ms = median_ms(() -> _quadrature_vjp_from_bases(
    fb, ab, dirs, dt, nq; direction_cache=cache,
))
endpoint_ms = median_ms(() -> begin
    _eval_krylov_batch(fb, [dt], fb.beta)
    _eval_krylov_batch(ab, [dt], ab.beta)
end)
spmv_buffer = zeros(length(state))
spmv_ms = median_ms(() -> begin
    for j in 1:fb.m
        mul!(spmv_buffer, A, view(fb.V,:,j))
    end
end)
@printf("  forward Arnoldi %.3f ms; adjoint Arnoldi %.3f ms; parameter evaluation %.3f ms; endpoint evaluations %.3f ms\n",
    forward_ms, adjoint_ms, parameter_ms, endpoint_ms)
@printf("  %d forward sparse matvecs alone %.3f ms; forward orthogonalization/norm overhead approximately %.3f ms\n",
    fb.m, spmv_ms, max(forward_ms-spmv_ms, 0.0))

cgs_workspace = ArnoldiWorkspace(Float64, length(state), m)
cgs_ms = median_ms(() -> arnoldi_cgs2!(cgs_workspace, A, state; m=m))
cgs = arnoldi_cgs2!(cgs_workspace, A, state; m=m)
mgs = arnoldi_basis!(fw, A, state; m=m)
cgs_endpoint = _eval_krylov_batch(cgs, [dt], cgs.beta)[:,1]
mgs_endpoint = _eval_krylov_batch(mgs, [dt], mgs.beta)[:,1]
cgs_orthogonality = norm(Matrix(cgs.V)'*Matrix(cgs.V)-I)
mgs_orthogonality = norm(Matrix(mgs.V)'*Matrix(mgs.V)-I)
@printf("  MGS %.3f ms versus BLAS CGS2 %.3f ms (%.2fx); action difference %.3e; orthogonality MGS %.3e, CGS2 %.3e\n",
    forward_ms, cgs_ms, forward_ms/cgs_ms,
    norm(cgs_endpoint-mgs_endpoint)/norm(mgs_endpoint),
    mgs_orthogonality, cgs_orthogonality)

cgs1_workspace = ArnoldiWorkspace(Float64, length(state), m)
cgs1_ms = median_ms(() -> arnoldi_cgs1!(cgs1_workspace, A, state; m=m))
cgs1 = arnoldi_cgs1!(cgs1_workspace, A, state; m=m)
cgs1_endpoint = _eval_krylov_batch(cgs1, [dt], cgs1.beta)[:,1]
cgs1_orthogonality = norm(Matrix(cgs1.V)'*Matrix(cgs1.V)-I)
@printf("  BLAS CGS1 %.3f ms (%.2fx versus MGS); action difference %.3e; orthogonality %.3e\n",
    cgs1_ms, forward_ms/cgs1_ms,
    norm(cgs1_endpoint-mgs_endpoint)/norm(mgs_endpoint), cgs1_orthogonality)

for window in (2,4,8,16,32)
    iom_workspace = ArnoldiWorkspace(Float64, length(state), m)
    iom_ms = median_ms(() -> arnoldi_iom!(iom_workspace, A, state; m=m, window=window))
    iom = arnoldi_iom!(iom_workspace, A, state; m=m, window=window)
    iom_endpoint = _eval_krylov_batch(iom, [dt], iom.beta)[:,1]
    iom_orthogonality = norm(Matrix(iom.V)'*Matrix(iom.V)-I)
    @printf("  IOM(%2d) %.3f ms (%.2fx); action difference %.3e; orthogonality %.3e\n",
        window, iom_ms, forward_ms/iom_ms,
        norm(iom_endpoint-mgs_endpoint)/norm(mgs_endpoint), iom_orthogonality)
end
