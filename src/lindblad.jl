using LinearAlgebra, SparseArrays

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const σp = ComplexF64[0 1; 0 0]
const σm = ComplexF64[0 0; 1 0]

# Build n-qubit operator: op on site k, identity elsewhere
function nsite_op(op::AbstractMatrix, site::Int, N::Int)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    foldl(kron, [k == site ? op : I2 for k in 1:N])
end

# Sparse counterpart used by large-system scaling experiments. Keeping this
# separate from `nsite_op` preserves the original small dense experiments.
function nsite_op_sparse(op::AbstractMatrix, site::Int, N::Int)
    I2 = spdiagm(0 => ones(ComplexF64, 2))
    op_sparse = sparse(ComplexF64.(op))
    foldl(kron, [k == site ? op_sparse : I2 for k in 1:N])
end

# Hamiltonian superoperator: L_H = -i(I⊗H - H*⊗I)
# Maps vec(ρ) → vec(-i[H,ρ])
function hamiltonian_superop(H::AbstractMatrix)
    n  = size(H, 1)
    In = Matrix{ComplexF64}(I, n, n)
    -im * (kron(In, Matrix(H)) - kron(conj(Matrix(H)), In))
end

function hamiltonian_superop_sparse(H::AbstractMatrix)
    Hs = sparse(ComplexF64.(H))
    n = size(Hs, 1)
    In = spdiagm(0 => ones(ComplexF64, n))
    sparse(-im * (kron(In, Hs) - kron(conj(Hs), In)))
end

# Lindblad dissipator superoperator: D[L](ρ) = LρL† - ½{L†L, ρ}
# Maps vec(ρ) → vec(D[L](ρ))
function lindblad_superop(L::AbstractMatrix)
    n   = size(L, 1)
    In  = Matrix{ComplexF64}(I, n, n)
    LdL = L' * L
    kron(conj(Matrix(L)), Matrix(L)) -
        0.5 * kron(In, LdL) -
        0.5 * kron(transpose(LdL), In)
end

function lindblad_superop_sparse(L::AbstractMatrix)
    Ls = sparse(ComplexF64.(L))
    n = size(Ls, 1)
    In = spdiagm(0 => ones(ComplexF64, n))
    LdL = Ls' * Ls
    sparse(
        kron(conj(Ls), Ls) -
        0.5 * kron(In, LdL) -
        0.5 * kron(transpose(LdL), In)
    )
end

# Build a library of superoperators from Hamiltonian basis matrices and
# Lindblad jump operators.  Returns (A₀, dirs) where A₀ = 0 and dirs[j] is
# the j-th superoperator.  Hamiltonian entries first, then dissipators.
function lindblad_library(H_basis::Vector, L_basis::Vector)
    A₀   = zeros(ComplexF64, size(hamiltonian_superop(H_basis[1]))...)
    dirs = AbstractMatrix{ComplexF64}[]
    for H in H_basis
        push!(dirs, hamiltonian_superop(H))
    end
    for L in L_basis
        push!(dirs, lindblad_superop(L))
    end
    return A₀, dirs
end
