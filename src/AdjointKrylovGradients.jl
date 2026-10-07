"""Adjoint Krylov gradients for affine matrix-exponential actions."""
module AdjointKrylovGradients

using Printf

include("shared_krylov.jl")
include("sparse_generator.jl")
include("lindblad.jl")
include("fpe.jl")
include("reactive_sindy.jl")

export SparseDirectionCache, ArnoldiWorkspace
export arnoldi_basis, arnoldi_basis!, gauss_legendre_01
export krylov_frechet_vjp, compressed_core_frechet_vjp
export SemigroupLayer, shared_gradient, shared_gradient_fisher, nll_gradient
export SparseGeneratorResult, assemble_generator, sparse_generator_stls
export σx, σy, σz, σp, σm, nsite_op, nsite_op_sparse
export hamiltonian_superop, hamiltonian_superop_sparse
export lindblad_superop, lindblad_superop_sparse, lindblad_library
export fpe_library_1d
export reactive_sindy_regression_stats, nonnegative_elastic_net

end
