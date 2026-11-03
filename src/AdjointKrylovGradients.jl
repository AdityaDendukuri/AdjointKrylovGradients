"""Adjoint Krylov gradients for affine matrix-exponential actions."""
module AdjointKrylovGradients

using Printf
using LinearAlgebra, SparseArrays, Statistics, Random
using ExponentialUtilities, ChainRulesCore, FastGaussQuadrature

include("shared_krylov.jl")
include("sparse_direct_rational.jl")
include("sparse_generator.jl")
include("lindblad.jl")
include("fpe.jl")
include("reactive_sindy.jl")

# Finite-state CME model and sparse inference pipeline (paper appendix)
include("core.jl")
include("gradient.jl")
include("gradient_adjoint.jl")
include("fpe2d.jl")
include("state_space.jl")
include("empirical.jl")
include("stls.jl")
include("mass_action.jl")

export SparseDirectionCache, ArnoldiWorkspace
export arnoldi_basis, arnoldi_basis!, gauss_legendre_01
export krylov_frechet_vjp, compressed_core_frechet_vjp
export sparse_direct_exponential, sparse_direct_exponential_vjp
export SemigroupLayer, shared_gradient, shared_gradient_fisher, nll_gradient
export SparseGeneratorResult, assemble_generator, sparse_generator_stls
export σx, σy, σz, σp, σm, nsite_op, nsite_op_sparse
export hamiltonian_superop, hamiltonian_superop_sparse
export lindblad_superop, lindblad_superop_sparse, lindblad_library
export fpe_library_1d, fpe_library_2d
export reactive_sindy_regression_stats, nonnegative_elastic_net

export quad_features, feature_dimension, feature_names
export cubic_features, cubic_feature_dimension, cubic_feature_names
export build_generator, compute_residuals, BlockOp, frechet_action
export gradient_adjoint_krylov!, gradient_adjoint_krylov, gradient_and_fisher!
export gradient_approx!, compute_weights
export build_derivatives, flatten_params, unflatten_params, gradient!, objective
export build_state_space, build_distributions
export count_transitions, empirical_initialization
export optimize_stls, STLSResult
export mass_action_feature_index, generate_candidate_reactions
export mass_action_initialization, extract_mass_action_rates, reaction_description

end
