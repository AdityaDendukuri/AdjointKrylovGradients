# Adjoint Krylov Gradients

Julia implementation and reproducibility scripts for adjoint Krylov gradients
of linearly parameterized matrix-exponential actions.

## Layout

- `src/shared_krylov.jl`: forward/adjoint Krylov spaces, Fréchet VJPs, and the
  differentiable semigroup layer
- `src/sparse_generator.jl`: sparse optimization over affine operator libraries
- `src/lindblad.jl` and `src/fpe.jl`: structured operators used in the tests
- `experiments/`: numerical studies reported in the paper
- `test/shared_krylov_tests.jl`: gradient and pullback regression tests

The main drivers are:

```text
gradient_accuracy.jl                  gradient convergence
benchmark_sisc_library_accuracy.jl   library-size scaling
brusselator_sparse_discovery.jl      CME identification
kuramoto_sivashinsky_composable.jl   semilinear PDE identification
lindblad_sparse_identification.jl    open-quantum-system identification
paired_norm_stopping_failure.jl      rigorous error bound versus successive-depth stopping
```

## Julia environment

From the repository root, use

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The reproduction scripts have a separate environment:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/gradient_accuracy.jl
```

The manuscript sources are maintained separately and are not included in this
repository. Generated results, figures, LaTeX files, and Julia package
artifacts are ignored.
