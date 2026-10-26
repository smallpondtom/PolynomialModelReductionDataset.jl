```@meta
CurrentModule = PolynomialModelReductionDataset
```

# Fast Solvers

Implicit and semi-implicit time integrators (backward Euler, Crank-Nicolson, SICN, CNAB) all solve a linear system of the form

```math
\left(\mathbf{I} - \alpha\mathbf{A}\right)\mathbf{u}(t_{k+1}) = \mathbf{r}_k
```

at every time step, where ``\mathbf{A}`` is the linear operator of the discretized PDE and ``\alpha`` depends on the scheme:

| Scheme                                  | ``\alpha``          |
|-----------------------------------------|---------------------|
| Backward Euler (`:BE`)                  | ``\Delta t``        |
| Crank-Nicolson / SICN / CNAB (`:CN`)    | ``\Delta t / 2``    |

Crank-Nicolson-type schemes also need the forward product ``\left(\mathbf{I} + \alpha\mathbf{A}\right)\mathbf{u}(t_k)``. Because ``\mathbf{A}`` and ``\alpha`` are fixed during a simulation, the `FastSolvers` submodule pre-processes ``\mathbf{A}`` once (eigendecomposition, FFT diagonalization, or LU factorization) and bakes ``\alpha`` into a cached factor. Each time step then reduces to a few structured multiplies with no allocations.

## Usage with the models

Every model provides a `build_fast_solver` method that picks the right solver for its boundary conditions, and an `integrate_model_fast` method that uses it. The `scheme` keyword sets ``\alpha`` and must match the integrator you use.

```julia
using PolynomialModelReductionDataset
const pomoreda = PolynomialModelReductionDataset

heat1d = pomoreda.Heat1DModel(
    spatial_domain=(0.0, 1.0), time_domain=(0.0, 1.0),
    Δx=1/2^7, Δt=1e-3, diffusion_coeffs=0.1
)
μ = heat1d.diffusion_coeffs
A, B = heat1d.finite_diff_model(heat1d, μ; same_on_both_ends=true)
Ubc = ones(heat1d.time_dim)           # boundary input

solver = pomoreda.build_fast_solver(heat1d, μ; scheme=:CN, same_on_both_ends=true)   # one-time precompute
U = pomoreda.integrate_model_fast(heat1d, solver, heat1d.tspan, heat1d.IC, Ubc;
                                  control_matrix=B, scheme=:CN)
```

See each model's page for the keyword arguments its `integrate_model_fast` method accepts.

## Using the solvers directly

The solvers can also be built from an operator matrix and used in your own time-stepping loop, for example to integrate a reduced-order model.

```julia
solver = FastSymTridiagSolver(A, Δt/2)   # α = Δt/2 for Crank-Nicolson

linsolve!(u_new, solver, rhs)            # solve (I - α A) u_new = rhs
mulIpA!(out, solver, u_old)              # out = (I + α A) u_old
update_timestep!(solver, Δt_new/2)       # change α without rebuilding
```

| Solver                                          | Structure of ``\mathbf{A}``                                          | Typical use                                  |
|-------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------|
| [`FastSymTridiagSolver`](@ref)                  | 1D symmetric tridiagonal                                             | 1D Dirichlet, Neumann, or mixed boundaries   |
| [`FastCirculant1DSolver`](@ref)                 | 1D circulant                                                         | 1D periodic boundaries                       |
| [`FastKronSumSolver`](@ref)                     | ``(\mathbf{A}_y \otimes \mathbf{I}) + (\mathbf{I} \otimes \mathbf{A}_x)``, symmetric tridiagonal factors | 2D Dirichlet boundaries |
| [`FastFFT2DSolver`](@ref)                       | Kronecker sum of two circulants                                      | 2D periodic boundaries                       |
| [`FastDenseSolver`](@ref)                       | dense, unstructured                                                  | reduced-order models                         |
| [`FactorizedSolver`](@ref)                      | any (sparse) matrix                                                  | fallback when no structure applies           |

## API

```@docs
PolynomialModelReductionDataset.FastSolvers
PolynomialModelReductionDataset.FastSolvers.AbstractFastSolver
PolynomialModelReductionDataset.FastSolvers.FastSymTridiagSolver
PolynomialModelReductionDataset.FastSolvers.FastCirculant1DSolver
PolynomialModelReductionDataset.FastSolvers.FastKronSumSolver
PolynomialModelReductionDataset.FastSolvers.FastFFT2DSolver
PolynomialModelReductionDataset.FastSolvers.FastDenseSolver
PolynomialModelReductionDataset.FastSolvers.FactorizedSolver
PolynomialModelReductionDataset.FastSolvers.linsolve!
PolynomialModelReductionDataset.FastSolvers.mulIpA!
PolynomialModelReductionDataset.FastSolvers.update_timestep!
PolynomialModelReductionDataset.FastSolvers.backward_euler_solve!
```
