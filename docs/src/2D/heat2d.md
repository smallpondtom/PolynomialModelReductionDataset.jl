# 2D Heat Equation

## Overview 

This is the 2D version of [1D heat equation](../1D/heat1d.md). It is defined by

```math
u_t = \mu (u_{xx} + u_{yy})
```
where $x,y\in[0,L]$ is the temperature and $\mu$ is the thermal diffusivity parameter.  

## Finite Difference Model

The finite difference approach is similar to the 1D version but with additional complications due to the addition of an extra dimension. If the 2D domain is spatially discretized using $N$ and $M$ grid points for the $x$ and $y$ directions, respectively, then the toeplitz matrices corresponding to each $x$ and $y$ directions are identical to that of the 1D case which is defined by

```math
\mathbf{A}_x\in\mathbb{R}^{N\times N} \qquad \text{and} \qquad \mathbf{A}_y\in\mathbb{R}^{M\times M}.
```

However, to construct the matrix for the overall system we utilize the Kronecker product. Define the state vector $\mathbf{z}$ which flattens the 2D grid into a vector then the linear matrix becomes

```math
\mathbf{A} = \mathbf{A}_y \otimes \mathbf{I}_{N} + \mathbf{I}_M \otimes \mathbf{A}_x
```

and the $\mathbf{B}$ matrix will be constructed such that they add the inputs to the appropriate indices of the flattened state vector $\mathbf z$.

Thus, we arrive at a $N$-dimensional linear time-invariant (LTI) system:

```math
\dot{\mathbf{u}}(t) = \mathbf{A}\mathbf{u}(t) + \mathbf{B}\mathbf{w}(t)
```

We then consider the numerical integration scheme. For our numerical integration we can consider three approaches
- Forward Euler
- Backward Euler
- Crank-Nicolson

Refer to [1D heat equation](../1D/heat1d.md) for details on the numerical integration schemes.

## Example

```@example Heat2D
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: Heat2DModel
using UniqueKronecker: invec

# Setup
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,2), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    diffusion_coeffs=0.1, BC=(:dirichlet, :dirichlet)
)
xgrid0 = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
ux0 = sin.(2π * xgrid0) .* cos.(2π * ygrid0)
heat2d.IC = vec(ux0)  # initial condition

# Boundary condition
Ubc = [1.0, 1.0, -1.0, -1.0]
Ubc = repeat(Ubc, 1, heat2d.time_dim)

# Operators
A, B = heat2d.finite_diff_model(heat2d, heat2d.diffusion_coeffs)

# Integrate
U = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC, Ubc; 
    operators=[A,B], system_input=true, integrator_type=:BackwardEuler
)

U2d = invec.(eachcol(U), heat2d.spatial_dim...)
fig1 = Figure()
ax1 = Axis3(fig1[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
sf = surface!(ax1, heat2d.xspan, heat2d.yspan, U2d[1])
Colorbar(fig1[1, 2], sf)
display(fig1)
```

```@example Heat2D
fig2 = Figure()
ax2 = Axis3(fig2[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
sf = surface!(ax2, heat2d.xspan, heat2d.yspan, U2d[end])
Colorbar(fig2[1, 2], sf)
display(fig2)
```

## API

```@docs
Heat2D
```
