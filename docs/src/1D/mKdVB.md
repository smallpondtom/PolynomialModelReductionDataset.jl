# Modified Korteweg-de Vries-Burgers Equation

## Overview

This is a modified version of the mKdV equation where we included the diffusion term $u_{xx}$. The equation is defined by

```math
u_t + \alpha u_{xxx} + \beta u^2u_{x} + \gamma u_{xx} = 0
```

## Model

The Modified Korteweg-de Vries-Burgers equation becomes a cubic model:

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{G}(\mathbf{u}(t) \otimes \mathbf{u}(t) \otimes \mathbf{u}(t)) + \mathbf{Bw}(t)
```

or 

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{E}(\mathbf{u}(t) \oslash \mathbf{u}(t) \oslash \mathbf{u}(t)) + \mathbf{Bw}(t)
```

where
- ``\mathbf{u}\in\mathbb{R}^N``: the state vector
- ``\mathbf{w}\in\mathbb{R}^m``: the input vector
- ``\mathbf{A}\in\mathbb{R}^{N\times N}``: the linear state matrix
- ``\mathbf{G}\in\mathbb{R}^{N\times N^3}``: the cubic state matrix with redundancy
- ``\mathbf{E}\in\mathbb{R}^{N\times N(N+1)(N+2)/6}``: the cubic state matrix without redundancy
- ``\mathbf{B}\in\mathbb{R}^{N\times m}``: the control input matrix

## Numerical Integration

For the numerical integration we consider two methods:
- Semi-Implicit Euler (SIE)
- Crank-Nicolson Adam-Bashforth (CNAB)

For the exact expressions of the time-stepping check [Allen-Cahn equation](allencahn.md).

## Example

```@example mKdVB
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: ModifiedKortewegDeVriesBurgersModel

# Setup
Ω = (0.0, 3.0)
Nx = 2^8; dt = 1e-3
mKdV = ModifiedKortewegDeVriesBurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    params=Dict(:a => 1, :b => 3, :c=> 0.1), BC=:dirichlet,
)
DS = 100
mKdV.IC = 2 * cos.(2π * mKdV.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * mKdV.xspan / (Ω[2] - Ω[1]))
Ubc1 = 0.5ones(1,mKdV.time_dim)
Ubc2 = -0.5ones(1,mKdV.time_dim)
Ubc = [Ubc1; Ubc2]

# Operators
A, E, B = mKdV.finite_diff_model(mKdV, mKdV.params)

# Integrate
U = mKdV.integrate_model(
    mKdV.tspan, mKdV.IC, Ubc; 
    linear_matrix=A, cubic_matrix=E, control_matrix=B,
    system_input=true, integrator_type=:CNAB,
)

# Surface plot
fig3, _, sf = CairoMakie.surface(mKdV.xspan, mKdV.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
fig3
```

```@example mKdVB
# Flow field
fig4, ax, hm = CairoMakie.heatmap(mKdV.xspan, mKdV.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
fig4

```

## API

```@docs
PolynomialModelReductionDataset.ModifiedKortewegDeVriesBurgers.ModifiedKortewegDeVriesBurgersModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.ModifiedKortewegDeVriesBurgers]
Order = [:module, :function, :macro]
```