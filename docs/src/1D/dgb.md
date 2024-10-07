# Damped Gardner-Burgers Equation


## Overview

This equation is a modified version of the Gardner equation where the diffusion term $u_{xx}$ and damping term $u$ is added for stability. The equation is defined by

```math
u_t = -\alpha u_{xxx} + \beta uu_x + \gamma u^2 u_{x} + \delta u_{xx} + \epsilon u
```

## Model

The damped Gardner-Burgers equation becomes the exact same as the original Gardner equation:

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{H}(\mathbf{u}(t) \otimes \mathbf{u}(t)) + \mathbf{G}(\mathbf{u}(t) \otimes \mathbf{u}(t) \otimes \mathbf{u}(t)) + \mathbf{Bw}(t)
```

or 

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{F}(\mathbf{u}(t) \oslash \mathbf{u}(t)) + \mathbf{E}(\mathbf{u}(t) \oslash \mathbf{u}(t) \oslash \mathbf{u}(t)) + \mathbf{Bw}(t)
```

where
- ``\mathbf{u}\in\mathbb{R}^N``: the state vector
- ``\mathbf{w}\in\mathbb{R}^m``: the input vector
- ``\mathbf{A}\in\mathbb{R}^{N\times N}``: the linear state matrix
- ``\mathbf{H}\in\mathbb{R}^{N\times N^2}``: the quadratic state matrix with redundancy
- ``\mathbf{F}\in\mathbb{R}^{N\times N(N+1)/2}``: the quadratic state matrix without redundancy
- ``\mathbf{G}\in\mathbb{R}^{N\times N^3}``: the cubic state matrix with redundancy
- ``\mathbf{E}\in\mathbb{R}^{N\times N(N+1)(N+2)/6}``: the cubic state matrix without redundancy
- ``\mathbf{B}\in\mathbb{R}^{N\times m}``: the control input matrix

## Numerical Integration

For the numerical integration we consider two methods:
- Semi-Implicit Euler (SIE)
- Crank-Nicolson Adam-Bashforth (CNAB)

For the exact expressions of the time-stepping check [Allen-Cahn equation](allencahn.md).

## Examples

```@example DGB
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: DampedGardnerBurgersModel

# Setup
Ω = (0.0, 3.0)
Nx = 2^8; dt = 1e-3
dgb = DampedGardnerBurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    params=Dict(:a => 1, :b => 3, :c => 5, :d => 0.2, :e => 0.5), BC=:dirichlet,
)
DS = 100
dgb.IC = 2 * cos.(2π * dgb.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * dgb.xspan / (Ω[2] - Ω[1]))
Ubc1 = 0.5ones(1,dgb.time_dim)
Ubc2 = -0.5ones(1,dgb.time_dim)
Ubc = [Ubc1; Ubc2]

# Operators
A, F, E, B = dgb.finite_diff_model(dgb, dgb.params)

# Integrate
U = dgb.integrate_model(
    dgb.tspan, dgb.IC, Ubc; 
    linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
    system_input=true, integrator_type=:CNAB,
)

# Surface plot
fig3, _, sf = CairoMakie.surface(dgb.xspan, dgb.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
fig3
```

```@example DGB
# Flow field
fig4, ax, hm = CairoMakie.heatmap(dgb.xspan, dgb.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
fig4
```

## API

```@docs
PolynomialModelReductionDataset.DampedGardnerBurgers.DampedGardnerBurgersModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.DampedGardnerBurgers]
Order = [:module, :function, :macro]
```