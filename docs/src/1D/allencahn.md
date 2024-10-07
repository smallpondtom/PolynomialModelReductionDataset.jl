# Allen-Cahn equation

## Overview

The Allen-Cahn equation is a fundamental partial differential equation (PDE) in the field of mathematical physics and materials science. Introduced by Samuel Allen and John W. Cahn in 1979, it models the process of phase separation in multi-component alloy systems, particularly the evolution of interfaces between different phases in a material.

The Allen-Cahn equation is typically written as:

```math
u_t = \Delta u - F'(u)
```

- ``u(x,t)``: represents a scalar field, often associated with the order parameter or concentration difference between two phases at position $x$ and time $t$.
- ``\Delta``: denotes the Laplacian operator, representing diffusion in space.
- ``F(u)``: A double-well potential function, commonly chosen as $F(u) = \frac{1}{4}(u^2 - 1)^2$. The derivative $F'(u)=u^3-u$ represents the reaction term driving the phase separation.

### Physical Interpretation

- __Phase Separation__: the equation models how a homogeneous mixture evolves into distinct phases over time, a process driven by minimization of the system's free energy.
- __Interface Dynamics__: it describes the motion of interfaces (or domain walls) between different phases, influenced by surface tension and curvature effects.

### Some properties

1. __Gradient Flow__: the Allen-Cahn equation is the $L_2$-gradient flow of the Ginzburg-Landau energy functional:

   ```math
   E(u) = \int \left( \frac{1}{2}|\nabla u|^2 + F(u) \right)dx
   ```

    This means the evolution of $u$ seeks to decrease the energy $E(u)$ over time.
2. __Connection to Mean Curvature Flow__: in the sharp interface limit (when the interface thickness tends to zero), the motion of the interface approximates mean curvature flow. This links the Allen-Cahn equation to geometric PDEs and has significant implications in differential geometry.
3. __Metastability__: the equation exhibits metastable behavior, where solutions can remain in unstable equilibrium states for extended periods before transitioning to a stable configuration.

## Model

The Allen-Cahn equation becomes a cubic model:

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
- Semi-Implicit Crank-Nicolson (SICN)
- Crank-Nicolson Adam-Bashforth (CNAB)

The time stepping for each methods are 

__SICN__:

```math
\mathbf{u}(t_{k+1}) = \left(\mathbf{I}-\frac{\Delta t}{2}\mathbf{A}\right)^{-1} \left\{ \left( \mathbf{I} + \frac{\Delta t}{2}\mathbf{A} \right)\mathbf{u}(t_k) + \Delta t\mathbf{E}\mathbf{u}^{\langle 3\rangle}(t_k) + \frac{\Delta t}{2} \mathbf{B}\left[ \mathbf{w}(t_{k+1}) + \mathbf{w}(t_k) \right]\right\}
```

__CNAB__:

If $k=1$

```math
\mathbf{u}(t_{k+1}) = \left(\mathbf{I}-\frac{\Delta t}{2}\mathbf{A}\right)^{-1} \left\{ \left( \mathbf{I} + \frac{\Delta t}{2}\mathbf{A} \right)\mathbf{u}(t_k) + \Delta t\mathbf{E}\mathbf{u}^{\langle 3\rangle}(t_k) + \frac{\Delta t}{2} \mathbf{B}\left[ \mathbf{w}(t_{k+1}) + \mathbf{w}(t_k) \right]\right\}
```

If $k\geq 2$

```math
\mathbf{u}(t_{k+1}) = \left(\mathbf{I}-\frac{\Delta t}{2}\mathbf{A}\right)^{-1} \left\{ \left( \mathbf{I} + \frac{\Delta t}{2}\mathbf{A} \right)\mathbf{u}(t_k) + \frac{3\Delta t}{2}\mathbf{E}\mathbf{u}^{\langle 3\rangle}(t_k) - \frac{\Delta t}{2}\mathbf{E}\mathbf{u}^{\langle 3\rangle}(t_{k-1}) + \frac{\Delta t}{2} \mathbf{B}\left[ \mathbf{w}(t_{k+1}) + \mathbf{w}(t_k) \right]\right\}
```

where $\mathbf{u}^{\langle 3 \rangle}=\mathbf{u} \oslash \mathbf{u} \oslash \mathbf{u}$.

!!! Note
    Please go over the derivations in [1D heat equation](heat1d.md) and [Viscous Burgers' equation](burgers.md) for details on how to construct each operators.

## Example

```@example AllenCahn
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: AllenCahnModel

# Setup
Ω = (-1.0, 1.0)
T = (0.0, 3.0)
Nx = 2^8
dt = 1e-3
allencahn = AllenCahnModel(
    spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
    params=Dict(:μ => 0.001, :ϵ => 1.0), BC=:dirichlet
)
DS = 10
allencahn.IC = 0.53*allencahn.xspan + 0.47*sin.(-1.5π * allencahn.xspan)
Ubc1 = ones(1,allencahn.time_dim)
Ubc2 = -ones(1,allencahn.time_dim)
Ubc = [Ubc1; Ubc2]

# Operators
A, E, B = allencahn.finite_diff_model(allencahn, allencahn.params)

# Integrate
U = allencahn.integrate_model(
    allencahn.tspan, allencahn.IC, Ubc; 
    linear_matrix=A, cubic_matrix=E, control_matrix=B,
    system_input=true, integrator_type=:CNAB,
)

# Surface plot
fig3, _, sf = CairoMakie.surface(allencahn.xspan, allencahn.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
fig3
```

```@example AllenCahn
# Flow field
fig4, ax, hm = CairoMakie.heatmap(allencahn.xspan, allencahn.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
fig4
```

## API

```@docs
PolynomialModelReductionDataset.AllenCahn.AllenCahnModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.AllenCahn]
Order = [:module, :function, :macro]
```