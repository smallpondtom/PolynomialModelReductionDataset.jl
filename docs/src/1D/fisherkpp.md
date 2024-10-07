# Fisher-KPP Equation

## Overview

The **Fisher-KPP equation**, also known as the Fisher equation or the Kolmogorov-Petrovsky-Piskunov equation, is a fundamental nonlinear partial differential equation (PDE) in mathematical biology. It models the spread of an advantageous gene in a population or, more generally, the propagation of a biological or chemical species. Independently introduced by **Ronald Fisher** in 1937 and by **Kolmogorov, Petrovsky, and Piskunov (KPP)** in the same year, it combines diffusion and logistic growth to describe wave-like phenomena in population dynamics.

The one-dimensional Fisher-KPP equation is given by:

```math
u_t = D u_{xx} + r u (1 - u)
```

Where:

- `` u(x, t) `` is the density of the species or the frequency of an advantageous gene at position \( x \) and time \( t \).
- `` D `` is the diffusion coefficient, representing spatial dispersal.
- `` r `` is the intrinsic growth rate of the population.

### Key Features

- **Reaction-Diffusion System**: Combines local reactions (logistic growth) and spatial diffusion.
- **Nonlinearity**: The term `` r u (1 - u) `` introduces nonlinearity, leading to complex dynamics like traveling waves.
- **Traveling Wave Solutions**: Admits solutions of the form \( u(x, t) = U(x - c t) \), representing waves propagating at constant speed \( c \).

### Physical Interpretation

- **Population Genetics**: Models the spread of a beneficial gene through a spatially distributed population.
- **Ecology**: Describes the invasion of a species into new territory.
- **Chemical Kinetics**: Represents autocatalytic chemical reactions spreading through a medium.

### Properties

#### Traveling Wave Solutions

- Seeking solutions of the form \( u(x, t) = U(z) \) with \( z = x - c t \), the equation becomes an ordinary differential equation (ODE):

  ```math
  -c \frac{dU}{dz} = D \frac{d^2 U}{dz^2} + r U (1 - U)
  ```

- Boundary conditions for invasion problems are:

  ```math
  U(-\infty) = 1, \quad U(\infty) = 0
  ```

- **Minimum Wave Speed**:

  The minimum speed \( c_{\text{min}} \) at which traveling waves propagate is:

  ```math
  c_{\text{min}} = 2 \sqrt{D r}
  ```

#### Stability and Asymptotic Behavior

- **Stability of Traveling Waves**: Waves traveling at speeds \( c \geq c_{\text{min}} \) are stable.
- **Asymptotic Spread**: The population front advances at speed \( c_{\text{min}} \) over long times.

#### Linearization Near Zero

- Near \( u = 0 \), the equation can be linearized:

  ```math
  \frac{\partial u}{\partial t} \approx D \frac{\partial^2 u}{\partial x^2} + r u
  ```

- Solutions grow exponentially if \( u > 0 \), leading to the invasion of the population.

### Applications

- **Population Dynamics**: Modeling species invasion, range expansion, and the spread of epidemics.
- **Genetics**: Describing gene propagation in spatially structured populations.
- **Ecology and Conservation Biology**: Understanding the impact of habitat fragmentation and corridors on species spread.
- **Chemical and Biological Waves**: Studying flame propagation, neural activity, and reaction-diffusion systems.

### Generalizations

- **Higher Dimensions**: Extension to two or three spatial dimensions:

  ```math
  \frac{\partial u}{\partial t} = D \nabla^2 u + r u (1 - u)
  ```

- **Heterogeneous Media**: Incorporating spatially varying parameters \( D(x) \) and \( r(x) \).
- **Time-Delayed Reactions**: Introducing delays in the reaction term to model maturation time.

## Model

This equation is a quadratic model

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{F}(\mathbf{u}(t) \oslash \mathbf{u}(t)) + \mathbf{Bw}(t)
```

## Numerical Integration

For the numerical integration we consider two methods:
- Semi-Implicit Crank-Nicolson (SICN)
- Crank-Nicolson Adam-Bashforth (CNAB)

The time stepping for each methods are 

__SICN__:

```math
\mathbf{u}(t_{k+1}) = \left(\mathbf{I}-\frac{\Delta t}{2}\mathbf{A}\right)^{-1} \left\{ \left( \mathbf{I} + \frac{\Delta t}{2}\mathbf{A} \right)\mathbf{u}(t_k) + \Delta t\mathbf{F}\mathbf{u}^{\langle 2\rangle}(t_k) + \frac{\Delta t}{2} \mathbf{B}\left[ \mathbf{w}(t_{k+1}) + \mathbf{w}(t_k) \right]\right\}
```

__CNAB__:

If $k=1$

```math
\mathbf{u}(t_{k+1}) = \left(\mathbf{I}-\frac{\Delta t}{2}\mathbf{A}\right)^{-1} \left\{ \left( \mathbf{I} + \frac{\Delta t}{2}\mathbf{A} \right)\mathbf{u}(t_k) + \Delta t\mathbf{F}\mathbf{u}^{\langle 2\rangle}(t_k) + \frac{\Delta t}{2} \mathbf{B}\left[ \mathbf{w}(t_{k+1}) + \mathbf{w}(t_k) \right]\right\}
```

If $k\geq 2$

```math
\mathbf{u}(t_{k+1}) = \left(\mathbf{I}-\frac{\Delta t}{2}\mathbf{A}\right)^{-1} \left\{ \left( \mathbf{I} + \frac{\Delta t}{2}\mathbf{A} \right)\mathbf{u}(t_k) + \frac{3\Delta t}{2}\mathbf{F}\mathbf{u}^{\langle 2\rangle}(t_k) - \frac{\Delta t}{2}\mathbf{F}\mathbf{u}^{\langle 2\rangle}(t_{k-1}) + \frac{\Delta t}{2} \mathbf{B}\left[ \mathbf{w}(t_{k+1}) + \mathbf{w}(t_k) \right]\right\}
```

where $\mathbf{u}^{\langle 2 \rangle}=\mathbf{u} \oslash \mathbf{u}$.

## Example

```@example FisherKPP
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: FisherKPPModel

# Setup
Ω = (0, 3)  # Choose integers
T = (0.0, 5.0)
Nx = 2^8
dt = 1e-3
fisherkpp = FisherKPPModel(
    spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
    params=Dict(:D => 1.0, :r => 1.0), BC=:dirichlet
)
DS = 10

# Create piecewise IC
a, b, c = (sort ∘ rand)(1:fisherkpp.spatial_dim, 3)
seg1 = ones(length(fisherkpp.xspan[1:a]))
seg2 = zeros(length(fisherkpp.xspan[a+1:b]))
seg3 = rand(0.1:0.1:0.5) * ones(length(fisherkpp.xspan[b+1:c]))
seg4 = zeros(length(fisherkpp.xspan[c+1:end]))
fisherkpp.IC = [seg1; seg2; seg3; seg4]

# Boundary conditions
Ubc1 = ones(1,fisherkpp.time_dim)
Ubc2 = zeros(1,fisherkpp.time_dim)
Ubc = [Ubc1; Ubc2]

# Operators
A, F, B = fisherkpp.finite_diff_model(fisherkpp, fisherkpp.params)

# Integrate
U = fisherkpp.integrate_model(
    fisherkpp.tspan, fisherkpp.IC, Ubc; 
    linear_matrix=A, quadratic_matrix=F, control_matrix=B,
    system_input=true, integrator_type=:CNAB,
)

# Surface plot
fig3, _, sf = CairoMakie.surface(fisherkpp.xspan, fisherkpp.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
fig3
```

```@example FisherKPP
# Flow field
fig4, ax, hm = CairoMakie.heatmap(fisherkpp.xspan, fisherkpp.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
fig4
```

## API

```@docs
PolynomialModelReductionDataset.FisherKPP.FisherKPPModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.FisherKPP]
Order = [:module, :function, :macro]
```