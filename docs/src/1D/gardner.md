# Gardner Equation

## Overview

The **Gardner equation**, also known as the **combined Korteweg-de Vries-modified Korteweg-de Vries (KdV-mKdV) equation**, is a nonlinear partial differential equation (PDE) that unifies the Korteweg-de Vries (KdV) and the modified Korteweg-de Vries (mKdV) equations. It models the propagation of nonlinear waves in dispersive media and is significant in the study of soliton interactions, integrable systems, and nonlinear wave phenomena in fluid dynamics and plasma physics.

The standard form of the Gardner equation in one spatial dimension is:

```math
u_t = -\alpha u_{xxx} + \beta uu_x + \gamma u^2 u_{x}
```

Where:

- `` u(x, t) `` is the wave profile at position \( x \) and time \( t \).
- `` \alpha ``, \( \beta \), and \( \gamma \) are constants representing the strengths of the linear and nonlinear terms and dispersion, respectively.

## Key Features

- **Combination of Nonlinearities**: The Gardner equation includes both quadratic (\( u \frac{\partial u}{\partial x} \)) and cubic (\( u^2 \frac{\partial u}{\partial x} \)) nonlinear terms, bridging the KdV and mKdV equations.
- **Dispersion**: The term \( \alpha \frac{\partial^3 u}{\partial x^3} \) accounts for dispersive effects, crucial for balancing nonlinearity to form solitons.
- **Integrability**: The equation is integrable via the inverse scattering transform (IST), indicating the existence of multi-soliton solutions and infinite conservation laws.
- **Soliton Interactions**: Models complex interactions between solitons, including the fusion and fission of solitary waves.

## Physical Interpretation

- **Fluid Dynamics**: Describes shallow water waves in channels where both quadratic and cubic nonlinear effects are significant.
- **Plasma Physics**: Models ion-acoustic waves in plasma with non-Maxwellian electron distributions or in the presence of multiple ion species.
- **Internal Waves**: Represents internal solitary waves in stratified fluids where density variations lead to modified nonlinear effects.

## Properties

### Soliton Solutions

- **One-Soliton Solution**: The Gardner equation admits exact one-soliton solutions, which can be written as:

  ```math
  u(x, t) = A \operatorname{sech}^2 \left( k (x - v t - x_0) \right) + B \operatorname{sech} \left( k (x - v t - x_0) \right)
  ```

  Where:

  - `` A ``, \( B \), and \( k \) are constants determined by \( \alpha \), \( \beta \), \( \gamma \), and the soliton parameters.
  - `` v `` is the velocity of the soliton.
  - `` x_0 `` is the initial position.

- **Multi-Soliton Solutions**: Due to its integrability, the equation supports multi-soliton solutions that interact elastically.

### Transformations

- **Miura Transformation**: The Gardner equation relates to the KdV and mKdV equations through transformations, enabling the mapping of solutions between these equations.

  - From mKdV to KdV:

    ```math
    u_{\mathrm{KdV}} = u_{\mathrm{mKdV}}^2 + \lambda u_{{\mathrm{mKdV}}_x}
    ```

    Where \( \lambda \) is a parameter.

- **Bäcklund Transformations**: Used to generate new solutions from known ones, further illustrating the integrable nature of the equation.

## Applications

- **Shallow Water Waves**: Modeling wave propagation in channels or coastal areas where higher-order nonlinear effects are non-negligible.
- **Plasma Physics**: Studying soliton dynamics in plasmas with complex ion compositions or temperature distributions.
- **Nonlinear Optics**: Describing pulse propagation in optical fibers where both quadratic and cubic nonlinearities are present.
- **Internal Solitary Waves**: Investigating waves within oceans or atmospheres where density stratification affects wave behavior.

## Generalizations

- **Variable Coefficient Gardner Equation**: Incorporates spatial or temporal variations in coefficients to model inhomogeneous or non-stationary media.

  ```math
  \frac{\partial u}{\partial t} + \alpha(x, t) u \frac{\partial u}{\partial x} + \beta(x, t) u^2 \frac{\partial u}{\partial x} + \gamma(x, t) \frac{\partial^3 u}{\partial x^3} = 0
  ```

- **Higher Dimensions**: Extensions to two or three spatial dimensions for more complex wave phenomena.

- **Non-Integrable Perturbations**: Adding terms that break integrability to study the effects of perturbations on soliton dynamics.

## Model

The Gardner equation becomes a quadratic and cubic model:

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

```@example Gardner
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: GardnerModel

# Setup
Ω = (0.0, 3.0)
Nx = 2^8; dt = 1e-3
gardner = GardnerModel(
    spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    params=Dict(:a => 1, :b => 3, :c => 5), BC=:dirichlet,
)
DS = 100
gardner.IC = 2 * cos.(2π * gardner.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * gardner.xspan / (Ω[2] - Ω[1]))
Ubc1 = 0.5ones(1,gardner.time_dim)
Ubc2 = -0.5ones(1,gardner.time_dim)
Ubc = [Ubc1; Ubc2]

# Operators
A, F, E, B = gardner.finite_diff_model(gardner, gardner.params)

# Integrate
U = gardner.integrate_model(
    gardner.tspan, gardner.IC, Ubc; 
    linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
    system_input=true, integrator_type=:CNAB,
)

# Surface plot
fig3, _, sf = CairoMakie.surface(gardner.xspan, gardner.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
display(fig3)
```

```@example Gardner
# Flow field
fig4, ax, hm = CairoMakie.heatmap(gardner.xspan, gardner.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig4)
```

## API

```@docs
Gardner
```