# Modified Korteweg-de Vries Equation

The **Modified Korteweg-de Vries (mKdV) equation** is a nonlinear partial differential equation (PDE) that is a variation of the well-known Korteweg-de Vries (KdV) equation. The mKdV equation is significant in the study of solitons, integrable systems, and nonlinear wave propagation. It models phenomena in various physical contexts, such as nonlinear optics, plasma physics, and fluid dynamics.

The standard form of the mKdV equation in one spatial dimension is:

```math
u_t + \alpha u_{xxx} + \beta u^2u_{x} = 0
```

Where:

- `` u(x, t) `` is a scalar function representing the wave profile at position \( x \) and time \( t \).
- ``\alpha`` and ``\beta`` are parameters.

## Key Features

- **Nonlinearity**: The term \( u^2 \frac{\partial u}{\partial x} \) introduces cubic nonlinearity, leading to rich dynamics.
- **Dispersion**: The third-order derivative \( \frac{\partial^3 u}{\partial x^3} \) accounts for dispersive effects.
- **Integrability**: The mKdV equation is integrable via the inverse scattering transform (IST), similar to the KdV equation.

## Physical Interpretation

- **Nonlinear Wave Propagation**: Describes the evolution of nonlinear waves in dispersive media.
- **Solitons**: Supports solitary wave solutions that maintain their shape during propagation and after interactions.
- **Applications**:
  - **Plasma Physics**: Modeling ion-acoustic waves in plasmas.
  - **Nonlinear Optics**: Describing pulse propagation in optical fibers under certain conditions.
  - **Fluid Dynamics**: Representing internal waves in stratified fluids.

## Properties

### Soliton Solutions

- The mKdV equation admits soliton solutions, which can be obtained using methods like the inverse scattering transform.
- **One-Soliton Solution**:

  For the focusing mKdV equation, the one-soliton solution is:

  ```math
  u(x, t) = \pm \frac{v}{\sqrt{2}} \operatorname{sech} \left( \frac{v}{\sqrt{2}} (x - v t - x_0) \right)
  ```

  Where:

  - \( v \) is the velocity of the soliton.
  - \( x_0 \) is the initial position.

### Miura Transformation

- There exists a connection between the mKdV and KdV equations via the **Miura transformation**:

  ```math
  u = w_x + w^2
  ```

  Where \( w \) satisfies the KdV equation. This transformation maps solutions of the mKdV equation to solutions of the KdV equation, linking their integrable structures.

## Generalizations

- **Higher-Order mKdV Equations**: Including higher-order terms to model more complex physical situations.
- **Coupled mKdV Equations**: Systems of mKdV equations modeling interactions between multiple wave modes.
- **Non-integrable Variants**: Modifications that break integrability but model additional physical effects.

## Applications

- **Nonlinear Optics**: Pulse shaping and propagation in optical fibers with specific nonlinear characteristics.
- **Plasma Physics**: Studying nonlinear structures like solitons and shock waves in plasma environments.
- **Mathematical Physics**: Exploring integrable systems, symmetry reductions, and exactly solvable models.

## Model 

The Modified Korteweg-de Vries equation becomes a cubic model:

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

```@example mKdV
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: ModifiedKortewegDeVriesModel

# Setup
Ω = (0.0, 3.0)
Nx = 2^8; dt = 1e-3
mKdV = ModifiedKortewegDeVriesModel(
    spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    params=Dict(:a => 1, :b => 3), BC=:dirichlet,
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
display(fig3)
```

```@example mKdV
# Flow field
fig4, ax, hm = CairoMakie.heatmap(mKdV.xspan, mKdV.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig4)
```

## API

```@docs
ModifiedKortewegDeVries
```