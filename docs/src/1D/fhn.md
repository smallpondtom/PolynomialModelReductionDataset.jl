# FitzHugh-Nagumo Equation

## Overview

The Fitzhugh-Nagumo equation is a simplified mathematical model used to describe the behavior of excitable systems, particularly the dynamics of nerve cells or neurons. It was developed independently by Richard Fitzhugh and John Nagumo in the 1960s as a modification of the Hodgkin-Huxley model, aiming to capture the essential characteristics of action potential propagation.

The equation takes the form of a system of ordinary differential equations and consists of two main variables: a fast-recovery variable (usually denoted as v) and a slow-activating variable (usually denoted as w). The Fitzhugh-Nagumo equation is given by:

```math
\begin{align*}
    \frac{dv}{dt} &= v - \frac{v^3}{3} - w + I \\
    \frac{dw}{dt} &= \epsilon (v + a - bw)
\end{align*}
```

where:
- ``v``: represents the membrane potential of the neuron.
- ``w``: represents a recovery variable related to the inactivation of ion channels.
- ``I``: is an external input current that stimulates the neuron.
- ``\epsilon``: is a positive parameter that controls the timescale separation between ``v`` and ``w`` dynamics.
- ``a`` and ``b`` are constants that affect the equilibrium points and behavior of the system.

The Fitzhugh-Nagumo model simplifies the complex behavior of neurons, focusing on the interplay between the fast-responding action potential and the slower recovery process. This simplification makes it easier to analyze and understand the fundamental mechanisms underlying excitable systems. The Fitzhugh-Nagumo equation has been widely used in the study of neuronal dynamics, cardiac rhythm modeling, and other excitable phenomena, providing insights into the generation and propagation of electrical signals in various biological and physical contexts.

## Model

The Fitzhugh-Nagumo model can be modelled as a quadratic-bilinear system after a process called __lifting__ 

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{H}(\mathbf{u}(t) \otimes \mathbf{u}(t)) + \mathbf{Bw}(t) + \mathbf{Nu}(t)\mathbf{w}(t) + \mathbf{K}
```

or

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{F}(\mathbf{u}(t) \oslash \mathbf{u}(t)) + \mathbf{Bw}(t)+ \mathbf{Nu}(t)\mathbf{w}(t) + \mathbf{K}
```

where
- ``\mathbf{u}\in\mathbb{R}^N``: the state vector
- ``\mathbf{w}\in\mathbb{R}^m``: the input vector
- ``\mathbf{A}\in\mathbb{R}^{N\times N}``: the linear state matrix
- ``\mathbf{H}\in\mathbb{R}^{N\times N^2}``: the quadratic state matrix with redundancy
- ``\mathbf{F}\in\mathbb{R}^{N\times N(N+1)/2}``: the quadratic state matrix without redundancy
- ``\mathbf{B}\in\mathbb{R}^{N\times m}``: the control input matrix
- ``\mathbf{N}\in\mathbb{R}^{N\times N}``: the bilinear matrix
- ``\mathbf{K}\in\mathbb{R}^{N}``: the constant matrix

For full details on the model see [Morwiki_modFHN](@citet) and [Qian2020](@citet).

## Numerical Integration

The numerical integration is handled by standard forward Euler scheme.

## Example 

This example is a reproduction of the example in [Qian2020](@citet).

```@example 
using CairoMakie
using Kronecker: ⊗
using LinearAlgebra
using PolynomialModelReductionDataset: FitzHughNagumoModel

# Setup
Ω = (0.0, 1.0); dt = 1e-4; Nx = 2^9
fhn = FitzHughNagumoModel(
    spatial_domain=Ω, time_domain=(0.0,4.0), Δx=(Ω[2] - 1/Nx)/Nx, Δt=dt,
    alpha_input_params=500, beta_input_params=10,
)
α = 500; β = 10
g(t) = α * t^3 * exp(-β * t)
U = g.(fhn.tspan)'
DS = 100  # downsample rate

# Operators
Af, Bf, Cf, Kf, f = fhn.full_order_model(fhn.spatial_dim, fhn.spatial_domain[2])
fom(x, u) = Af * x + Bf * u + f(x,u) + Kf

# Integrate
X = fhn.integrate_model(fhn.tspan, fhn.IC, g; functional=fom)[:, 1:DS:end]

# Plot solution
fig1 = Figure()
gp = fhn.spatial_dim
ax1 = Axis(fig1[1, 1], xlabel="t", ylabel="x", title="x1")
hm1 = CairoMakie.heatmap!(ax1, fhn.tspan[1:DS:end], fhn.xspan, X[1:gp, :]')
CairoMakie.Colorbar(fig1[1, 2], hm1)
ax2 = Axis(fig1[1, 3], xlabel="t", ylabel="x", title="x2")
hm2 = CairoMakie.heatmap!(ax2, fhn.tspan[1:DS:end], fhn.xspan, X[gp+1:end, :]')
CairoMakie.Colorbar(fig1[1, 4], hm2)
fig1
```

## API

```@docs
PolynomialModelReductionDataset.FitzHughNagumo.FitzHughNagumoModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.FitzHughNagumo]
Order = [:module, :function, :macro]
```