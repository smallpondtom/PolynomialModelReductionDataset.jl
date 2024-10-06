# 1D Heat Equation

## Overview 

The heat equation is a fundamental partial differential equation that describes the diffusion of heat in a given medium over time. It is used to model how temperature changes spread through materials, such as solids, liquids, and gases. The equation is given by:

```math
u_t = \mu u_{xx}
```

where 

```math
\begin{align}
u_t &= \frac{\partial u(t,x)}{\partial t} \\
u_{xx} &= \frac{\partial^2 u(t,x)}{\partial x^2}
\end{align}
```

where $x\in[0,L]$ is the temperature and $\mu$ is the thermal diffusivity parameter.  

## Finite Difference Model

In order to spatially discretize this PDE to obtain the ODE, which we can numerically integrate, we use the finite difference approach. Let $k = \{1,2,3,\ldots,K\}$ and $n = \{1,2,3,\ldots,N\}$. This means that we are spatially discretizing this PDE using $N$ spatial grid points. Then let $u^k_n$ indicate the state $u$ at the $k$-th discrete time and $n$-th discrete spatial point. With this discrete expression we can express $u_t$ and $u_{xx}$ as follows.

```math
\begin{align}
u_t &= \frac{\partial u(t_k,x_n)}{\partial t} = \frac{u^{k+1}_n - u^k_n}{\Delta t} + \mathcal{O}(\Delta t) \\
u_{xx} &= \frac{\partial^2 u(t_k,x_n)}{\partial x^2} = \frac{u^k_{n+1} - 2u^k_n + u^k_{n-1}}{\Delta x^2} + \mathcal{O}(\Delta x^2)
\end{align}
```

The equations above represent the finite difference for the forward Euler approach. For the backward Euler approach it will change slightly (discussed below). It follows that 

```math
\begin{gather}
u_t  =  \mu\frac{u^k_{n+1} - 2u^k_n + u^k_{n-1}}{\Delta x^2}.
\end{gather}
```

The right-hand side represents the well-known tridiagonal structure arising from the Laplacian operator. Note that, we are abusing some notations since the finite difference method is an approximation where the higher-order terms are truncated, and people use the capital $U$ to denote the approximated state variables.

```math
u_{xx} \approx U_{xx} = \frac{u^k_{n+1} - 2u^k_n + u^k_{n-1}}{\Delta x^2}
```

Now, if we consider all points corresponding to each spatial grid point, the ODE can be represented as 

```math
\dot{\mathbf{u}} = \mathbf{A}\mathbf{u}
```

where 

```math
\begin{gather}
\mathbf{A} = \frac{\mu}{\Delta x^2} \begin{bmatrix}
-2 & 1 & 0 & 0 & 0 & \cdots & 0 \\
1 & -2 & 1 & 0 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & -2 & 1 & \cdots & 0 \\
0 & 0 & 0 & 1 & -2 & \ddots & \vdots \\
\vdots & \vdots & \vdots & \vdots & \ddots & \ddots & 1 \\
0 & 0 & 0 & 0 & 0 & 1 & -2
\end{bmatrix}
\end{gather}
```

Suppose we are using a **Dirichlet** boundary condition of 

```math
u(0,t) = g(t), \qquad u(L,t) = h(t)
```

for our model, then for the initial and last states we can see that $(n-1)$-th and $(n+1)$-th elements are missing, respectively. Thus, we compliment these missing elements for the boundary conditions using the control/input matrix. The boundary conditions can be incorporated by

```math
\begin{gather}
B = \frac{\mu}{\Delta x^2}\begin{bmatrix}
u_0^k & 0 \\ 0 & 0 \\ \vdots & \vdots \\ 0 & 0 \\ 0 & u_{N+1}^k
\end{bmatrix}
\end{gather}
```

Then we arrive at a $N$-dimensional linear time-invariant (LTI) system:

```math
\dot{\mathbf{u}}(t) = \mathbf{A}\mathbf{u}(t) + \mathbf{B}\mathbf{w}(t)
```

where $\mathbf{u}\in\mathbb{R}^N$ is the state vector and $\mathbf{w}= [g(t)~~h(t)]^\top\in\mathbb{R}^2$ is the input vector.

!!! tip Important
    In finite difference approach, it is important to first do the spatial discretization of the system, and then develop the numerical integration scheme. Treating them separately will make your life easier...

We then consider the numerical integration scheme. For our numerical integration we can consider three approaches
- Forward Euler
- Backward Euler
- Crank-Nicolson

We omit the details of each scheme (assuming you can google that). However, if you were to implement the forward Euler scheme, the integration step would be as follows

```math
\begin{gather*}
\frac{\mathbf{u}(t_{k+1}) - \mathbf{u}(t_k)}{\Delta t} = \mathbf{A}\mathbf{u}(t_k) + \mathbf{B}\mathbf{w}(t_k) \\
\mathbf{u}(t_{k+1}) = (\mathbf{I} + \Delta t \mathbf{A})\mathbf{u}(t_k) + \Delta t \mathbf{B}\mathbf{w}(t_k).
\end{gather*}
```

The backward Euler would be 

```math
\begin{gather*}
\frac{\mathbf{u}(t_{k+1}) - \mathbf{u}(t_k)}{\Delta t} = \mathbf{A}\mathbf{u}(t_{k+1}) + \mathbf{B}\mathbf{w}(t_k) \\
\mathbf{u}(t_{k+1}) = (\mathbf{I} - \Delta t \mathbf{A})^{-1}\left[\mathbf{u}(t_k) + \Delta t \mathbf{B}\mathbf{w}(t_k)\right].
\end{gather*}
```

and Crank-Nicolson would be

```math
\begin{gather*}
\frac{\mathbf{u}(t_{k+1}) - \mathbf{u}(t_k)}{\Delta t} = \frac{1}{2}\mathbf{A}\left[\mathbf{u}(t_{k+1})+\mathbf{u}(t_k)\right] + \frac{1}{2}\mathbf{B}\left[\mathbf{w}(t_{k+1})+\mathbf{w}(t_k)\right] \\
\mathbf{u}(t_{k+1}) = \left(\mathbf{I} - \frac{\Delta t}{2} \mathbf{A}\right)^{-1}\left[\left(\mathbf{I}+\frac{\Delta t}{2}\mathbf{A}\right)\mathbf{u}(t_k) + \frac{\Delta t}{2}\mathbf{B}\left(\mathbf{w}(t_{k+1})+\mathbf{w}(t_k)\right)\right].
\end{gather*}
```

And this completes the finite difference model for the 1D heat equation. You can follow the same procedure for the periodic, Neumann, mixed, and other boundary conditions, which I will leave as an exercise for you.

## Example

The example below uses the settings from [Peherstorfer2016](@cite).

```@example Heat1D
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: Heat1DModel

# Setup
Nx = 2^7; dt = 1e-3
heat1d = Heat1DModel(
    spatial_domain=(0.0, 1.0), time_domain=(0.0, 1.0), 
    Δx=1/Nx, Δt=dt, diffusion_coeffs=0.1
)
Ubc = ones(heat1d.time_dim) # boundary condition

# Model operators
A, B = heat1d.finite_diff_model(heat1d, heat1d.diffusion_coeffs; same_on_both_ends=true)

# Integrate
U = heat1d.integrate_model(
    heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
    system_input=true, integrator_type=:BackwardEuler
)

# Surface plot
fig1, _, sf = CairoMakie.surface(heat1d.xspan, heat1d.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig1[1, 2], sf)
display(fig1)
```


```@example Heat1D
# Flow field
fig2, ax, hm = CairoMakie.heatmap(heat1d.xspan, heat1d.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig2[1, 2], hm)
display(fig2)
```

## API

All models are a subset of `AbstractModel`

```@docs
PolynomialModelReductionDataset.AbstractModel
```

```@docs
PolynomialModelReductionDataset.Heat1D.Heat1DModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.Heat1D]
Order = [:module, :function, :macro]
```