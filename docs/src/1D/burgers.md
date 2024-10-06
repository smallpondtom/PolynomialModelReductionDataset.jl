# Viscous Burgers' Equation

## Overview 

The viscous Burger's equation or viscous Bateman-Burger's equation is a second order PDE used widely in many fields as a simplified model, for example, fluid mechanics, acoustics, etc. The state $u(x,t)$ is a velocity field consisting of the spatial and temporal variables. The PDE is expressed as 

```math
u_t = \mu u_{xx} - uu_x
```

where 

```math
\begin{align}
u_t &= \frac{\partial u(t,x)}{\partial t} \\
u_x &= \frac{\partial u(t,x)}{\partial x} \\
u_{xx} &= \frac{\partial^2 u(t,x)}{\partial x^2}
\end{align}
```

for $x\in[0,L]$, and $\mu$ is the viscosity parameter. 

## Finite Difference Model

Let $k = \{1,2,3,\ldots,K\}$ and $n = \{0, 1,2,3,\ldots,N\}$. Further let $u^k_j$ indicate the state $u$ at the $k$-th discrete time and $n$-th discrete spatial point. With this discrete expression we can express $u_t$, $u_x$, and $u_{xx}$ as follows.

```math
\begin{align}
u_t &= \frac{\partial u(t_k,x_n)}{\partial t} = \frac{u^{k+1}_n - u^k_n}{\Delta t} + \mathcal{O}(\Delta t) \\
u_x &= \frac{\partial u(t_k,x_n)}{\partial t} = \frac{u^{k}_{n+1} - u^k_{n-1}}{2\Delta x} + \mathcal{O}(\Delta x) \\
u_{xx} &= \frac{\partial^2 u(t_k,x_n)}{\partial x^2} = \frac{u^k_{n+1} - 2u^k_n + u^k_{n-1}}{\Delta x^2} + \mathcal{O}(\Delta x^2)
\end{align}
```

For $u_t$ we use the standard first order finite difference, and for $u_{xx}$ we use the second order central finite difference and since we will implement the semi-implicit integration (backward Euler for only this term without the viscid term). Now, if we look carefully, we see that for $u_x$ we use the leap-frog finite difference where we jump from the $(n+1)$-th term to the $(n-1)$-th term. Plugging in the above expressions and with some abuse of notation we have

```math
\begin{gather}
    u_t  =  \mu\frac{u^{k+1}_{n+1} - 2u^{k+1}_n + u^{k+1}_{n-1}}{\Delta x^2} - u_n^k \frac{u^k_{n+1} - u^k_{n-1}}{2\Delta x}
\end{gather}
```

From this, we can form the system matrices, but keep in mind that for the Burger's equation we include the boundary conditions in the $A$ and $F$ matrix unlike the 1D heat equation example. This is because we see that the quadratic term also includes the $(n+1)$- and $(n-1)$-th term requiring the state vector to contain the terms from $u_0$ all the way to $u_{N+1}$. To deal with this, assuming constant time-step, we approximate the time derivative of the state variables corresponding to the boundary as 

```math
\begin{align}
    \dot u_0 &= \frac{u_0^{k+1} - u_0^k}{\Delta t} \\
    \dot u_{N+1} &= \frac{u_{N+1}^{k+1} - u_{N+1}^k}{\Delta t} 
\end{align}
```

If we have Dirichlet boundary condition 

```math
u(0,t) = g(t), \qquad u(L,t) = h(t)
```

then

```math
\begin{align}
    \dot u_0 &= \frac{1}{\Delta t}g(t_{k+1}) - \frac{1}{\Delta t}u_0^k \\
    \dot u_{N+1} &= \frac{1}{\Delta t}h(t_{k+1}) - \frac{1}{\Delta t}u_{N+1}^k
\end{align}
```

To accommodate for the time derivative terms of the boundary, we construct the linear system matrix and control matrix as follows:

```math
\begin{gather}
A = \frac{\mu}{\Delta x^2} \begin{bmatrix}
\frac{\Delta x^2}{\mu}\left( -\frac{1}{\Delta t} \right) & 0 & 0 & 0 & 0 & \cdots & 0 & 0\\
1 & -2 & 1 & 0 & 0 & \cdots & 0 & 0\\
0 & 1 & -2 & 1 & 0 & \cdots & 0 & 0\\
0 & 0 & 1 & -2 & 1 & \cdots & 0 & 0\\
0 & 0 & 0 & 1 & -2 & \cdots & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots &\vdots \\
0 & 0 & 0 & 0 & 0 & \cdots & -2 & 1 \\
0 & 0 & 0 & 0 & 0 & \cdots & 0 & \frac{\Delta x^2}{\mu} \left( -\frac{1}{\Delta t} \right)
\end{bmatrix}
\end{gather}
```

and 

```math
\begin{gather}
B = \begin{bmatrix}
\frac{1}{\Delta t} & 0 \\[0.5em] 0 & 0 \\ \vdots & \vdots \\ 0 & 0 \\[0.5em] 0 & \frac{1}{\Delta t}
\end{bmatrix}
\end{gather}
```

In $A$, besides the first and last rows, we have a tridiagonal toeplitz matrix formed by the stencil $\{1,-2, 1\}$. However, the first and last rows of matrices $A$ and $B$ construct the finite difference at the boundaries.

The $F$ matrix, which is the quadratic operator without the non-redundancy from the symmetry of Kronecker products, is a bit tricky to wrap your mind around. So let's go through it with examples. First, let $u_n \in \mathbb{R}^3$ where $n = \{0,1,2\}$, then 

```math
\begin{gather}
u^{(2)} = \begin{bmatrix}
    u_0^2 & u_1u_0 & u_2u_0 & u_1^2 & u_2u_1 & u_2^2 
\end{bmatrix}^\top
\end{gather}
```

and 

```math
\begin{gather}
    F = \frac{1}{2\Delta x} \begin{bmatrix}
        0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & -1 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
\end{gather}
```

The first and last rows of $F$ are all zeros since the quadratic operator does not contribute to the boundary conditions, and the middle row have non-zero entries for the $u_nu_{n-1}$ and $u_nu_{n+1}$ terms where $n=1$ in this case. Similarly, for $u_n \in \mathbb{R}^4$ where $n = \{0,1,2,3\}$, then 

```math
\begin{gather}
u^{(2)} = \begin{bmatrix}
    u_0^2 & u_1u_0 & u_2u_0 & u_3u_0 & u_1^2 & u_2u_1 & u_3u_1 & u_2^2 & u_3u_2 & u_3^2
\end{bmatrix}^\top
\end{gather}
```

and 

```math
\begin{gather}
    F = \frac{1}{2\Delta x} \begin{bmatrix}
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & -1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
\end{gather}
```
    
where the first and last rows are zeros and the second and third rows have non-zero elements for the $u_nu_{n-1}$ and $u_nu_{n+1}$ terms. Note that the sequence which represents the row with nonzero entries in the $F$ matrix can be represented as 

```math
\begin{gather}
    \begin{cases}
        a_1 = 2 \\
        a_{n+1} = a_n + N - n + 1
    \end{cases}
\end{gather}
```

where $N$ is the total number of variables, which leads to

```math
\begin{gather}
    a_n = a_1 + \sum_{k=1}^{n-1}(N - k + 1) = 2 - \frac{1}{2}n(n-1) + (N+1)(n-1) 
\end{gather}
```

Thus the we arrive at a $N$-dimensional nonlinear ODE:

```math
\dot{\mathbf{u}}(t) = \mathbf{Au}(t) + \mathbf{F}(\mathbf{u}(t) \oslash \mathbf{u}(t)) + \mathbf{Bw}(t)
```

where $\mathbf{u}\in\mathbb{R}^N$, $\oslash$ indicates the unique Kronecker product which omits the redundant terms from the standard Kronecker product, and $\mathbf{w} = [g(t)~~h(t)]^\top \in \mathbb{R}^2$ is the input vector.

For numerical integration, we apply the semi-implicit Euler where the linear term is implicit and the quadratic and control terms are explicit. The time-stepping expression is defined by

```math
\begin{gather}
\mathbf{u}(t_{k+1}) = (\mathbf{I} - \Delta t \mathbf{A})^{-1}\left\{\mathbf{u}(t_k) + \Delta tF\left[\mathbf{u}(t_k) \oslash \mathbf{u}(t_k) \right] + \Delta t B\mathbf{w}(t_{k+1})\right\}.
\end{gather}
```

You can follow the same procedure for the periodic, Neumann, mixed, and other boundary conditions, which I will leave as an exercise for you.

## Conservativeness

For the periodic model, we consider 3 different models which are:

- conservative form: $u_t = \mu u_{xx} + \nabla_x (\frac{1}{2}u^2)$
- non-conservative form: $u_t = \mu u_{xx} + uu_x$
- energy-preserving form which is a convex combination of the conservative and non-conservative forms with a weight of $1/3$ which preserves the energy of the system [Aref1984](@cite)

## Example

The example below uses the settings from [Peherstorfer2016](@cite).

```@example Burgers
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: BurgersModel

# Setup
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.1, BC=:dirichlet,
)
Ubc = rand(burgers.time_dim) # boundary condition

# Model operators
A, F, B = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs; opposite_sign_on_ends=true)

# Integrate
U = burgers.integrate_model(
    burgers.tspan, burgers.IC, Ubc; 
    linear_matrix=A, control_matrix=B, quadratic_matrix=F,
    system_input=true
)

# Surface plot
fig1, _, sf = CairoMakie.surface(burgers.xspan, burgers.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig1[1, 2], sf)
display(fig1)
```

```@example Burgers
# Flow field
fig2, ax, hm = CairoMakie.heatmap(burgers.xspan, burgers.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig2[1, 2], hm)
display(fig2)
```

## API

```@docs
PolynomialModelReductionDataset.Burgers.BurgersModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.Burgers]
Order = [:module, :function, :macro]
```