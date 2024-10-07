# Kuramoto-Sivashinsky Equation

## Overview

The Kuramoto-Sivashinsky (KS) equation is a partial differential equation that describes the dynamics of certain spatiotemporal patterns in various physical systems, particularly in fluid dynamics and combustion processes. It was introduced by Yoshiki Kuramoto and Grigoriĭ Isaakovich Sivashinsky independently in the 1970s.

The equation is given by:

```math
\begin{gather*}
    u_t + uu_x + u_{xx} + \mu u_{xxxx} = 0,  \qquad x \in (-\infty, \infty) \\
    u(x,t) = u(x+L,t), \quad u_x(x,t) = u_x(x+L,t), \quad t \geq 0 
\end{gather*}
```

where:
- ``u(x, t)`` represents the spatially and temporally varying quantity being modeled (e.g., the amplitude of certain patterns in a fluid or combustion system).
- ``t`` is time.
- ``x`` is a spatial coordinate.

The terms in the equation have the following meanings:
- ``u_t``: Represents the time rate of change of the quantity u.
- ``uu_x``: Represents advection, describing how the quantity u is transported along the spatial coordinate x by its own gradient.
- ``u_{xx}``: Represents diffusion, describing how the quantity u diffuses or spreads out over space.
- ``u_{xxxx}``: Represents a fourth-order spatial derivative that accounts for certain nonlinear interactions and dispersion effects.
- ``\mu``: Viscosity parameter.

The Kuramoto-Sivashinsky equation is known for producing a rich variety of complex spatiotemporal patterns, including traveling waves, oscillations, and turbulence-like behavior. It has applications in understanding and modeling various physical phenomena, such as the dynamics of flames, fluid interfaces, and certain chemical reactions. The equation's complexity and the diversity of patterns it can exhibit make it a subject of study in nonlinear dynamics and chaos theory.

## Model

For our analysis, we will construct a numerical model by descretizing the original PDE and separating the system into the linear and nonlinear components in the form of 

```math
\begin{gather*}
    \dot{\mathbf u}(t) = \mathbf A \mathbf u(t) + \mathbf F \mathbf u^{\langle 2\rangle}(t)
\end{gather*}
```

where $\mathbf A$, $\mathbf F$ are the linear and quadratic (non-redundant) operators respectively.

!!! todo 
    Currently, only the periodic boundary condition is implemented for the KS equation. Hence, we disregard the control input.

## Numerical Integration

Once we have the numerical model, we integrate it using the Crank-Nicolson and Adam-Bashforth Implicit scheme. The method is as follows.

```math
\begin{gather*}
    \frac{\mathbf u(t_{k+1}) - \mathbf u(t_k)}{\Delta t} = \mathbf A \left(\frac{\mathbf u(t_{k+1}) + \mathbf u(t_k)}{2}\right) + \left[ \frac{3}{2}\mathbf F \left( \mathbf u(t_k) \right)^{\langle 2\rangle} -\frac{1}{2}\mathbf F\left( \mathbf{u}(t_{k-1}) \right)^{\langle 2 \rangle } \right] 
\end{gather*}
```

Hence 

```math
\begin{gather*}
    \mathbf u(t_{k+1}) = 
    \begin{cases} 
        \left(\mathbf I - \frac{\Delta t}{2}\mathbf A \right)^{-1}\left[ \left( \mathbf I + \frac{\Delta t}{2}\mathbf A \right)\mathbf u(t_k) + \Delta t\mathbf F \left( \mathbf u(t_k) \right)^{\langle 2\rangle} \right] \qquad  k = 1 \\[0.3cm]
        \left(\mathbf I - \frac{\Delta t}{2}\mathbf A \right)^{-1}\left[ \left( \mathbf I + \frac{\Delta t}{2}\mathbf A \right)\mathbf u(t_k) + \frac{3\Delta t}{2}\mathbf F \left( \mathbf u(t_k) \right)^{\langle 2\rangle} -\frac{\Delta t}{2}\mathbf F\left( \mathbf u(t_{k-1}) \right)^{\langle 2 \rangle }\right] \qquad  k \geq 2
    \end{cases}
\end{gather*}
```

## Finite Difference Model

Similar to the discretization of the [1D Heat equation](heat1d.md), [viscous Burgers' equation](burgers.md), and etc. we discretize the PDE using finite difference approach. In order to discretize in the spatial and temporal domains we define the grid size $\Delta x$ and time-step $\Delta t$. Furthermore, let the subscript and superscript indicate the spatial and temporal indices respectively, i.e., $u_{n}^k$. Then we show how we discretize each term below.

```math
\begin{align*}
    u_{xxxx} &\approx \frac{1}{\Delta x^4} \left( u_{n+2} - 4u_{n+1} + 6u_n - 4u_{n-1} + u_{n-2} \right) \\
    u_{xx} &\approx \frac{1}{\Delta x^2} \left( u_{n+1} - 2u_n + u_{n-1} \right) \\
    u_x &\approx \frac{1}{2\Delta x} \left( u_{n+1} - u_{n-1} \right) \quad .
\end{align*}
```

Then we can represent the KS equation model with distinct linear and nonlinear terms

```math
\begin{gather*}
    \dot u_n = \underbrace{ \left[ -\frac{\mu}{\Delta x^4} (u_{n+2} + u_{n-2}) + \left( \frac{4\mu}{\Delta x^4}-\frac{1}{\Delta x^2} \right)(u_{n+1} + u_{n-1}) + \left( \frac{2}{\Delta x^2}-\frac{6\mu}{\Delta x^4} \right)u_n \right] }_{\text{linear}}
    + \underbrace{\frac{1}{2\Delta x}\left( u_nu_{n-1} - u_nu_{n+1} \right)}_{\text{nonlinear}}
\end{gather*}
```

Thus, assuming we have periodic boundary conditions, we can represent the KS equation as a linear-quadratic ODE with respect to time by expanding the above expression to all of the spatial grid:

```math
\begin{gather*}
    \mathbf{\dot u}(t) = \mathbf{A}\mathbf u(t) + \mathbf{F} \mathbf u^{\langle 2 \rangle}(t)
\end{gather*}
```

where $\mathbf A \in \mathbb R^{N\times N}$ is the linear operator, $\mathbf F \in \mathbb R^{N \times N(N+1)/2}$ is the quadratic operator, and  $\mathbf u^{\langle 2\rangle} \in \mathbb R^{N(N+1)/2}$ represents the quadratic states with no redundancy. The matrix $\mathbf A$ would be a toeplitz matrix (except for the periodic terms) and the $\mathbf F$ would be a sparse matrix.

## Spectral Method

We consider a periodic domain $[0,L]$ for the solution of the Kuramoto-Sivashinsky equation. Our periodic grid has $N$ points $\{x_n\}_{n=1}^N$, where $x_n = n\Delta x$ and $\Delta x = L/N$. With $N$ degrees of freedom we can write the solution $u(x,t)$ as a truncated Fourier expansion with $N$ modes:

```math
\begin{align*}
u(x,t) = \int_{-\infty}^\infty \hat u_k(t)\exp\left(\frac{2\pi j kx}{L}\right)dk \approx \sum_{k=-N/2}^{N/2-1}\hat{u}_k(t)\exp\left(\frac{2\pi jkx}{L}\right).
\end{align*}
```

Now from here, we introduce two possible methods using the Fourier transform. The first method, uses the Fast Fourier Transform (FFT) or Pseudo-Spectral (PS) method to deal with the nonlinear term. In contrast, the second method directly uses the Fourier Mode by formulating the problem with the Spectral Galerkin (SG) method.

### Pseudo-Spectral Method

We begin by plugging the approximate Fourier transform of $u(x,t)$ back into the original PDE which give us 

```math
\begin{align*}
    \dot u(x,t) &\approx \sum_{k=-N/2}^{N/2-1} \dot{\hat u_k}(t)\exp\left( \frac{2\pi jkx}{L} \right) \\
    \mu u_{xxxx} + u_{xx} &\approx  \sum_{k=-N/2}^{N/2-1}\mu\left( \frac{2\pi j k}{L} \right)^4 \hat u_k(t) \exp\left( \frac{2\pi j k x}{L} \right) + \sum_{k=-N/2}^{N/2-1}\left( \frac{2\pi j k}{L} \right)^2  \hat u_k(t) \exp\left( \frac{2\pi j kx}{L} \right) = -\sum_{k=-N/2}^{N/2-1} \left[ \left( \frac{2\pi k}{L} \right)^2 - \mu\left( \frac{2\pi  k}{L} \right)^4 \right] \hat u_k(t) \exp\left( \frac{2\pi j kx}{L} \right) \\
    uu_x &= \frac{1}{2}\left( u^2 \right)_x \approx \frac{1}{2} \frac{2\pi j k}{L}\left( \hat u^2 \right)_k = \frac{1}{2} \frac{2\pi j k }{L} ~\left(\mathrm{FFT}\left[ u^2(t) \right]\right)_k
\end{align*}
```

Then if we collect the terms within the summation and multiplied by $\exp(2\pi jk x/L )$ we have

```math
\begin{gather*}
    \dot{\hat u}_k(t) = \underbrace{\left[ \left( \frac{2\pi k}{L}\right)^2 - \mu\left( \frac{2\pi k}{L} \right)^4 \right]\hat u_k(t)}_{\text{linear}} - \underbrace{\frac{\pi j k}{L}~\left(\mathrm{FFT}\left[ u^2(t) \right]\right)_k}_{\text{nonlinear}}
\end{gather*}
```

For more detail on the derivation, refer to [this paper](http://pubs.sciepub.com/ajna/2/3/5/abstract.html) by Gentian Zavalani. If we write this in the form of $\mathbf{\dot u}(t) = \mathbf{A}\mathbf u(t) + \mathbf{F} \mathbf u^{\langle 2 \rangle}(t)$ we will have a diagonal matrix of $\mathbf A$ and for $\mathbf F$. The original states $u(t)$ of the KS equation can be retained by performing the inverse FFT (iFFT) on the states $\hat u(t)$.

Or you could let $\mathbf A$ and $\mathbf F$ be a vector and do element-wise multiplications to speed-up the integration process.

## Spectral Galerkin (SG) Method

In the SG method, you take the inner product between the Fourier transformed expression and the exponential to retrieve the Fourier coefficient with the orthogonality condition. For example, for the $-u_{xx}$ term

```math
\begin{align*}
    \left\langle -\partial_x u, \partial_x u\right\rangle &= \left\langle -\partial_x \sum_{l=-N/2}^{N/2-1}\hat u_l(t) \exp\left(\frac{2\pi j l}{L}x\right), ~\partial_x\exp\left( \frac{2\pi j k}{L}x\right) \right\rangle \\
    &= \left\langle - \sum_{l=-N/2}^{N/2-1} \frac{2\pi j l}{L} \hat u_l(t) \exp\left(\frac{2\pi j l}{L}x\right), ~ \frac{2\pi jk}{L} \exp\left( \frac{2\pi j k}{L}x\right) \right\rangle \\
    &{\xrightarrow{l=k}} \left( \frac{2\pi k }{L} \right)^2 \hat u_k(t)
\end{align*}
```

For $u_{xxxx}$ we obtain the same expression as the previous pseudo-spectral method, and therefore, the linear part of the model is exactly the same. However, we take a different route for the nonlinear term. If we consider the conservative advection nonlinearity, the spectral Galerkin would be as follows. 

```math
\begin{align*}
    \left\langle -\frac{1}{2}u^2, ~\partial_x\exp\left( \frac{2\pi jk}{L}x\right)\right\rangle &= \left\langle  -\frac{1}{2}\left[\sum_{p=-N/2}^{N/2-1}\hat u_p(t) \exp\left(\frac{2\pi j p}{L}x\right)\right]\left[\sum_{q=-N/2}^{N/2-1}\hat u_q(t) \exp\left(\frac{2\pi j q}{L}x\right) \right], ~\frac{2\pi jk}{L}\exp\left( \frac{2\pi jk}{L} \right) \right\rangle \\
    &=\left\langle  -\frac{1}{2} \sum_{p=-N/2}^{N/2-1}\sum_{q=-N/2}^{N/2-1}\hat u_p(t) \hat u_q(t) \exp\left[\frac{2\pi j (p+q)}{L}x\right], ~\frac{2\pi jk}{L}\exp\left( \frac{2\pi jk}{L} \right) \right\rangle \\
    &\xrightarrow{p+q~=~k} -\frac{\pi jk}{L}\sum_{p+q=k} \hat u_p(t) \hat u_q(t)
\end{align*}
```

With the linear and nonlinear terms together, we have

```math
\begin{gather*}
    \dot{\hat u}_k(t) = \underbrace{\left[ \left( \frac{2\pi k}{L}\right)^2 - \mu\left( \frac{2\pi k}{L} \right)^4 \right]}_{\text{linear}}\hat u_k(t) +  \underbrace{\frac{-\pi jk}{L} \sum_{p+q=k}\hat u_p(t) \hat u_q(t)}_{\text{nonlinear}} ~ .
\end{gather*}
```

Assume $\hat u_k$ is pure imaginary. Then let us define  $\hat u_k(t) = j\hat v_k(t)$, where $v_k(t) \in \mathbb R^N$, to limit the model in the real space, which brings us to

```math
\begin{gather*}
    \dot{\hat v}_k(t) = \underbrace{\left[ \left( \frac{2\pi k}{L}\right)^2 - \mu\left( \frac{2\pi k}{L} \right)^4 \right]\hat v_k(t)}_{\text{linear}} + \underbrace{\frac{-\pi k}{L} \sum_{p+q=k}\hat v_p(t) \hat v_q(t)}_{\text{nonlinear}} \quad .
\end{gather*}
```

Thus, in the real Fourier space, the model can be expressed as 

```math
\begin{gather*}
    \dot{\hat{\mathbf{v}}}(t) = \mathbf{A}\hat{\mathbf v}(t) + \mathbf{F} \hat{\mathbf v}^{\langle 2 \rangle}(t) \quad .
\end{gather*}
```

Now, since the Fourier transform $\mathcal F$ is a linear operator, so is the inverse Fourier transform $\mathcal F^{-1}$. Hence, the KS model in the time domain is expressed as 

```math
\begin{gather*}
    \mathcal F^{-1}\left[ \dot{\hat{\mathbf{v}}}(t) \right] = \mathbf{A}~\mathcal F^{-1} \left[\hat{\mathbf v}(t)\right] + \mathbf{F} ~\mathcal F^{-1}\left[\hat{\mathbf v}^{\langle 2 \rangle}(t) \right] \quad .
\end{gather*}
```

where the linear and quadratic operators do not change. For more details consult [this paper](https://dx.doi.org/10.1088/0951-7715/10/1/004). 

However if we would not want to assume $\hat u_k$ to be pure imaginary then we will just let 

```math
\begin{gather*}
    \dot{\hat{\mathbf{u}}}(t) = \mathbf{A}\hat{\mathbf u}(t) + \mathbf{F} \hat{\mathbf u}^{\langle 2 \rangle}(t) \quad ,
\end{gather*}
```

where $\mathbf F$ would have complex valued entries. To acquire the original states $u(t)$ of the KS equation we will have to perform the inverse FFT (iFFT) on the state data in the Fourier space.


## Example

For this example, we follow the setup by [Koike2024](@citet).

```@example KSE
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: KuramotoSivashinskyModel

# Settings for the KS equation
Ω = (0.0, 22.0)
dt = 0.01
N = 256
kse = KuramotoSivashinskyModel(
    spatial_domain=Ω, time_domain=(0.0, 300.0), diffusion_coeffs=1.0,
    Δx=(Ω[2] - 1/N)/N, Δt=dt
)
DS = 100
L = kse.spatial_domain[2]

# Initial condition
a = 1.0
b = 0.1
u0 = a*cos.((2*π*kse.xspan)/L) + b*cos.((4*π*kse.xspan)/L)

# Operators
A, F = kse.finite_diff_model(kse, kse.diffusion_coeffs)

# Integrate
U = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F, const_stepsize=true)

# Heatmap
fig1, ax, hm = CairoMakie.heatmap(kse.tspan[1:DS:end], kse.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig1[1, 2], hm)
fig1
```

```@example KSE
# Surface plot
fig11, _, sf = CairoMakie.surface(kse.xspan, kse.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig1[1, 2], sf)
fig11
```

## API

```@docs
PolynomialModelReductionDataset.KuramotoSivashinsky.KuramotoSivashinskyModel
```

```@autodocs
Modules = [PolynomialModelReductionDataset.KuramotoSivashinsky]
Order = [:module, :function, :macro]
```