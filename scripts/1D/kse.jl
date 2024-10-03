"""
Kuramoto-Sivashinsky equation
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: KuramotoSivashinskyModel

#======================#
## Model (Periodic BC)
#======================#
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

#=============================#
## Finite Difference Operators
#==============================#
A, F = kse.finite_diff_model(kse, kse.diffusion_coeffs)

#==================#
## Integrate Model
#==================#
U = kse.integrate_model(kse.tspan, u0, nothing; operators=[A, F], const_stepsize=true)

#================#
## Plot Solution
#================#
# Heatmap
fig1, ax, hm = CairoMakie.heatmap(kse.tspan[1:DS:end], kse.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig1[1, 2], hm)
display(fig1)

# Surface plot
fig11, _, sf = CairoMakie.surface(kse.xspan, kse.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig1[1, 2], sf)
display(fig11)

#============================#
## Pseudo-Spectral Operators
#============================#
A, F = kse.pseudo_spectral_model(kse, kse.diffusion_coeffs)

#==================#
## Integrate Model
#==================#
U, Uhat = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F)

#================#
## Plot Solution
#================#
fig2, ax, hm = CairoMakie.heatmap(kse.tspan[1:DS:end], kse.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig2[1, 2], hm)
display(fig2)

#========================================#
## Elementwise Pseudo-Spectral Operators
#========================================#
A, F = kse.elementwise_pseudo_spectral_model(kse, kse.diffusion_coeffs)

#==================#
## Integrate Model
#==================#
U, Uhat = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F)

#================#
## Plot Solution
#================#
fig3, ax, hm = CairoMakie.heatmap(kse.tspan[1:DS:end], kse.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig3[1, 2], hm)
display(fig3)

#==============================#
## Spectral-Galerkin Operators
#==============================#
A, F = kse.spectral_galerkin_model(kse, kse.diffusion_coeffs)

#==================#
## Integrate Model
#==================#
U, Uhat = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F)

#================#
## Plot Solution
#================#
fig4, ax, hm = CairoMakie.heatmap(kse.tspan[1:DS:end], kse.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig4)