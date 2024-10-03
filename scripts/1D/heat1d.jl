"""
1D Heat Equation Example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: Heat1DModel

#========#
## Model
#========#
Nx = 2^7; dt = 1e-3
heat1d = Heat1DModel(
    spatial_domain=(0.0, 1.0), time_domain=(0.0, 1.0), 
    Δx=1/Nx, Δt=dt, diffusion_coeffs=0.1
)
Ubc = ones(heat1d.time_dim) # boundary condition

#==================#
## Model Operators
#==================#
A, B = heat1d.finite_diff_model(heat1d, heat1d.diffusion_coeffs; same_on_both_ends=true)

#================#
## Forward Euler
#================#
U = heat1d.integrate_model(
    heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B, 
    system_input=true, integrator_type=:ForwardEuler
)
println("Forward Euler is unstable for 1D heat equation")

#=================#
## Backward Euler
#=================#
U = heat1d.integrate_model(
    heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
    system_input=true, integrator_type=:BackwardEuler
)

#================#
## Plot Solution
#================#
# Surface plot
fig1, _, sf = CairoMakie.surface(heat1d.xspan, heat1d.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig1[1, 2], sf)
display(fig1)

# Flow field
fig2, ax, hm = CairoMakie.heatmap(heat1d.xspan, heat1d.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig2[1, 2], hm)
display(fig2)

#=================#
## Crank-Nicolson
#=================#
U = heat1d.integrate_model(
    heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
    system_input=true, integrator_type=:CrankNicolson
)

#================#
## Plot Solution
#================#
# Surface plot
fig3, _, sf = CairoMakie.surface(heat1d.xspan, heat1d.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
display(fig3)

# Flow field
fig4, ax, hm = CairoMakie.heatmap(heat1d.xspan, heat1d.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig4)