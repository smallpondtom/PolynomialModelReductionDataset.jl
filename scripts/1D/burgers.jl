"""
Viscous Burgers' Equation Example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: BurgersModel

#=======================#
## Model (Dirichlet BC)
#=======================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.1, BC=:dirichlet,
)
Ubc = rand(burgers.time_dim) # boundary condition


#==================#
## Model Operators
#==================#
A, B, F = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs; opposite_sign_on_ends=true)

#============#
## Integrate
#============#
U = burgers.integrate_model(
    burgers.tspan, burgers.IC, Ubc; 
    linear_matrix=A, control_matrix=B, quadratic_matrix=F,
    system_input=true
)

#================#
## Plot Solution
#================#
# Surface plot
fig1, _, sf = CairoMakie.surface(burgers.xspan, burgers.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig1[1, 2], sf)
display(fig1)

# Flow field
fig2, ax, hm = CairoMakie.heatmap(burgers.xspan, burgers.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig2[1, 2], hm)
display(fig2)

#===================================#
## Model (Periodic BC Conservative)
#===================================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.1, BC=:periodic, conservation_type=:C
)
burgers.IC = sin.(2π * burgers.xspan) # initial condition

#==================#
## Model Operators
#==================#
A, F = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs)

#============#
## Integrate
#============#
U = burgers.integrate_model(
    burgers.tspan, burgers.IC, Float64[]; 
    linear_matrix=A, quadratic_matrix=F,
    system_input=false
)

#================#
## Plot Solution
#================#
# Surface plot
fig3, _, sf = CairoMakie.surface(burgers.xspan, burgers.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
display(fig3)

# Flow field
fig4, ax, hm = CairoMakie.heatmap(burgers.xspan, burgers.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig4)

#=======================================#
## Model (Periodic BC Non-Conservative)
#=======================================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.1, BC=:periodic, conservation_type=:NC
)
burgers.IC = sin.(2π * burgers.xspan) # initial condition

#==================#
## Model Operators
#==================#
A, F = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs)

#============#
## Integrate
#============#
U = burgers.integrate_model(
    burgers.tspan, burgers.IC, Float64[]; 
    linear_matrix=A, quadratic_matrix=F,
    system_input=false
)

#================#
## Plot Solution
#================#
# Surface plot
fig5, _, sf = CairoMakie.surface(burgers.xspan, burgers.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig5[1, 2], sf)
display(fig5)

# Flow field
fig6, ax, hm = CairoMakie.heatmap(burgers.xspan, burgers.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig6[1, 2], hm)
display(fig6)

#========================================#
## Model (Periodic BC Energy-Preserving)
#========================================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.1, BC=:periodic, conservation_type=:EP
)
burgers.IC = sin.(2π * burgers.xspan) # initial condition

#==================#
## Model Operators
#==================#
A, F = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs)

#============#
## Integrate
#============#
U = burgers.integrate_model(
    burgers.tspan, burgers.IC, Float64[]; 
    linear_matrix=A, quadratic_matrix=F,
    system_input=false
)

#================#
## Plot Solution
#================#
# Surface plot
fig7, _, sf = CairoMakie.surface(burgers.xspan, burgers.tspan, U, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig7[1, 2], sf)
display(fig7)

# Flow field
fig8, ax, hm = CairoMakie.heatmap(burgers.xspan, burgers.tspan, U)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig8[1, 2], hm)
display(fig8)