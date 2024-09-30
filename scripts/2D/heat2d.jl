"""
2D Heat Equation Example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: Heat2DModel
using UniqueKronecker: invec

#========#
## Model
#========#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,2), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    diffusion_coeffs=0.1, BC=(:dirichlet, :dirichlet)
)
xgrid0 = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
ux0 = sin.(2π * xgrid0) .* cos.(2π * ygrid0)
heat2d.IC = vec(ux0)  # initial condition

# Boundary condition
Ubc = [1.0, 1.0, -1.0, -1.0]
Ubc = repeat(Ubc, 1, heat2d.time_dim)

#==================#
## Model Operators
#==================#
A, B = heat2d.finite_diff_model(heat2d, heat2d.diffusion_coeffs)

#================#
## Forward Euler
#================#
U = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC, Ubc; 
    operators=[A,B], system_input=true, integrator_type=:ForwardEuler
)

#=================#
## Backward Euler
#=================#
U = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC, Ubc; 
    operators=[A,B], system_input=true, integrator_type=:BackwardEuler
)

#==================#
## Animate Solution
#==================#
U2d = invec.(eachcol(U), heat2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, heat2d.xspan, heat2d.yspan, U2d[1])
    hm = heatmap!(ax2, heat2d.xspan, heat2d.yspan, U2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/2D/plots/heat2D_temperature_backwardeuler.mp4", 1:heat2d.time_dim) do i
        sf[3] = U2d[i]
        hm[3] = U2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end

#=================#
## Crank-Nicolson
#=================#
U = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC, Ubc; 
    operators=[A,B], system_input=true, integrator_type=:CrankNicolson
)

#==================#
## Animate Solution
#==================#
U2d = invec.(eachcol(U), heat2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, heat2d.xspan, heat2d.yspan, U2d[1])
    hm = heatmap!(ax2, heat2d.xspan, heat2d.yspan, U2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/2D/plots/heat2D_temperature_cranknicolson.mp4", 1:heat2d.time_dim) do i
        sf[3] = U2d[i]
        hm[3] = U2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end