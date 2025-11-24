"""
2D Heat Equation Example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using Revise
using PolynomialModelReductionDataset: AllenCahn2DModel
using UniqueKronecker: invec

#=======================#
## Model (Dirichlet BC)
#=======================#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
allencahn2d = AllenCahn2DModel(
    spatial_domain=Ω, time_domain=(0,1.0), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    params=Dict(:μ => 0.001, :ϵ => 1.0), BC=(:dirichlet, :dirichlet)
)
# spatial meshes (Ny x Nx for grid arrays to match vec column-major flattening)
xg = allencahn2d.xspan
yg = allencahn2d.yspan
X = repeat(xg', length(yg), 1)    # size (Ny, Nx)
Y = repeat(yg, 1, length(xg))     # size (Ny, Nx)

# initial condition: low-frequency pattern + small random perturbation (Ny x Nx)
ux0 = 0.1 .* sin.(2π .* X) .* cos.(2π .* Y) .+ 0.02 .* randn(size(X))
allencahn2d.IC = vec(ux0)  # column-major flattening

# Dirichlet boundary values: [left, right, bottom, top] fixed to stable phases
Ubc_vals = [1.0, 1.0, -1.0, -1.0]
Ubc = repeat(reshape(Ubc_vals, 4, 1), 1, allencahn2d.time_dim)

#==================#
## Model Operators
#==================#
A, E, B = allencahn2d.finite_diff_model(allencahn2d, allencahn2d.params)

#==================#
## Integrate model
#==================#
U = allencahn2d.integrate_model(
    allencahn2d.tspan, allencahn2d.IC, Ubc; linear_matrix=A, control_matrix=B,
    cubic_matrix=E, system_input=true, const_stepsize=true, 
    integrator_type=:SICN,
)

#==================#
## Animate Solution
#==================#
U2d = invec.(eachcol(U), allencahn2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    hm = heatmap!(ax2, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/2D/plots/allencahn2D_dirichlet.mp4", 1:allencahn2d.time_dim) do i
        sf[3] = U2d[i]
        hm[3] = U2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end

#================================================#
## Model (Dirichlet BC) (random initial condition)
#================================================#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
allencahn2d = AllenCahn2DModel(
    spatial_domain=Ω, time_domain=(0,1.0), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
     params=Dict(:μ => 1e-2, :ϵ => 1.0), BC=(:dirichlet, :dirichlet)
)

# Random initial condition for phase separation
using Random
rng = MersenneTwister(1234)  # for reproducibility
xgrid0 = allencahn2d.xspan
ygrid0 = allencahn2d.yspan
X_grid = repeat(xgrid0', length(ygrid0), 1)    # size (Ny, Nx)
Y_grid = repeat(ygrid0, 1, length(xgrid0))     # size (Ny, Nx)

# Random values between -1 and 1, scaled by 0.1
ux0 = 0.1 .* (2.0 .* rand(rng, size(X_grid)...) .- 1.0)
allencahn2d.IC = vec(ux0)  # initial condition

# Dirichlet boundary values: [left, right, bottom, top] fixed to stable phases
Ubc_vals = [0.0, 0.0, 0.0, 0.0]
Ubc = repeat(reshape(Ubc_vals, 4, 1), 1, allencahn2d.time_dim)

#==================#
## Model Operators
#==================#
A, E, B = allencahn2d.finite_diff_model(allencahn2d, allencahn2d.params)

#=================#
## Crank-Nicolson
#=================#
U = allencahn2d.integrate_model(
    allencahn2d.tspan, allencahn2d.IC, Ubc; 
    linear_matrix=A, control_matrix=B,
    cubic_matrix=E, system_input=true, const_stepsize=true, 
)

#==================#
## Animate Solution
#==================#
U2d = invec.(eachcol(U), allencahn2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)",)
                # limits=(
                #     nothing, nothing,
                #     nothing, nothing,
                #     0.1, 1.0
                # ))
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    hm = heatmap!(ax2, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/2D/plots/allencahn2d_random.mp4", 1:allencahn2d.time_dim) do i
        sf[3] = U2d[i]
        hm[3] = U2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end

#================================================#
## Model (Periodic BC) (star initial condition)
#================================================#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
allencahn2d = AllenCahn2DModel(
    spatial_domain=Ω, time_domain=(0,1.0), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
     params=Dict(:μ => 1e-2, :ϵ => 1.0), BC=(:periodic, :periodic)
)

# Star-shaped initial condition (centered at 0.5, 0.5)
xgrid0 = allencahn2d.xspan
ygrid0 = allencahn2d.yspan
X_grid = repeat(xgrid0', length(ygrid0), 1)    # size (Ny, Nx)
Y_grid = repeat(ygrid0, 1, length(xgrid0))     # size (Ny, Nx)

# Compute angle θ for each grid point
θ = zeros(size(X_grid))
for j in axes(X_grid, 1), i in axes(X_grid, 2)
    x_centered = X_grid[j, i] - 0.5
    y_centered = Y_grid[j, i] - 0.5
    if x_centered > 0.0
        θ[j, i] = atan(y_centered / x_centered)
    else
        θ[j, i] = π + atan(y_centered / x_centered)
    end
end

# Star radius with 6 points
r_star = 0.25 .+ 0.1 .* cos.(6 .* θ)

# Distance from center
r_actual = sqrt.((X_grid .- 0.5).^2 .+ (Y_grid .- 0.5).^2)

# Diffuse interface parameter (from model)
ϵ = allencahn2d.params[:ϵ]

# Star initial condition using tanh
ux0 = tanh.((r_star .- r_actual) ./ sqrt(2 * ϵ))
allencahn2d.IC = vec(ux0)  # initial condition

#==================#
## Model Operators
#==================#
A, E = allencahn2d.finite_diff_model(allencahn2d, allencahn2d.params)

#=================#
## Crank-Nicolson
#=================#
U = allencahn2d.integrate_model(
    allencahn2d.tspan, allencahn2d.IC; linear_matrix=A, cubic_matrix=E,
    system_input=false, const_stepsize=true,
)

#==================#
## Animate Solution
#==================#
U2d = invec.(eachcol(U), allencahn2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)",)
                # limits=(
                #     nothing, nothing,
                #     nothing, nothing,
                #     -1.0, 1.0
                # ))
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    hm = heatmap!(ax2, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/2D/plots/allencahn2d_star.mp4", 1:allencahn2d.time_dim) do i
        sf[3] = U2d[i]
        hm[3] = U2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end

#================================================#
## Model (periodic BC) (torus initial condition)
#================================================#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
allencahn2d = AllenCahn2DModel(
    spatial_domain=Ω, time_domain=(0,1.0), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
     params=Dict(:μ => 3e-2, :ϵ => 1.0), BC=(:periodic, :periodic)
)

# Torus-shaped initial condition (centered at 0.5, 0.5)
xgrid0 = allencahn2d.xspan
ygrid0 = allencahn2d.yspan
X_grid = repeat(xgrid0', length(ygrid0), 1)    # size (Ny, Nx)
Y_grid = repeat(ygrid0, 1, length(xgrid0))     # size (Ny, Nx)

# XY = (x - 0.5)^2 + (y - 0.5)^2
XY = sqrt.((X_grid .- 0.5).^2 .+ (Y_grid .- 0.5).^2)

# Torus radii
R1 = 0.35  # major (outside) circle radius
R2 = 0.15  # minor (inside) circle radius

# Diffuse interface parameter (from model)
ϵ = 1 / sqrt(allencahn2d.params[:ϵ])

# Torus initial condition using tanh
ux0 = -1.0 .+ tanh.((R1 .- XY) ./ sqrt(2 * ϵ)) .- tanh.((R2 .- XY) ./ sqrt(2 * ϵ))
allencahn2d.IC = vec(ux0)  # initial condition

#==================#
## Model Operators
#==================#
A, E = allencahn2d.finite_diff_model(allencahn2d, allencahn2d.params)

#=================#
## Crank-Nicolson
#=================#
U = allencahn2d.integrate_model(
    allencahn2d.tspan, allencahn2d.IC; linear_matrix=A, 
    cubic_matrix=E, system_input=false, const_stepsize=true,
)

#==================#
## Animate Solution
#==================#
U2d = invec.(eachcol(U), allencahn2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)",)
                # limits=(
                #     nothing, nothing,
                #     nothing, nothing,
                #     -1.45, -0.8
                # ))
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    hm = heatmap!(ax2, allencahn2d.xspan, allencahn2d.yspan, U2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/2D/plots/allencahn2d_torus.mp4", 1:allencahn2d.time_dim) do i
        sf[3] = U2d[i]
        hm[3] = U2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end

