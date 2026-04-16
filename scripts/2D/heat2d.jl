"""
2D Heat Equation Example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using Revise
using PolynomialModelReductionDataset: Heat2DModel, FastDenseSolver, build_fast_be_solver, integrate_model_fast
using UniqueKronecker: invec

#=======================#
## Model (Dirichlet BC)
#=======================#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,0.5), 
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
    heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B, 
    system_input=true, integrator_type=:ForwardEuler
)

#=================#
## Backward Euler
#=================#
t1 = @elapsed U = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B,
    system_input=true, integrator_type=:BackwardEuler
)

#========================#
## Backward Euler (Fast)
#========================#
μ = heat2d.diffusion_coeffs
solver = build_fast_be_solver(heat2d, μ)  # one-time precompute
t2 = @elapsed U_fast = integrate_model_fast(solver, B, Ubc, heat2d.tspan, heat2d.IC)

# Verify fast implementation 
println("Backward Euler rel. error: ", norm(U - U_fast) / norm(U))


#========================================#
## ROM (dense unstructured fast solver)
#========================================#
F = svd(U)
Vr = F.U[:, 1:30]
Ar = Vr' * A * Vr
Br = Vr' * B
IC_r = Vr' * heat2d.IC
solver = FastDenseSolver(Ar, heat2d.Δt)
t3 = @elapsed U_rom = integrate_model_fast(solver, heat2d.tspan, IC_r, Br, Ubc)
println("ROM Backward Euler time: ", t3, " seconds")

# Compare to ROM reconstruction with full order model
U_recon = Vr * U_rom
println("ROM rel. error: ", norm(U - U_recon) / norm(U))

## Run the old slower solver 
t4 = @elapsed U_rom2 = heat2d.integrate_model(
    heat2d.tspan, IC_r, Ubc; linear_matrix=Ar, control_matrix=Br,
    system_input=true, integrator_type=:BackwardEuler
)
println("ROM Backward Euler (old solver) time: ", t4, " seconds")
U_recon2 = Vr * U_rom2
println("ROM (old solver) rel. error: ", norm(U - U_recon2) / norm(U))


# # Change Δt cheaply
# update_timestep!(solver, Δt_new)
# u_rom2 = integrate_model_fast(solver, tdata_new, u0r, Br, input_new)

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
        # autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end

#=================#
## Crank-Nicolson
#=================#
U = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B, 
    system_input=true, integrator_type=:CrankNicolson
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

#=======================#
## Model (Periodic BC)
#=======================#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^5
Ny = 2^5
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,0.5), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    diffusion_coeffs=0.1, BC=(:periodic, :periodic)
)
xgrid0 = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
ux0 = sin.(2π * xgrid0) .* cos.(2π * ygrid0)
heat2d.IC = vec(ux0)  # initial condition

#==================#
## Model Operators
#==================#
A = heat2d.finite_diff_model(heat2d, heat2d.diffusion_coeffs)

#=================#
## Backward Euler
#=================#
t1 = @elapsed U = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC; linear_matrix=A, 
    system_input=false, integrator_type=:BackwardEuler
)

#=================#
## Backward Euler (Fast)
#=================#
μ = heat2d.diffusion_coeffs
t2 = @elapsed U_fast = integrate_model_fast(heat2d, μ, heat2d.tspan, heat2d.IC)

# Verify fast implementation
println("Backward Euler rel. error: ", norm(U - U_fast) / norm(U))

#==================#
## Animate Solution
#==================#
U2d = invec.(eachcol(U), heat2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)", 
                limits=(
                    nothing, nothing,
                    nothing, nothing,
                    -1.0, 1.0
                ))
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, heat2d.xspan, heat2d.yspan, U2d[1])
    hm = heatmap!(ax2, heat2d.xspan, heat2d.yspan, U2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/2D/plots/heat2D_temperature_cranknicolson_periodic.mp4", 1:heat2d.time_dim) do i
        sf[3] = U2d[i]
        hm[3] = U2d[i]
        # autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end

