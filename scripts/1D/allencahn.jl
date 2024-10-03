"""
Allen-cahn exmaple
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset: AllenCahnModel

#======================#
## Model (Periodic BC)
#======================#
Ω = (0.0, 2.0)
T = (0.0, 3.0)
Nx = 2^8
dt = 1e-3
allencahn = AllenCahnModel(
    spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
    params=Dict(:μ => 0.01, :ϵ => 1.0), BC=:periodic
)
DS = 10
allencahn.IC = exp.(-10 * cos.(π * allencahn.xspan).^2)

#==================#
## Model Operators
#==================#
A, E = allencahn.finite_diff_model(allencahn, allencahn.params)

#============#
## Integrate
#============#
U = allencahn.integrate_model(
    allencahn.tspan, allencahn.IC; 
    linear_matrix=A, cubic_matrix=E, system_input=false,
    integrator_type=:CNAB
)

#================#
## Plot Solution
#================#
# Surface plot
fig1, _, sf = CairoMakie.surface(allencahn.xspan, allencahn.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig1[1, 2], sf)
display(fig1)

# Flow field
fig2, ax, hm = CairoMakie.heatmap(allencahn.xspan, allencahn.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig2[1, 2], hm)
display(fig2)

#=======================#
## Model (Dirichlet BC)
#=======================#
Ω = (-1.0, 1.0)
T = (0.0, 3.0)
Nx = 2^8
dt = 1e-3
allencahn = AllenCahnModel(
    spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
    params=Dict(:μ => 0.001, :ϵ => 1.0), BC=:dirichlet
)
DS = 10
allencahn.IC = 0.53*allencahn.xspan + 0.47*sin.(-1.5π * allencahn.xspan)
Ubc1 = ones(1,allencahn.time_dim)
Ubc2 = -ones(1,allencahn.time_dim)
Ubc = [Ubc1; Ubc2]

#==================#
## Model Operators
#==================#
A, E, B = allencahn.finite_diff_model(allencahn, allencahn.params)

#============#
## Integrate
#============#
U = allencahn.integrate_model(
    allencahn.tspan, allencahn.IC, Ubc; 
    linear_matrix=A, cubic_matrix=E, control_matrix=B,
    system_input=true, integrator_type=:CNAB,
)

#================#
## Plot Solution
#================#
# Surface plot
fig3, _, sf = CairoMakie.surface(allencahn.xspan, allencahn.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
display(fig3)

# Flow field
fig4, ax, hm = CairoMakie.heatmap(allencahn.xspan, allencahn.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig4)
