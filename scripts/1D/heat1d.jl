"""
1D Heat Equation Example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset
const pomoreda = PolynomialModelReductionDataset

#========#
## Model
#========#
Nx = 2^7; dt = 1e-3
heat1d = pomoreda.Heat1DModel(
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
t_cn = @elapsed U = heat1d.integrate_model(
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


#======================#
## Fast BE 
#======================#
solver_be = pomoreda.build_fast_solver(
    heat1d, heat1d.diffusion_coeffs;
    scheme=:BE, same_on_both_ends=true
)
t_fast_be = @elapsed Ufast_be = pomoreda.integrate_model_fast(
    heat1d, solver_be, heat1d.tspan, heat1d.IC, Ubc;
    control_matrix=B, scheme=:BE
)

#================#
## Plot Solution
#================#
# Surface plot
fig5, _, sf = CairoMakie.surface(heat1d.xspan, heat1d.tspan, Ufast_be, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
display(fig5)

# Flow field
fig6, ax, hm = CairoMakie.heatmap(heat1d.xspan, heat1d.tspan, Ufast_be)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig6)

#======================#
## Fast CN
#======================#
solver_cn = pomoreda.build_fast_solver(
    heat1d, heat1d.diffusion_coeffs;
    scheme=:CN, same_on_both_ends=true
)
t_fast_cn = @elapsed Ufast_cn = pomoreda.integrate_model_fast(
    heat1d, solver_cn, heat1d.tspan, heat1d.IC, Ubc;
    control_matrix=B, scheme=:CN
)

#================#
## Plot Solution
#================#
# Surface plot
fig7, _, sf = CairoMakie.surface(heat1d.xspan, heat1d.tspan, Ufast_cn, 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig3[1, 2], sf)
display(fig7)

# Flow field
fig8, ax, hm = CairoMakie.heatmap(heat1d.xspan, heat1d.tspan, Ufast_cn)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig4[1, 2], hm)
display(fig8)