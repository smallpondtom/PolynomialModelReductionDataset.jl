"""
Kawahara equation
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using Revise
using PolynomialModelReductionDataset: KawaharaModel

#=============================================#
## Model (Periodic BC) dispersion order of 1 ##
#=============================================#
# Settings for the KS equation
Ω = (0.0, 50.0)
dt = 0.01
N = 512
kse = KawaharaModel(
    spatial_domain=Ω, time_domain=(0.0, 150.0), 
    params=Dict(:mu => 1.0, :delta => 0.15, :nu => 0.05),
    dispersion_order=5,
    conservation_type=:C,
    Δx=(Ω[2] - 1/N)/N, Δt=dt
)
DS = 100
L = kse.spatial_domain[2]

# Initial condition
a = 1.0
b = 0.1
u0 = a*cos.((2*π*kse.xspan)/L) + b*cos.((4*π*kse.xspan)/L)

#===============================#
## Finite Difference Operators ##
#===============================#
A, F = kse.finite_diff_model(kse, kse.params[:mu], kse.params[:delta], kse.params[:nu])

#===================#
## Integrate Model ##
#===================#
U = kse.integrate_model(
    kse.tspan, u0, nothing; 
    linear_matrix=A, quadratic_matrix=F, const_stepsize=true
)

#=================#
## Plot Solution ##
#=================#
# Heatmap
fig1, ax, hm = CairoMakie.heatmap(kse.tspan[1:DS:end], kse.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig1[1, 2], hm)
display(fig1)

# Surface plot
fig2, _, sf = CairoMakie.surface(kse.xspan, kse.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig2[1, 2], sf)
display(fig2)

#=============================================#
## Model (Periodic BC) dispersion order of 3 ##
#=============================================#
# Settings for the KS equation
Ω = (0.0, 50.0)
dt = 0.01
N = 512
kse = KawaharaModel(
    spatial_domain=Ω, time_domain=(0.0, 300.0), 
    params=Dict(:mu => 1.0, :delta => 0.15),
    dispersion_order=3,
    Δx=(Ω[2] - 1/N)/N, Δt=dt
)
DS = 100
L = kse.spatial_domain[2]

# Initial condition
a = 1.0
b = 0.1
u0 = a*cos.((2*π*kse.xspan)/L) + b*cos.((4*π*kse.xspan)/L)

#===============================#
## Finite Difference Operators ##
#===============================#
A, F = kse.finite_diff_model(kse, kse.params[:mu], kse.params[:delta])

#===================#
## Integrate Model ##
#===================#
U = kse.integrate_model(
    kse.tspan, u0, nothing; 
    linear_matrix=A, quadratic_matrix=F, const_stepsize=true
)

#=================#
## Plot Solution ##
#=================#
# Heatmap
fig3, ax, hm = CairoMakie.heatmap(kse.tspan[1:DS:end], kse.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig3[1, 2], hm)
display(fig3)

# Surface plot
fig4, _, sf = CairoMakie.surface(kse.xspan, kse.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig4[1, 2], sf)
display(fig4)
