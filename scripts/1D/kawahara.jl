"""
Kawahara equation
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using Revise
using PolynomialModelReductionDataset
const pomoreda = PolynomialModelReductionDataset

#=============================================#
## Model (Periodic BC) dispersion order of 1 ##
#=============================================#
# Settings for the KS equation
Ω = (0.0, 50.0)
dt = 0.01
N = 512
kawahara = pomoreda.KawaharaModel(
    spatial_domain=Ω, time_domain=(0.0, 150.0), 
    params=Dict(:mu => 1.0, :delta => 0.15, :nu => 0.05),
    dispersion_order=5,
    conservation_type=:C,
    Δx=(Ω[2] - 1/N)/N, Δt=dt
)
DS = 100
L = kawahara.spatial_domain[2]

# Initial condition
a = 1.0
b = 0.1
u0 = a*cos.((2*π*kawahara.xspan)/L) + b*cos.((4*π*kawahara.xspan)/L)

#===============================#
## Finite Difference Operators ##
#===============================#
A, F = kawahara.finite_diff_model(kawahara, kawahara.params[:mu], kawahara.params[:delta], kawahara.params[:nu])

#===================#
## Integrate Model ##
#===================#
t1 = @elapsed U = kawahara.integrate_model(
    kawahara.tspan, u0, nothing; 
    linear_matrix=A, quadratic_matrix=F, const_stepsize=true
)

#=================#
## Plot Solution ##
#=================#
# Heatmap
fig1, ax, hm = CairoMakie.heatmap(kawahara.tspan[1:DS:end], kawahara.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig1[1, 2], hm)
display(fig1)

# Surface plot
fig2, _, sf = CairoMakie.surface(kawahara.xspan, kawahara.tspan[1:DS:end], U[:, 1:DS:end], 
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
kawahara = KawaharaModel(
    spatial_domain=Ω, time_domain=(0.0, 300.0), 
    params=Dict(:mu => 1.0, :delta => 0.15),
    dispersion_order=3,
    Δx=(Ω[2] - 1/N)/N, Δt=dt
)
DS = 100
L = kawahara.spatial_domain[2]

# Initial condition
a = 1.0
b = 0.1
u0 = a*cos.((2*π*kawahara.xspan)/L) + b*cos.((4*π*kawahara.xspan)/L)

#===============================#
## Finite Difference Operators ##
#===============================#
A, F = kawahara.finite_diff_model(kawahara, kawahara.params[:mu], kawahara.params[:delta])

#===================#
## Integrate Model ##
#===================#
U = kawahara.integrate_model(
    kawahara.tspan, u0, nothing; 
    linear_matrix=A, quadratic_matrix=F, const_stepsize=true
)

#=================#
## Plot Solution ##
#=================#
# Heatmap
fig3, ax, hm = CairoMakie.heatmap(kawahara.tspan[1:DS:end], kawahara.xspan, U[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig3[1, 2], hm)
display(fig3)

# Surface plot
fig4, _, sf = CairoMakie.surface(kawahara.xspan, kawahara.tspan[1:DS:end], U[:, 1:DS:end], 
    axis=(type=Axis3, xlabel=L"x", ylabel=L"t", zlabel=L"u(x,t)"))
CairoMakie.Colorbar(fig4[1, 2], sf)
display(fig4)

#================#
## Fast solver 
#================#
solver = pomoreda.build_fast_solver(kawahara,
                                    kawahara.params[:mu],
                                    kawahara.params[:delta];
                                    scheme=:CN)
t2 = @elapsed Ufast = pomoreda.integrate_model_fast(
    kawahara, solver, kawahara.tspan, u0;
    quadratic_matrix=F
)

#================#
## Plot solution 
#================#
fig5, ax, hm = CairoMakie.heatmap(kawahara.tspan[1:DS:end], kawahara.xspan, Ufast[:, 1:DS:end]')
ax.xlabel = L"t"
ax.ylabel = L"x"
CairoMakie.Colorbar(fig5[1, 2], hm)
display(fig5)