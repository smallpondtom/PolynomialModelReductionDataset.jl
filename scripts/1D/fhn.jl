"""
FitzHugh-Nagumo Model
"""

#===========#
## Packages
#===========#
using CairoMakie
using Kronecker: ⊗
using LinearAlgebra
using PolynomialModelReductionDataset: FitzHughNagumoModel

#=======================#
## Model (Dirichlet BC)
#=======================#
Ω = (0.0, 1.0); dt = 1e-4; Nx = 2^9
fhn = FitzHughNagumoModel(
    spatial_domain=Ω, time_domain=(0.0,4.0), Δx=(Ω[2] - 1/Nx)/Nx, Δt=dt,
    alpha_input_params=500, beta_input_params=10,
)
α = 500; β = 10
g(t) = α * t^3 * exp(-β * t)
U = g.(fhn.tspan)'
DS = 100  # downsample rate

#=============================#
## Full Order Model Operators
#=============================#
Af, Bf, Cf, Kf, f = fhn.full_order_model(fhn.spatial_dim, fhn.spatial_domain[2])
fom(x, u) = Af * x + Bf * u + f(x,u) + Kf

#============#
## Integrate
#============#
X = fhn.integrate_model(fhn.tspan, fhn.IC, g; functional=fom)[:, 1:DS:end]

#================#
## Plot Solution
#================#
fig1 = Figure()
gp = fhn.spatial_dim
ax1 = Axis(fig1[1, 1], xlabel="t", ylabel="x", title="x1")
hm1 = CairoMakie.heatmap!(ax1, fhn.tspan[1:DS:end], fhn.xspan, X[1:gp, :]')
CairoMakie.Colorbar(fig1[1, 2], hm1)
ax2 = Axis(fig1[1, 3], xlabel="t", ylabel="x", title="x2")
hm2 = CairoMakie.heatmap!(ax2, fhn.tspan[1:DS:end], fhn.xspan, X[gp+1:end, :]')
CairoMakie.Colorbar(fig1[1, 4], hm2)
display(fig1)

# #=========================#
# ## Lifted Model Operators
# #=========================#
# A, B, C, H, N, K = fhn.lifted_finite_diff_model(fhn.spatial_dim, fhn.spatial_domain[2])
# flift(x, u) = A * x + B * u[1] + H * (x ⊗ x) + (N * x) * u[1] + K

# #============#
# ## Integrate
# #============#
# X = fhn.integrate_model(fhn.tspan, fhn.IC_lift, U; functional=flift)[:, 1:DS:end]

# #================#
# ## Plot Solution
# #================#
# fig2 = Figure()
# gp = fhn.spatial_dim
# ax1 = Axis(fig2[1, 1], xlabel="t", ylabel="x", title="x1")
# hm1 = CairoMakie.heatmap!(ax1, fhn.tspan[1:DS:end], fhn.xspan, X[1:gp, :]')
# CairoMakie.Colorbar(fig2[1, 2], hm1)
# ax2 = Axis(fig2[1, 3], xlabel="t", ylabel="x", title="x2")
# hm2 = CairoMakie.heatmap!(ax2, fhn.tspan[1:DS:end], fhn.xspan, X[gp+1:end, :]')
# CairoMakie.Colorbar(fig2[1, 4], hm2)
# display(fig2)