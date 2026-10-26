"""
Bistable Nagumo equation example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset
const pomoreda = PolynomialModelReductionDataset

#=======================#
## Model (Dirichlet BC)
#=======================#
Ω = (0.0, 20.0)
T = (0.0, 40.0)
Nx = 2^8
dt = 1e-2
nagumo = pomoreda.NagumoModel(
    spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
    params=Dict(:ν => 1.0, :a => 0.3), BC=:dirichlet
)
DS = 10
x = nagumo.xspan
xc = 0.5 * (Ω[1] + Ω[2])

#==================#
## Model Operators
#==================#
A, F, E, B = nagumo.finite_diff_model(nagumo, nagumo.params)
solver = pomoreda.build_fast_solver(nagumo, nagumo.params; scheme=:CN)

#=====================================================#
## Integrate: two bumps on either side of the threshold
#=====================================================#
# Small bump decays to the zero state
nagumo.IC = 0.25 * exp.(-((x .- xc) / 4.0).^2)
U_small = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC;
                                        quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)

# Large bump escapes to the plateau state near u ≡ 1
nagumo.IC = 1.0 * exp.(-((x .- xc) / 4.0).^2)
U_large = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC;
                                        quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)

#================#
## Plot Solution
#================#
fig1 = Figure(size=(900, 400))
ax1 = Axis(fig1[1, 1], xlabel=L"x", ylabel=L"t", title="Small bump: decays to 0")
hm1 = heatmap!(ax1, x, nagumo.tspan[1:DS:end], U_small[:, 1:DS:end], colorrange=(0, 1))
ax2 = Axis(fig1[1, 2], xlabel=L"x", ylabel=L"t", title="Large bump: escapes to the plateau")
hm2 = heatmap!(ax2, x, nagumo.tspan[1:DS:end], U_large[:, 1:DS:end], colorrange=(0, 1))
Colorbar(fig1[1, 3], hm2)
display(fig1)

#=====================================================#
## Threshold amplitude A*(w) for Gaussian bumps
#=====================================================#
# Bisection in the amplitude for a few bump widths. The curve A*(w) is a
# one-parameter cross-section of the boundary of the basin of u ≡ 0.
escapes(u0) = maximum(pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, u0;
                      quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)[:, end]) > 0.5
widths = [1.5, 2.0, 3.0, 4.0, 6.0]
Astar = similar(widths)
for (k, w) in enumerate(widths)
    lo, hi = 0.0, 2.0
    for _ in 1:12
        mid = 0.5 * (lo + hi)
        escapes(mid * exp.(-((x .- xc) / w).^2)) ? (hi = mid) : (lo = mid)
    end
    Astar[k] = 0.5 * (lo + hi)
end

fig2 = Figure(size=(500, 350))
ax = Axis(fig2[1, 1], xlabel="bump width w", ylabel="threshold amplitude A*")
scatterlines!(ax, widths, Astar)
hlines!(ax, [nagumo.params[:a]], linestyle=:dash, color=:gray, label=L"u = a")
axislegend(ax)
display(fig2)

#======================#
## Model (Periodic BC)
#======================#
Ω = (0.0, 20.0)
T = (0.0, 20.0)
nagumo = pomoreda.NagumoModel(
    spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
    params=Dict(:ν => 1.0, :a => 0.3), BC=:periodic
)
nagumo.IC = 0.6 .+ 0.3 * sin.(2π * nagumo.xspan / (Ω[2] - Ω[1]))
A, F, E = nagumo.finite_diff_model(nagumo, nagumo.params)
solver = pomoreda.build_fast_solver(nagumo, nagumo.params; scheme=:CN)
U = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC;
                                  quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)

fig3, ax, hm = CairoMakie.heatmap(nagumo.xspan, nagumo.tspan[1:DS:end], U[:, 1:DS:end])
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig3[1, 2], hm)
display(fig3)
