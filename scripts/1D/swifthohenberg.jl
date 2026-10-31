"""
Subcritical Swift-Hohenberg (SH23) equation example
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using PolynomialModelReductionDataset
const pomoreda = PolynomialModelReductionDataset

#======================#
## Model (Periodic BC)
#======================#
# Four critical wavelengths (k = 1) on a periodic domain. Use Δx = L/Nx so that
# the discrete period equals L.
Ω = (0.0, 8π)
T = (0.0, 100.0)
Nx = 2^7
dt = 1e-2
b = 1.8
r = -0.2  # inside the bistable window r_sn ≈ -0.37 < r < 0 (for b = 1.8)
sh = pomoreda.SwiftHohenbergModel(
    spatial_domain=Ω, time_domain=T, Δx=(Ω[2]-Ω[1])/Nx, Δt=dt,
    params=Dict(:r => r, :b => b), BC=:periodic
)
DS = 10
x = sh.xspan

#==================#
## Model Operators
#==================#
A, F, E = sh.finite_diff_model(sh, sh.params)
solver = pomoreda.build_fast_solver(sh, sh.params; scheme=:CN)
integrate(u0) = pomoreda.integrate_model_fast(sh, solver, sh.tspan, u0;
                                              quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)

#===============================================================#
## Integrate: two perturbations on either side of the threshold
#===============================================================#
U_small = integrate(0.2 * cos.(x))   # decays to u ≡ 0
U_large = integrate(1.0 * cos.(x))   # saturates on the periodic pattern

#================#
## Plot Solution
#================#
crange = (-1.5, 1.5)
fig1 = Figure(size=(900, 400))
ax1 = Axis(fig1[1, 1], xlabel=L"x", ylabel=L"t", title="Small perturbation: decays to 0")
heatmap!(ax1, x, sh.tspan[1:DS:end], U_small[:, 1:DS:end], colorrange=crange, colormap=:balance)
ax2 = Axis(fig1[1, 2], xlabel=L"x", ylabel=L"t", title="Large perturbation: saturates on the pattern")
hm = heatmap!(ax2, x, sh.tspan[1:DS:end], U_large[:, 1:DS:end], colorrange=crange, colormap=:balance)
Colorbar(fig1[1, 3], hm)
display(fig1)

#=====================================================#
## Threshold amplitude A* of the critical mode A cos(x)
#=====================================================#
# Bisection in the amplitude for several values of r. A*(r) is a cross-section
# of the boundary of the basin of u ≡ 0 along the critical mode; it goes to
# zero as r → 0⁻ and the bistable window closes at the saddle-node r_sn.
escapes(u0) = maximum(abs, integrate(u0)[:, end]) > 0.5
rs = [-0.05, -0.1, -0.15, -0.2, -0.25, -0.3]
Astar = similar(rs)
for (k, rk) in enumerate(rs)
    sh.params[:r] = rk
    global A, F, E = sh.finite_diff_model(sh, sh.params)
    global solver = pomoreda.build_fast_solver(sh, sh.params; scheme=:CN)
    lo, hi = 0.0, 1.5
    for _ in 1:12
        mid = 0.5 * (lo + hi)
        escapes(mid * cos.(x)) ? (hi = mid) : (lo = mid)
    end
    Astar[k] = 0.5 * (lo + hi)
    @info "r = $rk: threshold amplitude A* ≈ $(round(Astar[k], digits=4))"
end
sh.params[:r] = r

fig2 = Figure(size=(500, 350))
ax = Axis(fig2[1, 1], xlabel=L"bifurcation parameter $r$", ylabel=L"threshold amplitude $A^*$")
scatterlines!(ax, rs, Astar)
display(fig2)

#========================================#
## Model (hinged Dirichlet BC, u = u_xx = 0)
#========================================#
sh = pomoreda.SwiftHohenbergModel(
    spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
    params=Dict(:r => r, :b => b), BC=:dirichlet
)
x = sh.xspan
xc = 0.5 * (Ω[1] + Ω[2])
A, F, E, B = sh.finite_diff_model(sh, sh.params)
solver = pomoreda.build_fast_solver(sh, sh.params; scheme=:CN)   # FastDenseSolver (pentadiagonal A)

# Localized wave packet that invades the domain
sh.IC = 1.0 * cos.(x .- xc) .* exp.(-((x .- xc) / 6).^2)
U = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC;
                                  quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)

fig3, ax, hm = CairoMakie.heatmap(x, sh.tspan[1:DS:end], U[:, 1:DS:end], colorrange=crange, colormap=:balance)
ax.xlabel = L"x"
ax.ylabel = L"t"
CairoMakie.Colorbar(fig3[1, 2], hm)
display(fig3)
