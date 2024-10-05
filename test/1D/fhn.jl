@testset "FitzHugh-Nagumo equation" begin
    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0.0, 1.0); dt = 1e-4; Nx = 2^9
    fhn = pomoreda.FitzHughNagumoModel(
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
    U = fhn.integrate_model(fhn.tspan, fhn.IC, g; functional=fom)
    @test size(U) == (fhn.spatial_dim*2, fhn.time_dim)

    #=========================#
    ## Lifted Model Operators
    #=========================#
    A, B, C, H, N, K = fhn.lifted_finite_diff_model(fhn.spatial_dim, fhn.spatial_domain[2])
end