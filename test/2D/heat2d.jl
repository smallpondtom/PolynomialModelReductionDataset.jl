@testset "2D Heat equation" begin
    #========#
    ## Model
    #========#
    Ω = ((0.0, 1.0), (0.0, 1.0))
    Nx = 2^4
    Ny = 2^4
    heat2d = pomoreda.Heat2DModel(
        spatial_domain=Ω, time_domain=(0,2), 
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
    @test size(U) == (prod(heat2d.spatial_dim), heat2d.time_dim)

    #=================#
    ## Backward Euler
    #=================#
    U = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B,
         system_input=true, integrator_type=:BackwardEuler
    )
    @test size(U) == (prod(heat2d.spatial_dim), heat2d.time_dim)

    #=================#
    ## Crank-Nicolson
    #=================#
    U = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B, 
        system_input=true, integrator_type=:CrankNicolson
    )
    @test size(U) == (prod(heat2d.spatial_dim), heat2d.time_dim)
end