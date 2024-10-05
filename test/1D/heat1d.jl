@testset "1D heat equation" begin 
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
    @test size(U) == (heat1d.spatial_dim, heat1d.time_dim)

    #=================#
    ## Backward Euler
    #=================#
    U = heat1d.integrate_model(
        heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
        system_input=true, integrator_type=:BackwardEuler
    )
    @test size(U) == (heat1d.spatial_dim, heat1d.time_dim)


    #=================#
    ## Crank-Nicolson
    #=================#
    U = heat1d.integrate_model(
        heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
        system_input=true, integrator_type=:CrankNicolson
    )
    @test size(U) == (heat1d.spatial_dim, heat1d.time_dim)
end