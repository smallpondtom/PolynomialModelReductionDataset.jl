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
    Ube = heat1d.integrate_model(
        heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
        system_input=true, integrator_type=:BackwardEuler
    )
    @test size(Ube) == (heat1d.spatial_dim, heat1d.time_dim)


    #=================#
    ## Crank-Nicolson
    #=================#
    Ucn = heat1d.integrate_model(
        heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
        system_input=true, integrator_type=:CrankNicolson
    )
    @test size(Ucn) == (heat1d.spatial_dim, heat1d.time_dim)

    #======================#
    ## Fast BE / CN solver
    #======================#
    solver_be = pomoreda.build_fast_solver(heat1d, heat1d.diffusion_coeffs;
                                            scheme=:BE, same_on_both_ends=true)
    @test solver_be isa pomoreda.FastSymTridiagSolver
    Ufast_be = pomoreda.integrate_model_fast(heat1d, solver_be, heat1d.tspan, heat1d.IC, Ubc;
                                              control_matrix=B, scheme=:BE)
    @test size(Ufast_be) == (heat1d.spatial_dim, heat1d.time_dim)
    @test Ufast_be ≈ Ube

    solver_cn = pomoreda.build_fast_solver(heat1d, heat1d.diffusion_coeffs;
                                            scheme=:CN, same_on_both_ends=true)
    Ufast_cn = pomoreda.integrate_model_fast(heat1d, solver_cn, heat1d.tspan, heat1d.IC, Ubc;
                                              control_matrix=B, scheme=:CN)
    @test Ufast_cn ≈ Ucn
end


@testset "1D heat equation (periodic, fast solver)" begin
    Nx = 2^7; dt = 1e-3
    heat1d = pomoreda.Heat1DModel(
        spatial_domain=(0.0, 1.0), time_domain=(0.0, 1.0),
        Δx=1/Nx, Δt=dt, diffusion_coeffs=0.1, BC=:periodic,
    )
    heat1d.IC = sin.(2π * heat1d.xspan)
    A = heat1d.finite_diff_model(heat1d, heat1d.diffusion_coeffs)

    Ube = heat1d.integrate_model(heat1d.tspan, heat1d.IC; linear_matrix=A,
                                  system_input=false, integrator_type=:BackwardEuler)
    Ucn = heat1d.integrate_model(heat1d.tspan, heat1d.IC; linear_matrix=A,
                                  system_input=false, integrator_type=:CrankNicolson)

    solver_be = pomoreda.build_fast_solver(heat1d, heat1d.diffusion_coeffs; scheme=:BE)
    @test solver_be isa pomoreda.FastCirculant1DSolver
    Ufast_be = pomoreda.integrate_model_fast(heat1d, solver_be, heat1d.tspan, heat1d.IC; scheme=:BE)
    @test Ufast_be ≈ Ube

    solver_cn = pomoreda.build_fast_solver(heat1d, heat1d.diffusion_coeffs; scheme=:CN)
    Ufast_cn = pomoreda.integrate_model_fast(heat1d, solver_cn, heat1d.tspan, heat1d.IC; scheme=:CN)
    @test Ufast_cn ≈ Ucn
end