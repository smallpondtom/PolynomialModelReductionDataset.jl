@testset "Gardner equation" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    Ω = (0.0, 3.0)
    Nx = 2^7; dt = 1e-3
    gardner = pomoreda.GardnerModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c => 5), BC=:periodic,
    )
    DS = 10
    gardner.IC = 2 * cos.(2π * gardner.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * gardner.xspan / (Ω[2] - Ω[1]))

    #==================#
    ## Model Operators
    #==================#
    A, F, E = gardner.finite_diff_model(gardner, gardner.params)

    #============#
    ## Integrate
    #============#
    U = gardner.integrate_model(
        gardner.tspan, gardner.IC; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, system_input=false,
        integrator_type=:CNAB
    )
    @test size(U) == (gardner.spatial_dim, gardner.time_dim)
    U = gardner.integrate_model(
        gardner.tspan, gardner.IC; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, system_input=false,
        integrator_type=:SIE
    )
    @test size(U) == (gardner.spatial_dim, gardner.time_dim)

    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0.0, 3.0)
    Nx = 2^7; dt = 1e-3
    gardner = pomoreda.GardnerModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c => 5), BC=:dirichlet,
    )
    DS = 100
    gardner.IC = 2 * cos.(2π * gardner.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * gardner.xspan / (Ω[2] - Ω[1]))
    Ubc1 = 0.5ones(1,gardner.time_dim)
    Ubc2 = -0.5ones(1,gardner.time_dim)
    Ubc = [Ubc1; Ubc2]

    #==================#
    ## Model Operators
    #==================#
    A, F, E, B = gardner.finite_diff_model(gardner, gardner.params)

    #============#
    ## Integrate
    #============#
    U = gardner.integrate_model(
        gardner.tspan, gardner.IC, Ubc; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB,
    )
    @test size(U) == (gardner.spatial_dim, gardner.time_dim)
    U = gardner.integrate_model(
        gardner.tspan, gardner.IC, Ubc; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SIE,
    )
    @test size(U) == (gardner.spatial_dim, gardner.time_dim)
end