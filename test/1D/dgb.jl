@testset "Damped Gardner-Burgers equation" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    Ω = (0.0, 3.0)
    Nx = 2^7; dt = 1e-3
    dgb = pomoreda.DampedGardnerBurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c => 5, :d => 0.2, :e => 0.5), BC=:periodic,
    )
    DS = 10
    dgb.IC = 2 * cos.(2π * dgb.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * dgb.xspan / (Ω[2] - Ω[1]))

    #==================#
    ## Model Operators
    #==================#
    A, F, E = dgb.finite_diff_model(dgb, dgb.params)

    #============#
    ## Integrate
    #============#
    U = dgb.integrate_model(
        dgb.tspan, dgb.IC; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, system_input=false,
        integrator_type=:CNAB
    )
    @test size(U) == (dgb.spatial_dim, dgb.time_dim)
    U = dgb.integrate_model(
        dgb.tspan, dgb.IC; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, system_input=false,
        integrator_type=:SIE
    )
    @test size(U) == (dgb.spatial_dim, dgb.time_dim)

    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0.0, 3.0)
    Nx = 2^7; dt = 1e-3
    dgb = pomoreda.DampedGardnerBurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c => 5, :d => 0.2, :e => 0.5), BC=:dirichlet,
    )
    DS = 100
    dgb.IC = 2 * cos.(2π * dgb.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * dgb.xspan / (Ω[2] - Ω[1]))
    Ubc1 = 0.5ones(1,dgb.time_dim)
    Ubc2 = -0.5ones(1,dgb.time_dim)
    Ubc = [Ubc1; Ubc2]

    #==================#
    ## Model Operators
    #==================#
    A, F, E, B = dgb.finite_diff_model(dgb, dgb.params)

    #============#
    ## Integrate
    #============#
    U = dgb.integrate_model(
        dgb.tspan, dgb.IC, Ubc; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SIE,
    )
    @test size(U) == (dgb.spatial_dim, dgb.time_dim)
    U = dgb.integrate_model(
        dgb.tspan, dgb.IC, Ubc; 
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB,
    )
    @test size(U) == (dgb.spatial_dim, dgb.time_dim)
end