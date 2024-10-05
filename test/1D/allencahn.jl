@testset "Allen-Cahn equation" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    Ω = (0.0, 2.0)
    T = (0.0, 3.0)
    Nx = 2^7
    dt = 1e-3
    allencahn = pomoreda.AllenCahnModel(
        spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
        params=Dict(:μ => 0.01, :ϵ => 1.0), BC=:periodic
    )
    DS = 10
    allencahn.IC = exp.(-10 * cos.(π * allencahn.xspan).^2)

    #==================#
    ## Model Operators
    #==================#
    A, E = allencahn.finite_diff_model(allencahn, allencahn.params)

    #============#
    ## Integrate
    #============#
    U = allencahn.integrate_model(
        allencahn.tspan, allencahn.IC; 
        linear_matrix=A, cubic_matrix=E, system_input=false,
        integrator_type=:CNAB
    )
    @test size(U) == (allencahn.spatial_dim, allencahn.time_dim)
    U = allencahn.integrate_model(
        allencahn.tspan, allencahn.IC; 
        linear_matrix=A, cubic_matrix=E, system_input=false,
        integrator_type=:SICN
    )
    @test size(U) == (allencahn.spatial_dim, allencahn.time_dim)

    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (-1.0, 1.0)
    T = (0.0, 3.0)
    Nx = 2^7
    dt = 1e-3
    allencahn = pomoreda.AllenCahnModel(
        spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
        params=Dict(:μ => 0.001, :ϵ => 1.0), BC=:dirichlet
    )
    DS = 10
    allencahn.IC = 0.53*allencahn.xspan + 0.47*sin.(-1.5π * allencahn.xspan)
    Ubc1 = ones(1,allencahn.time_dim)
    Ubc2 = -ones(1,allencahn.time_dim)
    Ubc = [Ubc1; Ubc2]

    #==================#
    ## Model Operators
    #==================#
    A, E, B = allencahn.finite_diff_model(allencahn, allencahn.params)

    #============#
    ## Integrate
    #============#
    U = allencahn.integrate_model(
        allencahn.tspan, allencahn.IC, Ubc; 
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SICN,
    )
    @test size(U) == (allencahn.spatial_dim, allencahn.time_dim)
    U = allencahn.integrate_model(
        allencahn.tspan, allencahn.IC, Ubc; 
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB,
    )
    @test size(U) == (allencahn.spatial_dim, allencahn.time_dim)
end