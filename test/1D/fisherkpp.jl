@testset "Fisher-KPP equation" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    Ω = (0.0, 2.0)
    T = (0.0, 3.0)
    Nx = 2^7
    dt = 1e-3
    fisherkpp = pomoreda.FisherKPPModel(
        spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
        params=Dict(:D => 0.001, :r => 0.1), BC=:periodic
    )
    DS = 10
    fisherkpp.IC = exp.(-10 * cos.(π * fisherkpp.xspan).^2)

    #==================#
    ## Model Operators
    #==================#
    A, F = fisherkpp.finite_diff_model(fisherkpp, fisherkpp.params)

    #============#
    ## Integrate
    #============#
    U = fisherkpp.integrate_model(
        fisherkpp.tspan, fisherkpp.IC; 
        linear_matrix=A, quadratic_matrix=F, system_input=false,
        integrator_type=:CNAB
    )
    @test size(U) == (fisherkpp.spatial_dim, fisherkpp.time_dim)
    U = fisherkpp.integrate_model(
        fisherkpp.tspan, fisherkpp.IC; 
        linear_matrix=A, quadratic_matrix=F, system_input=false,
        integrator_type=:SICN
    )
    @test size(U) == (fisherkpp.spatial_dim, fisherkpp.time_dim)

    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0, 3)  # Choose integers
    T = (0.0, 5.0)
    Nx = 2^7
    dt = 1e-3
    fisherkpp = pomoreda.FisherKPPModel(
        spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
        params=Dict(:D => 1.0, :r => 1.0), BC=:dirichlet
    )
    DS = 10

    # Create piecewise IC
    a, b, c = (sort ∘ rand)(1:fisherkpp.spatial_dim, 3)
    seg1 = ones(length(fisherkpp.xspan[1:a]))
    seg2 = zeros(length(fisherkpp.xspan[a+1:b]))
    seg3 = rand(0.1:0.1:0.5) * ones(length(fisherkpp.xspan[b+1:c]))
    seg4 = zeros(length(fisherkpp.xspan[c+1:end]))
    fisherkpp.IC = [seg1; seg2; seg3; seg4]

    # Boundary conditions
    Ubc1 = ones(1,fisherkpp.time_dim)
    Ubc2 = zeros(1,fisherkpp.time_dim)
    Ubc = [Ubc1; Ubc2]

    #==================#
    ## Model Operators
    #==================#
    A, F, B = fisherkpp.finite_diff_model(fisherkpp, fisherkpp.params)

    #============#
    ## Integrate
    #============#
    U = fisherkpp.integrate_model(
        fisherkpp.tspan, fisherkpp.IC, Ubc; 
        linear_matrix=A, quadratic_matrix=F, control_matrix=B,
        system_input=true, integrator_type=:CNAB,
    )
    @test size(U) == (fisherkpp.spatial_dim, fisherkpp.time_dim)
    U = fisherkpp.integrate_model(
        fisherkpp.tspan, fisherkpp.IC, Ubc; 
        linear_matrix=A, quadratic_matrix=F, control_matrix=B,
        system_input=true, integrator_type=:SICN,
    )
    @test size(U) == (fisherkpp.spatial_dim, fisherkpp.time_dim)
end