@testset "Modified Korteweg-de Vries-Burgers" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    Ω = (0.0, 3.0)
    Nx = 2^7; dt = 1e-3
    mKdV = pomoreda.ModifiedKortewegDeVriesBurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c => 0.1), BC=:periodic,
    )
    DS = 100
    mKdV.IC = 2 * cos.(2π * mKdV.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * mKdV.xspan / (Ω[2] - Ω[1]))

    #==================#
    ## Model Operators
    #==================#
    A, E = mKdV.finite_diff_model(mKdV, mKdV.params)

    #============#
    ## Integrate
    #============#
    U = mKdV.integrate_model(
        mKdV.tspan, mKdV.IC; 
        linear_matrix=A, cubic_matrix=E, system_input=false,
        integrator_type=:CNAB
    )
    @test size(U) == (mKdV.spatial_dim, mKdV.time_dim)
    U = mKdV.integrate_model(
        mKdV.tspan, mKdV.IC; 
        linear_matrix=A, cubic_matrix=E, system_input=false,
        integrator_type=:SIE
    )
    @test size(U) == (mKdV.spatial_dim, mKdV.time_dim)

    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0.0, 3.0)
    Nx = 2^7; dt = 1e-3
    mKdV = pomoreda.ModifiedKortewegDeVriesBurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c=> 0.1), BC=:dirichlet,
    )
    DS = 100
    mKdV.IC = 2 * cos.(2π * mKdV.xspan / (Ω[2] - Ω[1])) # + 0.5 * cos.(4π * mKdV.xspan / (Ω[2] - Ω[1]))
    Ubc1 = 0.5ones(1,mKdV.time_dim)
    Ubc2 = -0.5ones(1,mKdV.time_dim)
    Ubc = [Ubc1; Ubc2]

    #==================#
    ## Model Operators
    #==================#
    A, E, B = mKdV.finite_diff_model(mKdV, mKdV.params)

    #============#
    ## Integrate
    #============#
    U = mKdV.integrate_model(
        mKdV.tspan, mKdV.IC, Ubc; 
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB,
    )
    @test size(U) == (mKdV.spatial_dim, mKdV.time_dim)
    U = mKdV.integrate_model(
        mKdV.tspan, mKdV.IC, Ubc; 
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SIE,
    )
    @test size(U) == (mKdV.spatial_dim, mKdV.time_dim)
end