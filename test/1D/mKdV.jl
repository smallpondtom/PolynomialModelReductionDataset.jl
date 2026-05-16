@testset "Modified Vorteweg-de Vries equation" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    Ω = (0.0, 3.0)
    Nx = 2^7; dt = 1e-3
    mKdV = pomoreda.ModifiedKortewegDeVriesModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3), BC=:periodic,
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
    mKdV = pomoreda.ModifiedKortewegDeVriesModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3), BC=:dirichlet,
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
    Ucnab = mKdV.integrate_model(
        mKdV.tspan, mKdV.IC, Ubc;
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB, const_stepsize=true,
    )
    @test size(Ucnab) == (mKdV.spatial_dim, mKdV.time_dim)
    Usie = mKdV.integrate_model(
        mKdV.tspan, mKdV.IC, Ubc;
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SIE,
    )
    @test size(Usie) == (mKdV.spatial_dim, mKdV.time_dim)

    # Fast SIE / CNAB (Dirichlet → FactorizedSolver)
    solver_cn = pomoreda.build_fast_solver(mKdV, mKdV.params; scheme=:CN)
    solver_be = pomoreda.build_fast_solver(mKdV, mKdV.params; scheme=:BE)
    @test solver_cn isa pomoreda.FactorizedSolver
    Ufast_cnab = pomoreda.integrate_model_fast(mKdV, solver_cn, mKdV.tspan, mKdV.IC, Ubc;
                                                cubic_matrix=E, control_matrix=B,
                                                integrator_type=:CNAB)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    Ufast_sie = pomoreda.integrate_model_fast(mKdV, solver_be, mKdV.tspan, mKdV.IC, Ubc;
                                               cubic_matrix=E, control_matrix=B,
                                               integrator_type=:SIE)
    @test agrees_until_nan(Ufast_sie, Usie)
end


@testset "mKdV equation (periodic, fast SIE/CNAB)" begin
    Ω = (0.0, 3.0); Nx = 2^7; dt = 1e-3
    mk = pomoreda.ModifiedKortewegDeVriesModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3), BC=:periodic,
    )
    mk.IC = 2 * cos.(2π * mk.xspan / (Ω[2] - Ω[1]))
    A, E = mk.finite_diff_model(mk, mk.params)
    Ucnab = mk.integrate_model(mk.tspan, mk.IC; linear_matrix=A, cubic_matrix=E,
                                system_input=false, integrator_type=:CNAB, const_stepsize=true)
    Usie = mk.integrate_model(mk.tspan, mk.IC; linear_matrix=A, cubic_matrix=E,
                               system_input=false, integrator_type=:SIE)
    solver_cn = pomoreda.build_fast_solver(mk, mk.params; scheme=:CN)
    solver_be = pomoreda.build_fast_solver(mk, mk.params; scheme=:BE)
    @test solver_cn isa pomoreda.FastCirculant1DSolver
    Ufast_cnab = pomoreda.integrate_model_fast(mk, solver_cn, mk.tspan, mk.IC;
                                                cubic_matrix=E, integrator_type=:CNAB)
    Ufast_sie = pomoreda.integrate_model_fast(mk, solver_be, mk.tspan, mk.IC;
                                               cubic_matrix=E, integrator_type=:SIE)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    @test agrees_until_nan(Ufast_sie, Usie)
end