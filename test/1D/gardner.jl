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
    Ucnab = gardner.integrate_model(
        gardner.tspan, gardner.IC, Ubc;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB, const_stepsize=true,
    )
    @test size(Ucnab) == (gardner.spatial_dim, gardner.time_dim)
    Usie = gardner.integrate_model(
        gardner.tspan, gardner.IC, Ubc;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SIE,
    )
    @test size(Usie) == (gardner.spatial_dim, gardner.time_dim)

    # Fast SIE / CNAB (Dirichlet; A bakes Δt into boundary rows → factorized fallback)
    solver_cn = pomoreda.build_fast_solver(gardner, gardner.params; scheme=:CN)
    @test solver_cn isa pomoreda.FactorizedSolver
    solver_be = pomoreda.build_fast_solver(gardner, gardner.params; scheme=:BE)
    @test solver_be isa pomoreda.FactorizedSolver

    Ufast_cnab = pomoreda.integrate_model_fast(gardner, solver_cn, gardner.tspan, gardner.IC, Ubc;
                                                quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                                integrator_type=:CNAB)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    Ufast_sie = pomoreda.integrate_model_fast(gardner, solver_be, gardner.tspan, gardner.IC, Ubc;
                                               quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                               integrator_type=:SIE)
    @test agrees_until_nan(Ufast_sie, Usie)
end


@testset "Gardner equation (periodic, fast SIE/CNAB)" begin
    Ω = (0.0, 3.0); Nx = 2^7; dt = 1e-3
    g = pomoreda.GardnerModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c => 5), BC=:periodic,
    )
    g.IC = 2 * cos.(2π * g.xspan / (Ω[2] - Ω[1]))
    A, F, E = g.finite_diff_model(g, g.params)
    Ucnab = g.integrate_model(g.tspan, g.IC; linear_matrix=A, quadratic_matrix=F, cubic_matrix=E,
                               system_input=false, integrator_type=:CNAB, const_stepsize=true)
    Usie = g.integrate_model(g.tspan, g.IC; linear_matrix=A, quadratic_matrix=F, cubic_matrix=E,
                              system_input=false, integrator_type=:SIE)
    solver_cn = pomoreda.build_fast_solver(g, g.params; scheme=:CN)
    solver_be = pomoreda.build_fast_solver(g, g.params; scheme=:BE)
    @test solver_cn isa pomoreda.FastCirculant1DSolver
    @test solver_be isa pomoreda.FastCirculant1DSolver
    Ufast_cnab = pomoreda.integrate_model_fast(g, solver_cn, g.tspan, g.IC;
                                                quadratic_matrix=F, cubic_matrix=E,
                                                integrator_type=:CNAB)
    Ufast_sie = pomoreda.integrate_model_fast(g, solver_be, g.tspan, g.IC;
                                               quadratic_matrix=F, cubic_matrix=E,
                                               integrator_type=:SIE)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    @test agrees_until_nan(Ufast_sie, Usie)
end