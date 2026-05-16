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
    Usie = dgb.integrate_model(
        dgb.tspan, dgb.IC, Ubc;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SIE,
    )
    @test size(Usie) == (dgb.spatial_dim, dgb.time_dim)
    Ucnab = dgb.integrate_model(
        dgb.tspan, dgb.IC, Ubc;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB, const_stepsize=true,
    )
    @test size(Ucnab) == (dgb.spatial_dim, dgb.time_dim)

    # Fast SIE / CNAB (Dirichlet → FactorizedSolver fallback)
    solver_cn = pomoreda.build_fast_solver(dgb, dgb.params; scheme=:CN)
    solver_be = pomoreda.build_fast_solver(dgb, dgb.params; scheme=:BE)
    @test solver_cn isa pomoreda.FactorizedSolver
    Ufast_sie = pomoreda.integrate_model_fast(dgb, solver_be, dgb.tspan, dgb.IC, Ubc;
                                               quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                               integrator_type=:SIE)
    @test Ufast_sie ≈ Usie
    Ufast_cnab = pomoreda.integrate_model_fast(dgb, solver_cn, dgb.tspan, dgb.IC, Ubc;
                                                quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                                integrator_type=:CNAB)
    @test Ufast_cnab ≈ Ucnab
end


@testset "DGB equation (periodic, fast SIE/CNAB)" begin
    Ω = (0.0, 3.0); Nx = 2^7; dt = 1e-3
    dgb = pomoreda.DampedGardnerBurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        params=Dict(:a => 1, :b => 3, :c => 5, :d => 0.2, :e => 0.5), BC=:periodic,
    )
    dgb.IC = 2 * cos.(2π * dgb.xspan / (Ω[2] - Ω[1]))
    A, F, E = dgb.finite_diff_model(dgb, dgb.params)
    Ucnab = dgb.integrate_model(dgb.tspan, dgb.IC; linear_matrix=A, quadratic_matrix=F, cubic_matrix=E,
                                 system_input=false, integrator_type=:CNAB, const_stepsize=true)
    Usie = dgb.integrate_model(dgb.tspan, dgb.IC; linear_matrix=A, quadratic_matrix=F, cubic_matrix=E,
                                system_input=false, integrator_type=:SIE)
    solver_cn = pomoreda.build_fast_solver(dgb, dgb.params; scheme=:CN)
    solver_be = pomoreda.build_fast_solver(dgb, dgb.params; scheme=:BE)
    @test solver_cn isa pomoreda.FastCirculant1DSolver
    Ufast_cnab = pomoreda.integrate_model_fast(dgb, solver_cn, dgb.tspan, dgb.IC;
                                                quadratic_matrix=F, cubic_matrix=E,
                                                integrator_type=:CNAB)
    Ufast_sie = pomoreda.integrate_model_fast(dgb, solver_be, dgb.tspan, dgb.IC;
                                               quadratic_matrix=F, cubic_matrix=E,
                                               integrator_type=:SIE)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    @test agrees_until_nan(Ufast_sie, Usie)
end