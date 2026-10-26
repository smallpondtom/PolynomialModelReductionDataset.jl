@testset "Nagumo equation (Dirichlet BC, bistability)" begin
    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0.0, 20.0)
    T = (0.0, 40.0)
    Nx = 2^7
    dt = 1e-2
    nagumo = pomoreda.NagumoModel(
        spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
        params=Dict(:ν => 1.0, :a => 0.3), BC=:dirichlet
    )
    x = nagumo.xspan
    xc = 0.5 * (Ω[1] + Ω[2])

    #==================#
    ## Model Operators
    #==================#
    A, F, E, B = nagumo.finite_diff_model(nagumo, nagumo.params)
    @test size(A) == (nagumo.spatial_dim, nagumo.spatial_dim)
    @test size(F, 2) == nagumo.spatial_dim * (nagumo.spatial_dim + 1) ÷ 2
    @test size(E, 2) == nagumo.spatial_dim * (nagumo.spatial_dim + 1) * (nagumo.spatial_dim + 2) ÷ 6
    @test size(B) == (nagumo.spatial_dim, 2)

    # Reaction operators reproduce u(u-a)(1-u) pointwise (A carries the -a u part)
    u = randn(nagumo.spatial_dim)
    a = nagumo.params[:a]
    @test pomoreda.Nagumo.reaction_term(u, F, E) ≈ (1 + a) * u.^2 - u.^3

    #=============================================#
    ## Small bump: inside the basin of u ≡ 0
    #=============================================#
    nagumo.IC = 0.25 * exp.(-((x .- xc) / 4.0).^2)
    Ucnab = nagumo.integrate_model(
        nagumo.tspan, nagumo.IC;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E,
        system_input=false, integrator_type=:CNAB
    )
    @test size(Ucnab) == (nagumo.spatial_dim, nagumo.time_dim)
    @test maximum(abs, Ucnab[:, end]) < 1e-3

    Usicn = nagumo.integrate_model(
        nagumo.tspan, nagumo.IC;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E,
        system_input=false, integrator_type=:SICN
    )
    @test size(Usicn) == (nagumo.spatial_dim, nagumo.time_dim)
    @test maximum(abs, Usicn[:, end]) < 1e-3

    # Fast solver agrees with the reference integrator
    solver = pomoreda.build_fast_solver(nagumo, nagumo.params; scheme=:CN)
    @test solver isa pomoreda.FastSymTridiagSolver
    Ufast_cnab = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC;
                                               quadratic_matrix=F, cubic_matrix=E,
                                               integrator_type=:CNAB)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    Ufast_sicn = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC;
                                               quadratic_matrix=F, cubic_matrix=E,
                                               integrator_type=:SICN)
    @test agrees_until_nan(Ufast_sicn, Usicn)

    #=============================================#
    ## Large bump: escapes to the plateau near 1
    #=============================================#
    nagumo.IC = 1.0 * exp.(-((x .- xc) / 4.0).^2)
    Ubc = zeros(2, nagumo.time_dim)  # homogeneous Dirichlet values through B
    Ucnab = nagumo.integrate_model(
        nagumo.tspan, nagumo.IC, Ubc;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB
    )
    @test maximum(Ucnab[:, end]) > 0.9
    Usicn = nagumo.integrate_model(
        nagumo.tspan, nagumo.IC, Ubc;
        linear_matrix=A, quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SICN
    )
    @test maximum(Usicn[:, end]) > 0.9

    Ufast_cnab = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC, Ubc;
                                               quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                               integrator_type=:CNAB)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    Ufast_sicn = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC, Ubc;
                                               quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                               integrator_type=:SICN)
    @test agrees_until_nan(Ufast_sicn, Usicn)
end


@testset "Nagumo equation (periodic, fast SICN/CNAB)" begin
    Ω = (0.0, 20.0); T = (0.0, 10.0); Nx = 2^7; dt = 1e-2
    nagumo = pomoreda.NagumoModel(
        spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
        params=Dict(:ν => 1.0, :a => 0.3), BC=:periodic,
    )
    A, F, E = nagumo.finite_diff_model(nagumo, nagumo.params)
    @test size(A) == (nagumo.spatial_dim, nagumo.spatial_dim)

    # Spatially uniform data obeys the ODE u' = u(u-a)(1-u): below a decays, above a grows
    nagumo.IC = 0.2 * ones(nagumo.spatial_dim)
    Ulow = nagumo.integrate_model(nagumo.tspan, nagumo.IC; linear_matrix=A, quadratic_matrix=F,
                                  cubic_matrix=E, system_input=false, integrator_type=:CNAB)
    @test maximum(abs, Ulow[:, end]) < 0.05
    nagumo.IC = 0.5 * ones(nagumo.spatial_dim)
    Uhigh = nagumo.integrate_model(nagumo.tspan, nagumo.IC; linear_matrix=A, quadratic_matrix=F,
                                   cubic_matrix=E, system_input=false, integrator_type=:CNAB)
    @test minimum(Uhigh[:, end]) > 0.95

    nagumo.IC = 0.6 .+ 0.3 * sin.(2π * nagumo.xspan / (Ω[2] - Ω[1]))
    Ucnab = nagumo.integrate_model(nagumo.tspan, nagumo.IC; linear_matrix=A, quadratic_matrix=F,
                                   cubic_matrix=E, system_input=false, integrator_type=:CNAB)
    Usicn = nagumo.integrate_model(nagumo.tspan, nagumo.IC; linear_matrix=A, quadratic_matrix=F,
                                   cubic_matrix=E, system_input=false, integrator_type=:SICN)
    solver = pomoreda.build_fast_solver(nagumo, nagumo.params; scheme=:CN)
    @test solver isa pomoreda.FastCirculant1DSolver
    Ufast_cnab = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC;
                                               quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)
    Ufast_sicn = pomoreda.integrate_model_fast(nagumo, solver, nagumo.tspan, nagumo.IC;
                                               quadratic_matrix=F, cubic_matrix=E, integrator_type=:SICN)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    @test agrees_until_nan(Ufast_sicn, Usicn)
end
