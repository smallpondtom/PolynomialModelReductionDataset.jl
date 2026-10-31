@testset "Swift-Hohenberg SH23 (periodic BC, bistability)" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    # L = 8π holds exactly four critical wavelengths (k = 1); use Δx = L/Nx so the
    # discrete period equals L.
    Ω = (0.0, 8π)
    T = (0.0, 50.0)
    Nx = 2^7
    dt = 1e-2
    sh = pomoreda.SwiftHohenbergModel(
        spatial_domain=Ω, time_domain=T, Δx=(Ω[2]-Ω[1])/Nx, Δt=dt,
        params=Dict(:r => -0.2, :b => 1.8), BC=:periodic
    )
    x = sh.xspan
    @test sh.spatial_dim == Nx

    #==================#
    ## Model Operators
    #==================#
    A, F, E = sh.finite_diff_model(sh, sh.params)
    N = sh.spatial_dim
    @test size(A) == (N, N)
    @test size(F, 2) == N * (N + 1) ÷ 2
    @test size(E, 2) == N * (N + 1) * (N + 2) ÷ 6
    @test A ≈ A'

    # Linear operator reproduces the symbol r - (1 - k²)² on the critical mode k = 1
    r = sh.params[:r]
    @test A * cos.(x) ≈ (r - 0.0) * cos.(x) atol=1e-3

    # Reaction operators reproduce b u² - u³ pointwise
    u = randn(N)
    b = sh.params[:b]
    @test pomoreda.SwiftHohenberg.reaction_term(u, F, E) ≈ b * u.^2 - u.^3

    #=============================================#
    ## Small perturbation decays to u ≡ 0
    #=============================================#
    sh.IC = 0.2 * cos.(x)
    Ucnab = sh.integrate_model(sh.tspan, sh.IC; linear_matrix=A, quadratic_matrix=F,
                               cubic_matrix=E, system_input=false, integrator_type=:CNAB)
    @test size(Ucnab) == (sh.spatial_dim, sh.time_dim)
    @test maximum(abs, Ucnab[:, end]) < 1e-3
    Usicn = sh.integrate_model(sh.tspan, sh.IC; linear_matrix=A, quadratic_matrix=F,
                               cubic_matrix=E, system_input=false, integrator_type=:SICN)
    @test maximum(abs, Usicn[:, end]) < 1e-3

    solver = pomoreda.build_fast_solver(sh, sh.params; scheme=:CN)
    @test solver isa pomoreda.FastCirculant1DSolver
    Ufast_cnab = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC;
                                               quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)
    Ufast_sicn = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC;
                                               quadratic_matrix=F, cubic_matrix=E, integrator_type=:SICN)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    @test agrees_until_nan(Ufast_sicn, Usicn)

    #=============================================#
    ## Large perturbation saturates on the pattern
    #=============================================#
    sh.IC = 1.0 * cos.(x)
    Ucnab = sh.integrate_model(sh.tspan, sh.IC; linear_matrix=A, quadratic_matrix=F,
                               cubic_matrix=E, system_input=false, integrator_type=:CNAB)
    @test maximum(abs, Ucnab[:, end]) > 1.0
    Ufast_cnab = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC;
                                               quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
end


@testset "Swift-Hohenberg SH23 (hinged Dirichlet BC, fast SICN/CNAB)" begin
    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0.0, 8π)
    T = (0.0, 50.0)
    Nx = 2^7
    dt = 1e-2
    sh = pomoreda.SwiftHohenbergModel(
        spatial_domain=Ω, time_domain=T, Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
        params=Dict(:r => -0.2, :b => 1.8), BC=:dirichlet
    )
    x = sh.xspan
    xc = 0.5 * (Ω[1] + Ω[2])

    #==================#
    ## Model Operators
    #==================#
    A, F, E, B = sh.finite_diff_model(sh, sh.params)
    N = sh.spatial_dim
    @test size(A) == (N, N)
    @test size(B) == (N, 2)
    @test A ≈ A'
    @test maximum(eigvals(Symmetric(Matrix(A)))) < 0  # trivial state linearly stable for r < 0

    # A u + B g equals the full stencil on the grid extended by the ghost nodes
    # u_0 = g_L, u_{-1} = 2 g_L - u_1 (and likewise on the right), i.e. u = g, u_xx = 0
    Δx = sh.Δx
    r = sh.params[:r]
    u = randn(N); gL, gR = 0.7, -0.4
    ext = [2gL - u[1]; gL; u; gR; 2gR - u[end]]
    stencil = [(r - 1) * ext[i] - 2 * (ext[i-1] - 2ext[i] + ext[i+1]) / Δx^2 -
               (ext[i-2] - 4ext[i-1] + 6ext[i] - 4ext[i+1] + ext[i+2]) / Δx^4 for i in 3:N+2]
    @test A * u + B * [gL, gR] ≈ stencil

    solver = pomoreda.build_fast_solver(sh, sh.params; scheme=:CN)
    @test solver isa pomoreda.FastDenseSolver

    #=============================================#
    ## Small perturbation decays to u ≡ 0
    #=============================================#
    sh.IC = 0.2 * cos.(x .- xc) .* exp.(-((x .- xc) / 6).^2)
    Ucnab = sh.integrate_model(sh.tspan, sh.IC; linear_matrix=A, quadratic_matrix=F,
                               cubic_matrix=E, system_input=false, integrator_type=:CNAB)
    @test maximum(abs, Ucnab[:, end]) < 1e-3
    Usicn = sh.integrate_model(sh.tspan, sh.IC; linear_matrix=A, quadratic_matrix=F,
                               cubic_matrix=E, system_input=false, integrator_type=:SICN)
    Ufast_cnab = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC;
                                               quadratic_matrix=F, cubic_matrix=E, integrator_type=:CNAB)
    Ufast_sicn = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC;
                                               quadratic_matrix=F, cubic_matrix=E, integrator_type=:SICN)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    @test agrees_until_nan(Ufast_sicn, Usicn)

    #=============================================#
    ## Large perturbation saturates on the pattern
    #=============================================#
    sh.IC = 1.0 * cos.(x .- xc) .* exp.(-((x .- xc) / 6).^2)
    Ubc = zeros(2, sh.time_dim)  # homogeneous boundary values through B
    Ucnab = sh.integrate_model(sh.tspan, sh.IC, Ubc; linear_matrix=A, quadratic_matrix=F,
                               cubic_matrix=E, control_matrix=B, system_input=true, integrator_type=:CNAB)
    @test maximum(abs, Ucnab[:, end]) > 1.0
    Usicn = sh.integrate_model(sh.tspan, sh.IC, Ubc; linear_matrix=A, quadratic_matrix=F,
                               cubic_matrix=E, control_matrix=B, system_input=true, integrator_type=:SICN)
    @test maximum(abs, Usicn[:, end]) > 1.0
    Ufast_cnab = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC, Ubc;
                                               quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                               integrator_type=:CNAB)
    @test agrees_until_nan(Ufast_cnab, Ucnab)
    Ufast_sicn = pomoreda.integrate_model_fast(sh, solver, sh.tspan, sh.IC, Ubc;
                                               quadratic_matrix=F, cubic_matrix=E, control_matrix=B,
                                               integrator_type=:SICN)
    @test agrees_until_nan(Ufast_sicn, Usicn)
end
