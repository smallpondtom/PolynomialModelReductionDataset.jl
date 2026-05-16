@testset "2D Allen-Cahn equation" begin
    #========================#
    ## Model (Periodic BC)
    #========================#
    Ω = ((0.0, 1.0), (0.0, 1.0))
    Nx = 2^4
    Ny = 2^4
    allencahn2d = pomoreda.AllenCahn2DModel(
        spatial_domain=Ω, time_domain=(0,0.5), 
        Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
        params=Dict(:μ => 0.01, :ϵ => 1.0), BC=(:periodic, :periodic)
    )
    
    # Star-shaped initial condition (centered at 0.5, 0.5)
    xgrid0 = allencahn2d.xspan
    ygrid0 = allencahn2d.yspan
    X_grid = repeat(xgrid0', length(ygrid0), 1)    # size (Ny, Nx)
    Y_grid = repeat(ygrid0, 1, length(xgrid0))     # size (Ny, Nx)

    # Distance from center
    r_actual = sqrt.((X_grid .- 0.5).^2 .+ (Y_grid .- 0.5).^2)

    # Circle initial condition using tanh
    ϵ = allencahn2d.params[:ϵ]
    ux0 = tanh.((0.3 .- r_actual) ./ sqrt(2 * ϵ))
    allencahn2d.IC = vec(ux0)

    #==================#
    ## Model Operators
    #==================#
    A, E = allencahn2d.finite_diff_model(allencahn2d, allencahn2d.params)

    #============#
    ## Integrate
    #============#
    Ucnab = allencahn2d.integrate_model(
        allencahn2d.tspan, allencahn2d.IC;
        linear_matrix=A, cubic_matrix=E, system_input=false,
        integrator_type=:CNAB
    )
    @test size(Ucnab) == (prod(allencahn2d.spatial_dim), allencahn2d.time_dim)

    Usicn = allencahn2d.integrate_model(
        allencahn2d.tspan, allencahn2d.IC;
        linear_matrix=A, cubic_matrix=E, system_input=false,
        integrator_type=:SICN
    )
    @test size(Usicn) == (prod(allencahn2d.spatial_dim), allencahn2d.time_dim)

    # Fast SICN / CNAB (periodic 2D → FFT-based solver)
    solver = pomoreda.build_fast_solver(allencahn2d, allencahn2d.params; scheme=:CN)
    @test solver isa pomoreda.FastFFT2DSolver
    Ufast_cnab = pomoreda.integrate_model_fast(allencahn2d, solver,
                                                 allencahn2d.tspan, allencahn2d.IC;
                                                 cubic_matrix=E, integrator_type=:CNAB)
    @test Ufast_cnab ≈ Ucnab
    Ufast_sicn = pomoreda.integrate_model_fast(allencahn2d, solver,
                                                 allencahn2d.tspan, allencahn2d.IC;
                                                 cubic_matrix=E, integrator_type=:SICN)
    @test Ufast_sicn ≈ Usicn

    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = ((0.0, 1.0), (0.0, 1.0))
    Nx = 2^4
    Ny = 2^4
    allencahn2d = pomoreda.AllenCahn2DModel(
        spatial_domain=Ω, time_domain=(0,0.5), 
        Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
        params=Dict(:μ => 0.001, :ϵ => 1.0), BC=(:dirichlet, :dirichlet)
    )
    
    xg = allencahn2d.xspan
    yg = allencahn2d.yspan
    X = repeat(xg', length(yg), 1)
    Y = repeat(yg, 1, length(xg))
    
    # Initial condition
    ux0 = 0.1 .* sin.(2π .* X) .* cos.(2π .* Y)
    allencahn2d.IC = vec(ux0)
    
    # Boundary condition
    Ubc_vals = [1.0, 1.0, -1.0, -1.0]
    Ubc = repeat(reshape(Ubc_vals, 4, 1), 1, allencahn2d.time_dim)

    #==================#
    ## Model Operators
    #==================#
    A, E, B = allencahn2d.finite_diff_model(allencahn2d, allencahn2d.params)

    #============#
    ## Integrate
    #============#
    Usicn = allencahn2d.integrate_model(
        allencahn2d.tspan, allencahn2d.IC, Ubc;
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:SICN,
    )
    @test size(Usicn) == (prod(allencahn2d.spatial_dim), allencahn2d.time_dim)

    Ucnab = allencahn2d.integrate_model(
        allencahn2d.tspan, allencahn2d.IC, Ubc;
        linear_matrix=A, cubic_matrix=E, control_matrix=B,
        system_input=true, integrator_type=:CNAB,
    )
    @test size(Ucnab) == (prod(allencahn2d.spatial_dim), allencahn2d.time_dim)

    # Fast SICN / CNAB (Dirichlet 2D → FastKronSumSolver)
    solver = pomoreda.build_fast_solver(allencahn2d, allencahn2d.params; scheme=:CN)
    @test solver isa pomoreda.FastKronSumSolver
    Ufast_sicn = pomoreda.integrate_model_fast(allencahn2d, solver,
                                                 allencahn2d.tspan, allencahn2d.IC, Ubc;
                                                 cubic_matrix=E, control_matrix=B,
                                                 integrator_type=:SICN)
    @test Ufast_sicn ≈ Usicn
    Ufast_cnab = pomoreda.integrate_model_fast(allencahn2d, solver,
                                                 allencahn2d.tspan, allencahn2d.IC, Ubc;
                                                 cubic_matrix=E, control_matrix=B,
                                                 integrator_type=:CNAB)
    @test Ufast_cnab ≈ Ucnab
end
