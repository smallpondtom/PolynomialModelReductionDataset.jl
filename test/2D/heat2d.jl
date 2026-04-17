@testset "2D Heat equation" begin
    #========#
    ## Model
    #========#
    Ω = ((0.0, 1.0), (0.0, 1.0))
    Nx = 2^4
    Ny = 2^4
    heat2d = pomoreda.Heat2DModel(
        spatial_domain=Ω, time_domain=(0,2),
        Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
        diffusion_coeffs=0.1, BC=(:dirichlet, :dirichlet)
    )
    xgrid0 = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
    ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
    ux0 = sin.(2π * xgrid0) .* cos.(2π * ygrid0)
    heat2d.IC = vec(ux0)  # initial condition

    # Boundary condition
    Ubc = [1.0, 1.0, -1.0, -1.0]
    Ubc = repeat(Ubc, 1, heat2d.time_dim)

    #==================#
    ## Model Operators
    #==================#
    A, B = heat2d.finite_diff_model(heat2d, heat2d.diffusion_coeffs)

    #================#
    ## Forward Euler
    #================#
    U = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B,
        system_input=true, integrator_type=:ForwardEuler
    )
    @test size(U) == (prod(heat2d.spatial_dim), heat2d.time_dim)

    #=================#
    ## Backward Euler
    #=================#
    Ube = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B,
         system_input=true, integrator_type=:BackwardEuler
    )
    @test size(Ube) == (prod(heat2d.spatial_dim), heat2d.time_dim)

    #=================#
    ## Crank-Nicolson
    #=================#
    U = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Ubc; linear_matrix=A, control_matrix=B,
        system_input=true, integrator_type=:CrankNicolson
    )
    @test size(U) == (prod(heat2d.spatial_dim), heat2d.time_dim)

    #======================#
    ## Fast Backward Euler
    #======================#
    # If your package layout doesn't re-export these from the submodule,
    # replace `pomoreda.X` with `pomoreda.Heat2D.X` below.

    # (a) Build solver explicitly, then integrate
    solver = pomoreda.build_fast_be_solver(heat2d, heat2d.diffusion_coeffs)
    @test solver isa pomoreda.FastDirichletSolver

    Ufast = pomoreda.integrate_model_fast(solver, B, Ubc, heat2d.tspan, heat2d.IC)
    @test size(Ufast) == (prod(heat2d.spatial_dim), heat2d.time_dim)
    @test Ufast ≈ Ube                         # must match sparse-LU BE

    # Initial condition preserved exactly
    @test Ufast[:, 1] == heat2d.IC

    # (b) Convenience overload that builds the solver internally
    Ufast2 = pomoreda.integrate_model_fast(
        heat2d, heat2d.diffusion_coeffs, B, Ubc, heat2d.tspan, heat2d.IC
    )
    @test size(Ufast2) == (prod(heat2d.spatial_dim), heat2d.time_dim)
    @test Ufast2 ≈ Ube

    # (c) Reusing the same solver across multiple ICs gives consistent results
    IC2 = vec(cos.(2π * xgrid0) .* sin.(2π * ygrid0))
    Ufast_a = pomoreda.integrate_model_fast(solver, B, Ubc, heat2d.tspan, IC2)
    Ube2 = heat2d.integrate_model(
        heat2d.tspan, IC2, Ubc; linear_matrix=A, control_matrix=B,
        system_input=true, integrator_type=:BackwardEuler
    )
    @test Ufast_a ≈ Ube2
end


@testset "2D Heat equation (periodic, fast Backward Euler)" begin
    Ω = ((0.0, 1.0), (0.0, 1.0))
    Nx = 2^4
    Ny = 2^4
    heat2d = pomoreda.Heat2DModel(
        spatial_domain=Ω, time_domain=(0, 2),
        Δx=1/Nx, Δy=1/Ny, Δt=1e-3,
        diffusion_coeffs=0.1, BC=(:periodic, :periodic),
    )
    xg = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
    yg = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
    heat2d.IC = vec(sin.(2π * xg) .* cos.(2π * yg))

    # Periodic finite-difference model returns A only (no boundary inputs)
    A = heat2d.finite_diff_model(heat2d, heat2d.diffusion_coeffs)

    # Reference: original sparse-LU backward Euler with no system input
    Ube = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC; linear_matrix=A,
        system_input=false, integrator_type=:BackwardEuler,
    )
    @test size(Ube) == (prod(heat2d.spatial_dim), heat2d.time_dim)

    # Fast FFT-based solver
    solver = pomoreda.build_fast_be_solver(heat2d, heat2d.diffusion_coeffs)
    @test solver isa pomoreda.FastPeriodicSolver

    # No-input convenience overload (model + μ + tspan + IC)
    Ufast = pomoreda.integrate_model_fast(
        heat2d, heat2d.diffusion_coeffs, heat2d.tspan, heat2d.IC
    )
    @test size(Ufast) == (prod(heat2d.spatial_dim), heat2d.time_dim)
    @test Ufast ≈ Ube
    @test Ufast[:, 1] == heat2d.IC

    # Heat equation with periodic BCs conserves the spatial mean
    mean_t0  = sum(@view Ufast[:, 1])
    mean_end = sum(@view Ufast[:, end])
    @test isapprox(mean_t0, mean_end; atol=1e-10)

    # And the total energy (L2 norm) must be non-increasing
    energies = [sum(abs2, @view Ufast[:, j]) for j in 1:heat2d.time_dim]
    @test all(diff(energies) .≤ 1e-12)
end

@testset "FastDenseSolver (unstructured dense A, ROM use case)" begin
    using LinearAlgebra, SparseArrays
 
    # ----------------------------------------------------------------
    # Setup: build a small reduced system by projecting the heat operator
    # onto a random orthonormal basis, mimicking what you get from OpInf
    # or POD/Galerkin.
    # ----------------------------------------------------------------
    Ω = ((0.0, 1.0), (0.0, 1.0))
    Nx = 2^4; Ny = 2^4
    heat2d = pomoreda.Heat2DModel(
        spatial_domain=Ω, time_domain=(0, 0.5),
        Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
        diffusion_coeffs=0.1, BC=(:dirichlet, :dirichlet)
    )
    Afom, Bfom = heat2d.finite_diff_model(heat2d, heat2d.diffusion_coeffs)
 
    # Random orthonormal basis of dimension r
    r = 20
    N = prod(heat2d.spatial_dim)
    Vr = Matrix(qr(randn(N, r)).Q)
 
    # Dense reduced operators
    Ar = Matrix(Vr' * Afom * Vr)   # r × r, dense, generally non-symmetric
    Br = Vr' * Matrix(Bfom)        # r × m
 
    # Reduced IC and input
    xgrid0 = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
    ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
    IC_fom = vec(sin.(2π * xgrid0) .* cos.(2π * ygrid0))
    u0r = Vr' * IC_fom
 
    Ubc = repeat([1.0, 1.0, -1.0, -1.0], 1, heat2d.time_dim)
 
    Δt = heat2d.Δt
    tdata = heat2d.tspan
    Tdim = length(tdata)
 
    # ----------------------------------------------------------------
    # Reference: naive backward Euler on the dense reduced system
    # (re-factorizes every step, like the original code)
    # ----------------------------------------------------------------
    Uref = zeros(r, Tdim)
    Uref[:, 1] = u0r
    M = I(r) - Δt * Ar
    for j in 2:Tdim
        Uref[:, j] = M \ (Uref[:, j-1] + Δt * Br * Ubc[:, j-1])
    end
 
    # ----------------------------------------------------------------
    # (a) FastDenseSolver: construct and integrate with input
    # ----------------------------------------------------------------
    solver = pomoreda.FastDenseSolver(Ar, Δt)
    @test solver isa pomoreda.FastDenseSolver
    @test solver.r == r
    @test solver.eigen_based == true
 
    Ufast = pomoreda.integrate_model_fast(solver, tdata, u0r, Br, Ubc)
    @test size(Ufast) == (r, Tdim)
    @test Ufast[:, 1] == u0r
    @test Ufast ≈ Uref
 
    # ----------------------------------------------------------------
    # (b) No-input overload
    # ----------------------------------------------------------------
    Uref_noinput = zeros(r, Tdim)
    Uref_noinput[:, 1] = u0r
    for j in 2:Tdim
        Uref_noinput[:, j] = M \ Uref_noinput[:, j-1]
    end
 
    Ufast_noinput = pomoreda.integrate_model_fast(solver, tdata, u0r)
    @test size(Ufast_noinput) == (r, Tdim)
    @test Ufast_noinput ≈ Uref_noinput
 
    # ----------------------------------------------------------------
    # (c) update_timestep!: change Δt without re-eigendecomposing
    # ----------------------------------------------------------------
    Δt2 = 2e-3
    tdata2 = collect(0.0:Δt2:0.5)
    Tdim2 = length(tdata2)
 
    # Reference with new Δt
    M2 = I(r) - Δt2 * Ar
    Uref2 = zeros(r, Tdim2)
    Uref2[:, 1] = u0r
    Ubc2 = repeat([1.0, 1.0, -1.0, -1.0], 1, Tdim2)
    for j in 2:Tdim2
        Uref2[:, j] = M2 \ (Uref2[:, j-1] + Δt2 * Br * Ubc2[:, j-1])
    end
 
    pomoreda.update_timestep!(solver, Δt2)
    Ufast2 = pomoreda.integrate_model_fast(solver, tdata2, u0r, Br, Ubc2)
    @test Ufast2 ≈ Uref2
 
    # ----------------------------------------------------------------
    # (d) Non-symmetric A with complex eigenvalues
    # ----------------------------------------------------------------
    # Add a skew-symmetric part to force complex eigenvalues
    S = randn(r, r)
    S = 0.3 * (S - S') / 2                # skew-symmetric
    Ar_nonsym = Ar + S                    # non-symmetric, will have complex eigenvalues
    λ_nonsym = eigvals(Ar_nonsym)
 
    solver_ns = pomoreda.FastDenseSolver(Ar_nonsym, Δt)
    @test solver_ns.eigen_based == true
 
    M_ns = I(r) - Δt * Ar_nonsym
    Uref_ns = zeros(r, Tdim)
    Uref_ns[:, 1] = u0r
    for j in 2:Tdim
        Uref_ns[:, j] = M_ns \ Uref_ns[:, j-1]
    end
 
    Ufast_ns = pomoreda.integrate_model_fast(solver_ns, tdata, u0r)
    @test Ufast_ns ≈ Uref_ns
 
    # ----------------------------------------------------------------
    # (e) M_inv is mathematically correct: M_inv ≈ inv(I - Δt A)
    # ----------------------------------------------------------------
    M_inv_direct = inv(Matrix(I(r) - Δt * Ar))
    @test solver_ns.M_inv ≈ inv(Matrix(I(r) - Δt * Ar_nonsym))
 
    # Restore Δt for the original solver to check too
    pomoreda.update_timestep!(solver, Δt)
    @test solver.M_inv ≈ M_inv_direct
end