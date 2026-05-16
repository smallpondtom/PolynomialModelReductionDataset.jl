@testset "Kuramoto-Sivashinsky equation" begin
    #======================#
    ## Model (Periodic BC)
    #======================#
    # Settings for the KS equation
    Ω = (0.0, 22.0)
    dt = 0.01
    N = 256
    kse = pomoreda.KuramotoSivashinskyModel(
        spatial_domain=Ω, time_domain=(0.0, 300.0), diffusion_coeffs=1.0,
        Δx=(Ω[2] - 1/N)/N, Δt=dt
    )
    DS = 100
    L = kse.spatial_domain[2]

    # Initial condition
    a = 1.0
    b = 0.1
    u0 = a*cos.((2*π*kse.xspan)/L) + b*cos.((4*π*kse.xspan)/L)

    #=============================#
    ## Finite Difference Operators
    #==============================#
    A, F = kse.finite_diff_model(kse, kse.diffusion_coeffs)

    #==================#
    ## Integrate Model
    #==================#
    Uref = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F, const_stepsize=true)
    @test size(Uref) == (kse.spatial_dim, kse.time_dim)

    # Fast CNAB (finite-difference KSE only).
    # KSE is sensitive; over the full 300-second horizon the LU-based slow path
    # and the FFT-based fast path drift apart by O(1e-4). Verify agreement on a
    # short horizon where the two are still within tight numerical tolerance.
    kse_short = pomoreda.KuramotoSivashinskyModel(
        spatial_domain=Ω, time_domain=(0.0, 3.0), diffusion_coeffs=1.0,
        Δx=(Ω[2] - 1/N)/N, Δt=dt
    )
    u0_short = a*cos.((2*π*kse_short.xspan)/L) + b*cos.((4*π*kse_short.xspan)/L)
    A_s, F_s = kse_short.finite_diff_model(kse_short, kse_short.diffusion_coeffs)
    Uref_s = kse_short.integrate_model(kse_short.tspan, u0_short, nothing;
                                        linear_matrix=A_s, quadratic_matrix=F_s,
                                        const_stepsize=true)
    solver = pomoreda.build_fast_solver(kse_short, kse_short.diffusion_coeffs; scheme=:CN)
    @test solver isa pomoreda.FastCirculant1DSolver
    Ufast = pomoreda.integrate_model_fast(kse_short, solver, kse_short.tspan, u0_short;
                                           quadratic_matrix=F_s)
    @test Ufast ≈ Uref_s

    #============================#
    ## Pseudo-Spectral Operators
    #============================#
    A, F = kse.pseudo_spectral_model(kse, kse.diffusion_coeffs)

    #==================#
    ## Integrate Model
    #==================#
    U, Uhat = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F)
    @test size(U) == (kse.spatial_dim, kse.time_dim)

    #========================================#
    ## Elementwise Pseudo-Spectral Operators
    #========================================#
    A, F = kse.elementwise_pseudo_spectral_model(kse, kse.diffusion_coeffs)

    #==================#
    ## Integrate Model
    #==================#
    U, Uhat = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F)
    @test size(U) == (kse.spatial_dim, kse.time_dim)

    #==============================#
    ## Spectral-Galerkin Operators
    #==============================#
    A, F = kse.spectral_galerkin_model(kse, kse.diffusion_coeffs)

    #==================#
    ## Integrate Model
    #==================#
    U, Uhat = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F)
    @test size(U) == (kse.spatial_dim, kse.time_dim)
end