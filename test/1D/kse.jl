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
    U = kse.integrate_model(kse.tspan, u0, nothing; linear_matrix=A, quadratic_matrix=F, const_stepsize=true)
    @test size(U) == (kse.spatial_dim, kse.time_dim)

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