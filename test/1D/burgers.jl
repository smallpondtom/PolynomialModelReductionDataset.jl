@testset "Burgers equation" begin
    #=======================#
    ## Model (Dirichlet BC)
    #=======================#
    Ω = (0.0, 1.0)
    Nx = 2^7; dt = 1e-4
    burgers = pomoreda.BurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        diffusion_coeffs=0.1, BC=:dirichlet,
    )
    Ubc = rand(burgers.time_dim) # boundary condition

    #==================#
    ## Model Operators
    #==================#
    A, F, B = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs; opposite_sign_on_ends=true)

    #============#
    ## Integrate
    #============#
    U = burgers.integrate_model(
        burgers.tspan, burgers.IC, Ubc; 
        linear_matrix=A, control_matrix=B, quadratic_matrix=F,
        system_input=true
    )
    @test size(U) == (burgers.spatial_dim, burgers.time_dim)

    #===================================#
    ## Model (Periodic BC Conservative)
    #===================================#
    Ω = (0.0, 1.0)
    Nx = 2^7; dt = 1e-4
    burgers = pomoreda.BurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        diffusion_coeffs=0.1, BC=:periodic, conservation_type=:C
    )
    burgers.IC = sin.(2π * burgers.xspan) # initial condition

    #==================#
    ## Model Operators
    #==================#
    A, F = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs)

    #============#
    ## Integrate
    #============#
    U = burgers.integrate_model(
        burgers.tspan, burgers.IC, Float64[]; 
        linear_matrix=A, quadratic_matrix=F,
        system_input=false
    )
    @test size(U) == (burgers.spatial_dim, burgers.time_dim)

    #=======================================#
    ## Model (Periodic BC Non-Conservative)
    #=======================================#
    Ω = (0.0, 1.0)
    Nx = 2^7; dt = 1e-4
    burgers = pomoreda.BurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        diffusion_coeffs=0.1, BC=:periodic, conservation_type=:NC
    )
    burgers.IC = sin.(2π * burgers.xspan) # initial condition

    #==================#
    ## Model Operators
    #==================#
    A, F = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs)

    #============#
    ## Integrate
    #============#
    U = burgers.integrate_model(
        burgers.tspan, burgers.IC, Float64[]; 
        linear_matrix=A, quadratic_matrix=F,
        system_input=false
    )
    @test size(U) == (burgers.spatial_dim, burgers.time_dim)

    #========================================#
    ## Model (Periodic BC Energy-Preserving)
    #========================================#
    Ω = (0.0, 1.0)
    Nx = 2^7; dt = 1e-4
    burgers = pomoreda.BurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        diffusion_coeffs=0.1, BC=:periodic, conservation_type=:EP
    )
    burgers.IC = sin.(2π * burgers.xspan) # initial condition

    #==================#
    ## Model Operators
    #==================#
    A, F = burgers.finite_diff_model(burgers, burgers.diffusion_coeffs)

    #============#
    ## Integrate
    #============#
    U = burgers.integrate_model(
        burgers.tspan, burgers.IC, Float64[]; 
        linear_matrix=A, quadratic_matrix=F,
        system_input=false
    )
    @test size(U) == (burgers.spatial_dim, burgers.time_dim)
end