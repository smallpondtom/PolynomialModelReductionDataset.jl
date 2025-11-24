@testset "Kawahara equation" begin
    #===============================================#
    ## Model (Periodic BC) - 1st order dispersion
    #===============================================#
    Ω = (0.0, 50.0)
    dt = 0.01
    N = 128
    kawahara = pomoreda.KawaharaModel(
        spatial_domain=Ω, time_domain=(0.0, 50.0), 
        params=Dict(:mu => 1.0, :delta => 0.15, :nu => 0.0),
        dispersion_order=1,
        conservation_type=:NC,
        Δx=(Ω[2] - 1/N)/N, Δt=dt
    )
    L = kawahara.spatial_domain[2]

    # Initial condition
    a = 1.0
    b = 0.1
    u0 = a*cos.((2*π*kawahara.xspan)/L) + b*cos.((4*π*kawahara.xspan)/L)

    #==================#
    ## Model Operators
    #==================#
    A, F = kawahara.finite_diff_model(kawahara, kawahara.params[:mu], kawahara.params[:delta])

    #============#
    ## Integrate
    #============#
    U = kawahara.integrate_model(
        kawahara.tspan, u0, nothing; 
        linear_matrix=A, quadratic_matrix=F, const_stepsize=true
    )
    @test size(U) == (kawahara.spatial_dim, kawahara.time_dim)

    #===============================================#
    ## Model (Periodic BC) - 3rd order dispersion
    #===============================================#
    Ω = (0.0, 50.0)
    dt = 0.01
    N = 128
    kawahara = pomoreda.KawaharaModel(
        spatial_domain=Ω, time_domain=(0.0, 50.0), 
        params=Dict(:mu => 1.0, :delta => 0.15, :nu => 0.0),
        dispersion_order=3,
        conservation_type=:C,
        Δx=(Ω[2] - 1/N)/N, Δt=dt
    )
    L = kawahara.spatial_domain[2]

    # Initial condition
    u0 = a*cos.((2*π*kawahara.xspan)/L) + b*cos.((4*π*kawahara.xspan)/L)

    #==================#
    ## Model Operators
    #==================#
    A, F = kawahara.finite_diff_model(kawahara, kawahara.params[:mu], kawahara.params[:delta])

    #============#
    ## Integrate
    #============#
    U = kawahara.integrate_model(
        kawahara.tspan, u0, nothing; 
        linear_matrix=A, quadratic_matrix=F, const_stepsize=true
    )
    @test size(U) == (kawahara.spatial_dim, kawahara.time_dim)

    #===============================================#
    ## Model (Periodic BC) - 5th order dispersion
    #===============================================#
    Ω = (0.0, 50.0)
    dt = 0.01
    N = 128
    kawahara = pomoreda.KawaharaModel(
        spatial_domain=Ω, time_domain=(0.0, 50.0), 
        params=Dict(:mu => 1.0, :delta => 0.0, :nu => 0.05),
        dispersion_order=5,
        conservation_type=:EP,
        Δx=(Ω[2] - 1/N)/N, Δt=dt
    )
    L = kawahara.spatial_domain[2]

    # Initial condition
    u0 = a*cos.((2*π*kawahara.xspan)/L) + b*cos.((4*π*kawahara.xspan)/L)

    #==================#
    ## Model Operators
    #==================#
    A, F = kawahara.finite_diff_model(kawahara, kawahara.params[:mu], kawahara.params[:delta], kawahara.params[:nu])

    #============#
    ## Integrate
    #============#
    U = kawahara.integrate_model(
        kawahara.tspan, u0, nothing; 
        linear_matrix=A, quadratic_matrix=F, const_stepsize=true
    )
    @test size(U) == (kawahara.spatial_dim, kawahara.time_dim)

    #===============================================#
    ## Model (Periodic BC) - 3rd + 5th order dispersion
    #===============================================#
    Ω = (0.0, 50.0)
    dt = 0.01
    N = 128
    kawahara = pomoreda.KawaharaModel(
        spatial_domain=Ω, time_domain=(0.0, 50.0), 
        params=Dict(:mu => 1.0, :delta => 0.15, :nu => 0.05),
        dispersion_order=8,
        conservation_type=:NC,
        Δx=(Ω[2] - 1/N)/N, Δt=dt
    )
    L = kawahara.spatial_domain[2]

    # Initial condition
    u0 = a*cos.((2*π*kawahara.xspan)/L) + b*cos.((4*π*kawahara.xspan)/L)

    #==================#
    ## Model Operators
    #==================#
    A, F = kawahara.finite_diff_model(kawahara, kawahara.params[:mu], kawahara.params[:delta], kawahara.params[:nu])

    #============#
    ## Integrate
    #============#
    U = kawahara.integrate_model(
        kawahara.tspan, u0, nothing; 
        linear_matrix=A, quadratic_matrix=F, const_stepsize=true
    )
    @test size(U) == (kawahara.spatial_dim, kawahara.time_dim)
end
