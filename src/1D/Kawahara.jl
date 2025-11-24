"""
    Kawahara or Dispersively-Modified Kuramoto-Sivashinsky Equation model
"""
module Kawahara

using DocStringExtensions
using FFTW
using LinearAlgebra
using SparseArrays
using UniqueKronecker

import ..PolynomialModelReductionDataset: AbstractModel

export KawaharaModel


"""
$(TYPEDEF)

Kawahara equation model (also known as dispersively-modified Kuramoto-Sivashinsky 
equation or Benney-Lin equation) is a nonlinear PDE that describes the evolution 
of certain physical systems, such as fluid dynamics and plasma physics. The
equation can be written in several forms, including:

```math
\\frac{\\partial u}{\\partial t} = -\\mu\\frac{\\partial^4 u}{\\partial x^4} - 
\\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x} - 
\\delta\\frac{\\partial u}{\\partial x}
```
    
or 

```math
\\frac{\\partial u}{\\partial t} = -\\mu\\frac{\\partial^4 u}{\\partial x^4} - 
\\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x} - 
\\delta\\frac{\\partial^3 u}{\\partial x^3}
```

or 

```math
\\frac{\\partial u}{\\partial t} = -\\mu\\frac{\\partial^4 u}{\\partial x^4} - 
\\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x} - 
\\nu\\frac{\\partial^3 u}{\\partial x^5}
```

or 

```math
\\frac{\\partial u}{\\partial t} = -\\mu\\frac{\\partial^4 u}{\\partial x^4} - 
\\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x} - 
\\delta\\frac{\\partial^3 u}{\\partial x^3} - 
\\nu\\frac{\\partial^5 u}{\\partial x^5}
```

where ``u`` is the state variable, ``\\mu`` is the viscosity coefficient,
``\\delta`` is the 1st or 3rd order dispersion coefficient, and ``\\nu`` 
is the 5th order dispersion coefficient.

# Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Dict{Symbol,Tuple{Real,Real}}`: parameter domains
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `BC::Symbol`: boundary condition
- `IC::Array{Float64}`: initial condition
- `xspan::Vector{Float64}`: spatial grid points
- `tspan::Vector{Float64}`: temporal points
- `params::Dict{Symbol,<:Union{<:Real,<:AbstractArray{<:Real}}}`: parameters (diffusion and dispersion coefficients)
- `fourier_modes::Vector{Float64}`: Fourier modes
- `spatial_dim::Int64`: spatial dimension
- `time_dim::Int64`: temporal dimension
- `param_dim::Dict{Symbol,Int64}`: parameter dimensions
- `conservation_type::Symbol`: conservation type
- `finite_diff_model::Function`: finite difference model
- `integrate_model::Function`: integrator
- `jacobian::Function`: Jacobian matrix
"""
mutable struct KawaharaModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Dict{Symbol,<:Tuple{Real,Real}}  # parameter domain

    # Discritization grid
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size

    # Boundary condition
    BC::Symbol  # boundary condition

    # Initial conditino
    IC::Array{Float64}  # initial condition

    # grid points
    xspan::Vector{Float64}  # spatial grid points
    tspan::Vector{Float64}  # temporal points
    params::Dict{Symbol,<:Union{<:Real,<:AbstractArray{<:Real}}} # parameters

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Dict{Symbol,Int64} # parameter dimension

    # Convervation type
    conservation_type::Symbol

    # Model type
    dispersion_order::Int64

    finite_diff_model::Function
    integrate_model::Function
end



"""
$(SIGNATURES)

Constructor for the Kawahara equation model.
"""
function KawaharaModel(;
        spatial_domain::Tuple{Real,Real}, 
        time_domain::Tuple{Real,Real}, 
        Δx::Real, Δt::Real, 
        params::Dict{Symbol,<:Union{<:Real,<:AbstractArray{<:Real}}},
        BC::Symbol=:periodic,
        dispersion_order::Int64=1,
        conservation_type::Symbol=:NC)

    # Discritization grid info
    @assert BC ∈ (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy,
                  :flux) "Invalid boundary condition"

    if BC == :periodic
        xspan = collect(spatial_domain[1]:Δx:spatial_domain[2]-Δx)
    elseif BC ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        xspan = collect(spatial_domain[1]:Δx:spatial_domain[2])
    end

    tspan = collect(time_domain[1]:Δt:time_domain[2])
    spatial_dim = length(xspan)
    time_dim = length(tspan)

    # Initial condition
    IC = zeros(spatial_dim)

    # Check the parameters are `mu` and `delta`
    @assert all(k ∈ (:mu, :delta, :nu) for k in keys(params)) (
        "Parameters must include :mu (diffusion), :delta (1st or 3rd order " *
        "dispersion), and :nu (5th order dispersion) coefficients"
    )

    # Parameter dimensions or number of parameters 
    param_dim = Dict([k => length(v) for (k, v) in params])
    param_domain = Dict([k => extrema(v) for (k,v) in params])

    @assert conservation_type ∈ (:EP, :NC, :C) "Invalid conservation type"
    integrate_model = integrate_finite_diff_model

    @assert dispersion_order ∈ (1, 3, 5, 8) (
        "Invalid dispersion order, must be 1, 3, 5, or 3+5=`8`"
    )

    KawaharaModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, 
        BC, IC, 
        xspan, tspan, params,
        spatial_dim, time_dim, param_dim, 
        conservation_type, 
        dispersion_order,
        finite_diff_model, 
        integrate_model
    )
end


"""
$(SIGNATURES)

Finite Difference Model for the Kuramoto-Sivashinsky equation.

# Arguments
- `model::KawaharaModel`: Kuramoto-Sivashinsky equation model
- `μ::Real`: parameter value

# Returns
- `A`: A matrix
- `F`: F matrix
"""
function finite_diff_model(model::KawaharaModel, μ::Real, δ::Real, ν::Real=0.0)
    if model.BC == :periodic
        if model.conservation_type == :NC
            return finite_diff_periodic_nonconservative_model(
                model.spatial_dim, model.Δx, μ, δ, model.dispersion_order, ν)
        elseif model.conservation_type == :C
            return finite_diff_periodic_conservative_model(
                model.spatial_dim, model.Δx, μ, δ, model.dispersion_order, ν)
        elseif model.conservation_type == :EP
            return finite_diff_periodic_energy_preserving_model(
                model.spatial_dim, model.Δx, μ, δ, model.dispersion_order, ν)
        else
            error("Conservation type not implemented")
        end
    else
        error("Boundary condition not implemented")
    end
end

function finite_diff_linear_part(N, Δx, μ, δ, ν, ord)
    # Create A matrix
    if ord == 1
        np2 = -μ/Δx^4 
        np1 = 4*μ/Δx^4 - 1/Δx^2 - δ/(2*Δx)
        n   = -6*μ/Δx^4 + 2/Δx^2
        nm1 = 4*μ/Δx^4 - 1/Δx^2 + δ/(2*Δx)
        nm2 = -μ/Δx^4
    elseif ord == 3
        np2 = -μ/Δx^4 - δ/(2*Δx^3)
        np1 = 4*μ/Δx^4 + δ/(Δx^3) - 1/Δx^2
        n   = -6*μ/Δx^4 + 2/Δx^2
        nm1 = 4*μ/Δx^4 - δ/(Δx^3) - 1/Δx^2
        nm2 = -μ/Δx^4 + δ/(2*Δx^3)
    elseif ord == 5
        np3 = -ν/(2*Δx^5)
        np2 = 2*ν/Δx^5 - μ/Δx^4
        np1 = -5*ν/(2*Δx^5) + 4*μ/Δx^4 - 1/Δx^2
        n   = -6*μ/Δx^4 + 2/Δx^2
        nm1 = 5*ν/(2*Δx^5) + 4*μ/Δx^4 - 1/Δx^2
        nm2 = -2*ν/Δx^5 - μ/Δx^4
        nm3 = ν/(2*Δx^5)
    elseif ord == 8
        np3 = -ν/(2*Δx^5)
        np2 = 2*ν/Δx^5 - μ/Δx^4 - δ/(2*Δx^3)
        np1 = -5*ν/(2*Δx^5) + 4*μ/Δx^4 + δ/(Δx^3) - 1/Δx^2
        n   = -6*μ/Δx^4 + 2/Δx^2
        nm1 = 5*ν/(2*Δx^5) + 4*μ/Δx^4 - δ/(Δx^3) - 1/Δx^2
        nm2 = -2*ν/Δx^5 - μ/Δx^4 + δ/(2*Δx^3)
        nm3 = ν/(2*Δx^5)
    end

    if ord < 5
        A = spdiagm(
            2 => np2 * ones(N - 2),
            1 => np1 * ones(N - 1),
            0 =>  n  * ones(N),
            -1 => nm1 * ones(N - 1),
            -2 => nm2 * ones(N - 2)
        )
        # For the periodicity for the first and final few indices
        A[1, end-1:end] = [nm2, nm1]
        A[2, end] = nm2
        A[end-1, 1] = np2
        A[end, 1:2] = [np1, np2]
    else
        A = spdiagm(
            3 => np3 * ones(N - 3),
            2 => np2 * ones(N - 2),
            1 => np1 * ones(N - 1),
            0 =>  n  * ones(N),
            -1 => nm1 * ones(N - 1),
            -2 => nm2 * ones(N - 2),
            -3 => nm3 * ones(N - 3)
        )
        # For the periodicity for the first and final few indices
        A[1, end-2:end] = [nm3, nm2, nm1]
        A[2, end-1:end] = [nm3, nm2]
        A[3, end] = nm3
        A[end-2, 1] = np3
        A[end-1, 1:2] = [np2, np3]
        A[end, 1:3] = [np1, np2, np3]
    end
    return A
end

"""
$(SIGNATURES)

Finite Difference Model for the Kuramoto-Sivashinsky equation with periodic 
boundary condition.
"""
function finite_diff_periodic_nonconservative_model(N::Real, Δx::Real, 
                                                    μ::Real, δ::Real, 
                                                    ord::Int, ν::Real=0.0)
    # Create A matrix
    A = finite_diff_linear_part(N, Δx, μ, δ, ν, ord)

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx

        # For the periodicity for the first and final indices
        F[1, 2] = - 1 / 2 / Δx
        F[1, N] = 1 / 2 / Δx
        F[N, N] = - 1 / 2 / Δx
        F[N, end-1] = 1 / 2 / Δx 
    else
        F = zeros(N, S)
    end
    return A, sparse(F)
end


"""
$(SIGNATURES)

Finite Difference Model for the Kuramoto-Sivashinsky equation with periodic 
boundary condition.
"""
function finite_diff_periodic_conservative_model(N::Real, Δx::Real, 
                                                 μ::Real, δ::Real, 
                                                 ord::Int, ν::Real=0.0)
    # Create A matrix
    A = finite_diff_linear_part(N, Δx, μ, δ, ν, ord)

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        ii = repeat(2:(N-1), inner=2)
        m = 2:N-1
        mm = Int.([
            N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) 
            for m in 2:N-1
        ])  # this is where the x_{i-1}^2 term is
        mp = Int.([
            N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) 
            for m in 2:N-1
        ])  # this is where the x_{i+1}^2 term is
        jj = reshape([mp'; mm'],2*N-4);
        vv = reshape([-ones(1,N-2); ones(1,N-2)],2*N-4)/(4*Δx);
        F = sparse(ii,jj,vv,N,S)

        # Boundary conditions (Periodic)
        F[1,N+1] = -1/4/Δx
        F[1,end] = 1/4/Δx
        F[N,end-2] = 1/4/Δx
        F[N,1] = -1/4/Δx
    else
        F = zeros(N, S)
    end
    return A, sparse(F)
end


"""
$(SIGNATURES)

Finite Difference Model for the Kuramoto-Sivashinsky equation with periodic 
boundary condition.
"""
function finite_diff_periodic_energy_preserving_model(N::Real, Δx::Real, 
                                                      μ::Real, δ::Real, 
                                                      ord::Int, ν::Real=0.0)
    # Create A matrix
    A = finite_diff_linear_part(N, Δx, μ, δ, ν, ord)
    
    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        ii = repeat(2:(N-1), inner=4)
        m = 2:N-1
        mi = Int.([
            N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) 
            for m in 2:N-1
        ])               # this is where the xi^2 term is
        mm = Int.([
            N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) 
            for m in 2:N-1
        ])  # this is where the x_{i-1}^2 term is
        mp = Int.([
            N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) 
            for m in 2:N-1
        ])  # this is where the x_{i+1}^2 term is
        jp = mi .+ 1  # this is the index of the x_{i+1}*x_i term
        jm = mm .+ 1  # this is the index of the x_{i-1}*x_i term
        jj = reshape([mp'; mm'; jp'; jm'],4*N-8);
        vv = reshape([
            -ones(1,N-2); ones(1,N-2); -ones(1,N-2); ones(1,N-2)
        ], 4*N-8)/(6*Δx);
        F = sparse(ii,jj,vv,N,S)

        # Boundary conditions (Periodic)
        F[1,2] = -1/6/Δx
        F[1,N+1] = -1/6/Δx
        F[1,N] = 1/6/Δx
        F[1,end] = 1/6/Δx
        F[N,end-1] = 1/6/Δx
        F[N,end-2] = 1/6/Δx
        F[N,1] = -1/6/Δx
        F[N,N] = -1/6/Δx
    else
        F = zeros(N, S)
    end
    return A, sparse(F)
end


"""
$(SIGNATURES)

Integrator using Crank-Nicholson Adams-Bashforth method for (FD). 

# Arguments
- `tdata`: temporal points
- `IC`: initial condition

# Keyword Arguments
- `linear_matrix`: linear matrix
- `quadratic_matrix`: quadratic matrix
- `const_stepsize`: whether to use a constant time step size
- `u2_jm1`: u2 at j-1

# Returns
- `u`: state matrix
"""
function integrate_finite_diff_model(tdata, IC, args...; kwargs...)
    @assert haskey(kwargs, :linear_matrix) "Linear matrix is required"
    @assert haskey(kwargs, :quadratic_matrix) "Quadratic matrix is required"
    @assert haskey(kwargs, :const_stepsize) "Constant step size is required"
    A = kwargs[:linear_matrix]
    F = kwargs[:quadratic_matrix]
    const_stepsize = kwargs[:const_stepsize]

    Xdim = length(IC)
    Tdim = length(tdata)
    u = zeros(Xdim, Tdim)
    u[:, 1] = IC

    if haskey(kwargs, :u2_jm1)
        u2_jm1 = kwargs[:u2_jm1]
    else
        u2_jm1 = nothing
    end

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(1.0I(Xdim) - Δt/2 * A) \ 1.0I(Xdim) # |> sparse
        IpdtA = (1.0I(Xdim) + Δt/2 * A)

        for j in 2:Tdim
            u2 = u[:, j-1] ⊘ u[:, j-1]
            if j == 2 && isnothing(u2_jm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = ImdtA_inv * (
                    IpdtA * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2
                )
            end
            u2_jm1 = u2
        end
    else
        for j in 2:Tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = u[:, j-1] ⊘ u[:, j-1]
            if j == 2 && isnothing(u2_jm1)
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ (
                    (1.0I(Xdim) + Δt/2 * A) * u[:, j-1] 
                    + 
                    F * u2 * Δt
                )
            else
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ (
                    (1.0I(Xdim) + Δt/2 * A) * u[:, j-1] 
                    + 
                    F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2
                )
            end
            u2_jm1 = u2
        end
    end
    return u
end

end