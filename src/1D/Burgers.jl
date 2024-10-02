"""
    Viscous Burgers' equation model
"""
module Burgers

using DocStringExtensions
using LinearAlgebra
using SparseArrays
using UniqueKronecker

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export BurgersModel


"""
$(TYPEDEF)

Viscous Burgers' equation model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x}
```

# Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Tuple{Real,Real}`: parameter domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `BC::Symbol`: boundary condition
- `IC::Array{Float64}`: initial condition
- `xspan::Vector{Float64}`: spatial grid points
- `tspan::Vector{Float64}`: temporal points
- `diffusion_coeffs::Union{Real,AbstractArray{<:Real}}`: parameter vector
- `spatial_dim::Int64`: spatial dimension
- `time_dim::Int64`: temporal dimension
- `param_dim::Int64`: parameter dimension
- `conservation_type::Symbol`: conservation type
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: model integration

# Note
- The conservation type can be either `:EP` (Energy Preserving), `:NC` (Non-Conservative), or `:C` (Conservative).
"""
mutable struct BurgersModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Tuple{Real,Real}  # parameter domain

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

    # Parameters
    diffusion_coeffs::Union{Real,AbstractArray{<:Real}}  # parameter vector

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Int64  # parameter dimension

    # Convervation type
    conservation_type::Symbol

    # Functions
    finite_diff_model::Function
    integrate_model::Function
end


"""
$(SIGNATURES)

Constructor for the viscous Burgers' equation model.
"""
function BurgersModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                       diffusion_coeffs::Union{Real,AbstractArray{<:Real}}, BC::Symbol=:dirichlet,
                       conservation_type::Symbol=:NC)
    # Discritization grid info
    @assert BC ∈ (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy, :flux) "Invalid boundary condition"
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

    # Parameter dimensions or number of parameters 
    param_dim = length(diffusion_coeffs)
    param_domain = extrema(diffusion_coeffs)

    @assert conservation_type ∈ (:EP, :NC, :C) "Invalid conservation type. Choose from (:EP, :NC, :C)"

    BurgersModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, BC, IC, xspan, tspan, diffusion_coeffs,
        spatial_dim, time_dim, param_dim, conservation_type,
        finite_diff_model, integrate_model
    )
end


"""
$(SIGNATURES)

Finite Difference Model for Burgers equation

# Arguments
- `model::BurgersModel`: Burgers model
- `μ::Real`: diffusion coefficient

# Returns
- operators
"""
function finite_diff_model(model::BurgersModel, μ::Real; kwargs...)
    if model.BC == :periodic
        if model.conservation_type == :NC
            return finite_diff_periodic_nonconservative_model(model.spatial_dim, model.Δx, μ)
        elseif model.conservation_type == :C
            return finite_diff_periodic_conservative_model(model.spatial_dim, model.Δx, μ)
        elseif model.conservation_type == :EP
            return finite_diff_periodic_energy_preserving_model(model.spatial_dim, model.Δx, μ)
        else
            error("Conservation type not implemented")
        end
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, model.Δt, μ; kwargs...)
    else
        error("Boundary condition not implemented")
    end
end


"""
    finite_diff_dirichlet_model(N::Real, Δx::Real, μ::Real) → A, B, F

Generate A, B, F matrices for the Burgers' equation for Dirichlet boundary condition.
This is by default the non-conservative form.
"""
function finite_diff_dirichlet_model(N::Real, Δx::Real, Δt::Real, μ::Float64; 
                                     same_on_both_ends::Bool=false, opposite_sign_on_ends::Bool=true)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, 1:2] = [-1/Δt, 0]
    A[end, end-1:end] = [0, -1/Δt]

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx
    else
        F = zeros(N, S)
    end

    # Create B matrix
    if same_on_both_ends
        B = sparse([1; zeros(N - 2, 1); 1]) ./ Δt
    elseif opposite_sign_on_ends
        B = sparse([1; zeros(N - 2, 1); -1]) ./ Δt
    else
        B = spzeros(N,2)
        B[1,1] = 1 / Δt
        B[end,2] = 1 / Δt
    end

    return A, B, F
end


"""
    finite_diff_periodic_nonconservative_model(N::Real, Δx::Real, μ::Real) → A, F

Generate A, F matrices for the Burgers' equation for periodic boundary condition (Non-conservative).
"""
function finite_diff_periodic_nonconservative_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, end] = μ / Δx^2  # periodic boundary condition
    A[end, 1] = μ / Δx^2  

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx

        F[1, 2] = - 1 / 2 / Δx
        F[1, N] = 1 / 2 / Δx
        F[N, N] = - 1 / 2 / Δx
        F[N, end-1] = 1 / 2 / Δx 
    else
        F = zeros(N, S)
    end

    return A, F
end


"""
    finite_diff_periodic_conservative_model(N::Real, Δx::Real, μ::Real) → A, F

Generate A, F matrices for the Burgers' equation for periodic boundary condition (Conservative form).
"""
function finite_diff_periodic_conservative_model(N::Real, Δx::Real, μ::Float64)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, end] = μ / Δx^2  # periodic boundary condition
    A[end, 1] = μ / Δx^2  

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        ii = repeat(2:(N-1), inner=2)
        m = 2:N-1
        mm = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) for m in 2:N-1])  # this is where the x_{i-1}^2 term is
        mp = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) for m in 2:N-1])  # this is where the x_{i+1}^2 term is
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

    return A, F
end


"""
    finite_diff_periodic_energy_preserving_model(N::Real, Δx::Real, μ::Float64) → A, F

Generate A, F matrices for the Burgers' equation for periodic boundary condition (Energy preserving form).
"""
function finite_diff_periodic_energy_preserving_model(N::Real, Δx::Real, μ::Float64)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, N] = μ / Δx^2  # periodic boundary condition
    A[N, 1] = μ / Δx^2  # periodic boundary condition

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        ii = repeat(2:(N-1), inner=4)
        m = 2:N-1
        mi = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 2:N-1])               # this is where the xi^2 term is
        mm = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) for m in 2:N-1])  # this is where the x_{i-1}^2 term is
        mp = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) for m in 2:N-1])  # this is where the x_{i+1}^2 term is
        jp = mi .+ 1  # this is the index of the x_{i+1}*x_i term
        jm = mm .+ 1  # this is the index of the x_{i-1}*x_i term
        jj = reshape([mp'; mm'; jp'; jm'],4*N-8);
        vv = reshape([-ones(1,N-2); ones(1,N-2); -ones(1,N-2); ones(1,N-2)],4*N-8)/(6*Δx);
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

    return A, F
end


"""
$(SIGNATURES)

Semi-Implicit Euler scheme with control

# Notes
- `operators` should be in the order of `A, F, B`
"""
function integrate_model_with_control(tdata::AbstractArray{T}, u0::AbstractArray{T},
                                      input::AbstractArray{T}; operators) where {T<:Real}
    # Integration settings
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    # Operators
    A, B, F = operators

    # Integration
    for j in 2:tdim
        Δt = tdata[j] - tdata[j-1]
        u2 = u[:, j-1] ⊘ u[:, j-1]
        u[:, j] = (I(xdim) - Δt * A) \ (u[:, j-1] + F * u2 * Δt + B * input[:,j] * Δt)
    end
    return u
end


"""
$(SIGNATURES)

Semi-Implicit Euler scheme without control

# Notes
- `operators` should be in the order of `A, F`
"""
function integrate_model_without_control(tdata::AbstractArray{T}, u0::AbstractArray{T}; 
                                         operators) where {T<:Real}
    # Integration settings
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    # Operators
    A, F = operators

    for j in 2:tdim
        Δt = tdata[j] - tdata[j-1]
        u2 = u[:, j-1] ⊘ u[:, j-1]
        u[:, j] = (1.0I(xdim) - Δt * A) \ (u[:, j-1] + F * u2 * Δt)
    end
    return u
end


"""
$(SIGNATURES)

Integrate the viscous Burgers' equation model

# Arguments
- `tdata::AbstractArray{T}`: time data
- `u0::AbstractArray{T}`: initial condition
- `input::AbstractArray{T}=[]`: input data

# Keyword Arguments
- `operators`: operators A, F, B
- `system_input::Bool=false`: system input flag
- `integrator_type::Symbol=:ForwardEuler`: integrator type

# Returns
- `u::Array{T,2}`: integrated model states

# Notes
- Input is assumed to be a matrix of size (spatial dimension x time dimension).
  You will receive a warning if the input is a tall vector/column vector.
- `operators` should be in the order of [A, B, F] if `system_input` is true. If 
  `system_input` is false, then `operators` should be in the order of [A, F].
"""
function integrate_model(tdata::AbstractArray{T}, u0::AbstractArray{T}, 
                         input::AbstractArray{T}=T[]; kwargs...) where {T<:Real}
    # Check that keyword exists in kwargs
    @assert haskey(kwargs, :operators) "Keyword :operators not found"
    @assert haskey(kwargs, :system_input) "Keyword :system_input not found"
    
    # Unpack the keyword arguments
    operators = kwargs[:operators]
    system_input = kwargs[:system_input]

    # Integration settings
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:,1] = u0

    # Adjust input dimensions if system_input is true
    if system_input
        A, B, F = operators
        input_dim = size(B, 2)  # Number of inputs

        # Adjust the input
        input = adjust_input(input, input_dim, tdim)
    else
        A, F = operators
    end

    if system_input
        return integrate_model_with_control(tdata, u0, input; operators)
    else
        return integrate_model_without_control(tdata, u0; operators)
    end
end

end