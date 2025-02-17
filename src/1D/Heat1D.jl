"""
    1 Dimensional Heat Equation Model
"""
module Heat1D

using DocStringExtensions
using LinearAlgebra
using SparseArrays

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export Heat1DModel


"""
$(TYPEDEF)

1 Dimensional Heat Equation Model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\frac{\\partial^2 u}{\\partial x^2}
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
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: model integration
"""
mutable struct Heat1DModel <: AbstractModel
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
    IC::AbstractVector{<:Real}  # initial condition

    # grid points
    xspan::AbstractVector{<:Real}  # spatial grid points
    tspan::AbstractVector{<:Real}  # temporal points
    diffusion_coeffs::Union{Real,AbstractArray{<:Real}}  # parameter vector

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Int64  # parameter dimension

    # Functions
    finite_diff_model::Function
    integrate_model::Function
end


"""
$(SIGNATURES)

Constructor of 1D Heat Equation Model

# Arguments
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `diffusion_coeffs::Union{Real,AbstractArray{<:Real}}`: parameter vector
- `BC::Symbol=:dirichlet`: boundary condition

# Returns
- `Heat1D`: 1D heat equation model
"""
function Heat1DModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                     diffusion_coeffs::Union{Real,AbstractArray{<:Real}}, BC::Symbol=:dirichlet)
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

    Heat1DModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, BC, IC, xspan, tspan, diffusion_coeffs,
        spatial_dim, time_dim, param_dim, 
        finite_diff_model, integrate_model
    )
end


"""
$(SIGNATURES)

Finite Difference Model for 1D Heat Equation

# Arguments
- `model::Heat1DModel`: 1D heat equation model
- `μ::Real`: diffusion coefficient

# Returns
- operators
"""
function finite_diff_model(model::Heat1DModel, μ::Real; kwargs...)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, μ)
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, μ; kwargs...)
    elseif model.BC == :neumann
        return finite_diff_neumann_model(model.spatial_dim, model.Δx, μ)
    elseif model.BC == :mixed
        return finite_diff_mixed_model(model.spatial_dim, model.Δx, μ; kwargs...)
    elseif model.BC == :robin
        return finite_diff_robin_model(model.spatial_dim, model.Δx, μ; kwargs...)
    else
        error("Boundary condition not implemented")
    end
end


"""
$(SIGNATURES)

Finite Difference Model for 1D Heat Equation with periodic boundary condition.
"""
function finite_diff_periodic_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, end] = μ / Δx^2  # periodic boundary condition
    A[end, 1] = μ / Δx^2  
    return A
end


"""
$(SIGNATURES)

Finite Difference Model for 1D Heat Equation with Dirichlet boundary condition.
"""
function finite_diff_dirichlet_model(N::Real, Δx::Real, μ::Real; same_on_both_ends::Bool=true)
    A = spdiagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2

    if same_on_both_ends
        # Dirichlet boundary condition which is same at both ends
        B = [1; zeros(N-2,1); 1] * μ / Δx^2  
    else
        # Generalized input matrix for different dirichlet boundary conditions
        # for each end of the domain (u(1) and u(N))
        B = zeros(N,2)
        B[1,1] = μ / Δx^2
        B[end,2] = μ / Δx^2
    end

    return A, B
end


"""
$(SIGNATURES)

Finite difference model for the 1D Heat Equation with Neumann boundary condition.
"""
function finite_diff_neumann_model(N::Real, Δx::Real, μ::Real)
    # Linear state matrix
    A = spdiagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2
    A[1,1] = -1 * μ / Δx^2
    A[end,end] = -1 * μ / Δx^2

    # Input matrix for the boundary conditions
    B = zeros(N,2)
    B[1,1] = -μ / Δx
    B[end,2] = μ / Δx
    
    return A, B
end


"""
$(SIGNATURES)

Finite difference model for the 1D Heat Equation with Mixed boundary condition.
The mixed boundary condition is a combination of Neumann and Dirichlet boundary
conditions. The left boundary is Neumann and the right boundary is Dirichlet when
`order` is set to `["neumann", "dirichlet"]`. If `order` is set to `["dirichlet", "neumann"]`,
then the left boundary is Dirichlet and the right boundary is Neumann.
"""
function finite_diff_mixed_model(N::Real, Δx::Real, μ::Real; order::Vector{<:String}=["neumann", "dirichlet"])
    if order[1] == "neumann"  # Neumann (left) and Dirichlet (right)
        # Linear state matrix
        A = spdiagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2
        A[1,1] = -1 * μ / Δx^2

        # Input matrix for the boundary conditions
        B = zeros(N,2)
        B[1,1] = -μ / Δx
        B[end,2] = μ / Δx^2
    else  # Dirichlet (left) and Neumann (right)
        # Linear state matrix
        A = spdiagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2
        A[end,end] = -1 * μ / Δx^2

        # Input matrix for the boundary conditions
        B = zeros(N,2)
        B[1,1] = μ / Δx^2
        B[end,2] = μ / Δx
    end
    
    return A, B
end


"""
$(SIGNATURES)

Finite difference model for the 1D Heat Equation with Robin boundary condition. 
The Robin boundary condition is a generalization of the Neumann and Dirichlet 
boundary conditions, where the boundary condition is a linear combination of
the state and its derivative. That is,

```math
\\alpha u(0) + (1 - \\alpha) u'(0) = g(t).
```

The weights `α` and `β` are used to adjust the boundary conditions at the left
and right boundaries, respectively.
"""
function finite_diff_robin_model(N::Real, Δx::Real, μ::Real; α::Real=0.5, β::Real=0.5)
    # Linear state matrix
    A = spdiagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2
    A[1,1] += α / (α + α*Δx - Δx) * μ / Δx^2
    A[end,end] += β / (β - β*Δx + Δx) * μ / Δx^2

    # Input matrix for the boundary conditions
    B = zeros(N,2)
    B[1,1] = -Δx / (α + α*Δx - Δx) * μ / Δx^2
    B[end,2] = Δx / (β - β*Δx + Δx) * μ / Δx^2
    
    return A, B
end


"""
$(SIGNATURES)

Integrate the 1D Heat Equation Model using 3 different methods:
- Forward Euler
- Backward Euler
- Crank Nicolson

# Arguments
- `tdata::AbstractVector{T}`: time data
- `x0::AbstractVector{T}`: initial condition
- `u::AbstractArray{T}=[]`: input data

# Keyword Arguments
- `linear_matrix::AbstractArray{T,2}`: linear matrix
- `control_matrix::AbstractArray{T,2}`: control matrix
- `system_input::Bool=false`: system input flag
- `integrator_type::Symbol=:ForwardEuler`: integrator type
- `same_on_both_ends::Bool=true`: same boundary (Dirichlet) condition on both ends

# Returns
- `x::Array{T,2}`: integrated model states

# Notes
- Input is assumed to be a matrix of size (spatial dimension x time dimension).
  You will receive a warning if the input is a tall vector/column vector.
- `operators` should be in the order of [A, B] if `system_input` is true.
"""
function integrate_model(tdata::AbstractVector{T}, u0::AbstractVector{T},
                         input::AbstractArray{T}=T[]; kwargs...) where {T<:Real}
    # Check that keyword exists in kwargs
    @assert haskey(kwargs, :linear_matrix) "Keyword :linear_matrix not found"
    @assert haskey(kwargs, :system_input) "Keyword :system_input not found"
    @assert haskey(kwargs, :integrator_type) "Keyword :integrator_type not found"

    # Unpack the keyword arguments
    A = kwargs[:linear_matrix]
    system_input = kwargs[:system_input]
    integrator_type = kwargs[:integrator_type]

    # Integration settings
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:,1] = u0

    # Adjust input dimensions if system_input is true
    if system_input
        @assert haskey(kwargs, :control_matrix) "Keyword :control_matrix not found"
        B = kwargs[:control_matrix]
        input_dim = size(B, 2)  # Number of inputs

        # Adjust the input
        input = adjust_input(input, input_dim, tdim)
    end

    # Integrate the model
    if integrator_type == :ForwardEuler
        if system_input
            @inbounds for i in 2:tdim
                Δt = tdata[i] - tdata[i-1]
                u[:,i] = (I(xdim) + Δt * A) * u[:,i-1] + Δt * B * input[:,i-1]
            end
        else
            @inbounds for i in 2:tdim
                Δt = tdata[i] - tdata[i-1]
                u[:,i] = (I(xdim) + Δt * A) * u[:,i-1]
            end
        end
    elseif integrator_type == :BackwardEuler
        if system_input
            @inbounds for i in 2:tdim
                Δt = tdata[i] - tdata[i-1]
                u[:,i] = (I(xdim) - Δt * A) \ (u[:,i-1] + Δt * B * input[:,i-1])
            end
        else
            @inbounds for i in 2:tdim
                Δt = tdata[i] - tdata[i-1]
                u[:,i] = (I(xdim) - Δt * A) \ u[:,i-1]
            end
        end
    elseif integrator_type == :CrankNicolson
        if system_input
            @inbounds for i in 2:tdim
                Δt = tdata[i] - tdata[i-1]
                u[:,i] = (I(xdim) - 0.5 * Δt * A) \ ((I(xdim) + 0.5 * Δt * A) * u[:,i-1] + 0.5 * Δt * B * (input[:,i-1] + input[:,i]))
            end
        else
            @inbounds for i in 2:tdim
                Δt = tdata[i] - tdata[i-1]
                u[:,i] = (I(xdim) - 0.5 * Δt * A) \ ((I(xdim) + 0.5 * Δt * A) * u[:,i-1])
            end
        end
    else
        error("Integrator type not implemented. Choose from :ForwardEuler, :BackwardEuler, :CrankNicolson")
    end

    return u
end

end