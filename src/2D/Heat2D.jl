"""
    2D Heat Equation Model
"""
module Heat2D

using DocStringExtensions
using Kronecker: ⊗
using LinearAlgebra
using SparseArrays

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export Heat2DModel

"""
$(TYPEDEF)

2 Dimensional Heat Equation Model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\left(\\frac{\\partial^2 u}{\\partial x^2} + \\frac{\\partial^2 u}{\\partial y^2}\\right)
```

## Fields
- `spatial_domain::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}`: spatial domain (x, y)
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Tuple{Real,Real}`: parameter domain
- `Δx::Real`: spatial grid size (x-axis)
- `Δy::Real`: spatial grid size (y-axis)
- `Δt::Real`: temporal step size
- `spatial_dim::Tuple{Int64,Int64}`: spatial dimension x and y
- `time_dim::Int64`: temporal dimension
- `param_dim::Int64`: parameter dimension
- `BC::Tuple{Symbol,Symbol}`: boundary condition
- `IC::Array{Float64}`: initial condition
- `diffusion_coeffs::Union{AbstractArray{<:Real},Real}`: diffusion coefficients
- `xspan::Vector{Float64}`: spatial grid points (x-axis)
- `yspan::Vector{Float64}`: spatial grid points (y-axis)
- `tspan::Vector{Float64}`: temporal points
- `finite_diff_model::Function`: finite difference model
- `integrate_model::Function`: integrate model
"""
mutable struct Heat2DModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}  # spatial domain (x, y)
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Tuple{Real,Real}  # parameter domain

    # Grids
    Δx::Real  # spatial grid size (x-axis)
    Δy::Real  # spatial grid size (y-axis)
    Δt::Real  # temporal step size

    # Dimensions
    spatial_dim::Tuple{Int64,Int64}  # spatial dimension x and y
    time_dim::Int64  # temporal dimension
    param_dim::Int64  # parameter dimension

    # Boundary and Initial Conditions
    BC::Tuple{Symbol,Symbol}  # boundary condition
    IC::Array{Float64}  # initial condition

    # Parameters
    diffusion_coeffs::Union{AbstractArray{<:Real},Real} # diffusion coefficients

    # Data
    xspan::Vector{Float64}  # spatial grid points (x-axis)
    yspan::Vector{Float64}  # spatial grid points (y-axis)
    tspan::Vector{Float64}  # temporal points

    # Functions
    finite_diff_model::Function
    integrate_model::Function
end


function Heat2DModel(;spatial_domain::Tuple{Tuple{Real,Real},Tuple{Real,Real}}, time_domain::Tuple{Real,Real}, 
                 Δx::Real, Δy::Real, Δt::Real, diffusion_coeffs::Union{AbstractArray{<:Real},Real}, BC::Tuple{Symbol,Symbol})
    # Discritization grid info
    possible_BC = (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy, :flux)
    @assert all([BC[i] ∈ possible_BC for i in eachindex(BC)]) "Invalid boundary condition"
    # x-axis
    if BC[1] == :periodic
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2]-Δx)
    elseif BC[1] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2])[2:end-1]
    end
    # y-axis
    if BC[2] == :periodic
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2]-Δy)
    elseif BC[2] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2])[2:end-1]
    end
    tspan = collect(time_domain[1]:Δt:time_domain[2])
    spatial_dim = (length(xspan), length(yspan))
    time_dim = length(tspan)

    # Initial condition
    IC = zeros(prod(spatial_dim))

    # Parameter dimensions or number of parameters 
    param_dim = length(diffusion_coeffs)
    param_domain = extrema(diffusion_coeffs)

    Heat2DModel(spatial_domain, time_domain, param_domain, Δx, Δy, Δt,
           spatial_dim, time_dim, param_dim, BC, IC, diffusion_coeffs,
           xspan, yspan, tspan, 
           finite_diff_model, integrate_model)
end


function finite_diff_dirichlet_model(Nx::Integer, Ny::Integer, Δx::Real, Δy::Real, μ::Real)
    # A matrix
    Ax = spdiagm(0 => (-2)*ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1)) * μ / Δx^2
    Ay = spdiagm(0 => (-2)*ones(Ny), 1 => ones(Ny-1), -1 => ones(Ny-1)) * μ / Δy^2
    A = (Ay ⊗ I(Nx)) + (I(Ny) ⊗ Ax)

    # B matrix (different inputs for each boundary)
    # Bx matrix (left and right boundaries)
    Bx = spzeros(Nx*Ny, 2)

    # Left boundary indices (x = 0)
    left_indices = [ (j - 1) * Nx + 1 for j in 1:Ny ]
    Bx[left_indices, 1] .= μ / Δx^2

    # Right boundary indices (x = Lx)
    right_indices = [ (j - 1) * Nx + Nx for j in 1:Ny ]
    Bx[right_indices, 2] .= μ / Δx^2

    # By matrix (bottom and top boundaries)
    By = spzeros(Nx*Ny, 2)

    # Bottom boundary indices (y = 0)
    bottom_indices = [ i for i in 1:Nx ]
    By[bottom_indices, 1] .= μ / Δy^2

    # Top boundary indices (y = Ly)
    top_indices = [ (Ny - 1) * Nx + i for i in 1:Nx ]
    By[top_indices, 2] .= μ / Δy^2

    # Combine B matrices
    B = hcat(Bx, By)

    return A, B
end


"""
$(SIGNATURES)

Generate A and B matrices for the 2D heat equation.

## Arguments
- `model::heat2d`: 2D heat equation model
- `μ::Real`: diffusion coefficient

## Returns
- `A::Matrix{Float64}`: A matrix
- `B::Matrix{Float64}`: B matrix
"""
function finite_diff_model(model::Heat2DModel, μ::Real)
    if all(model.BC .== :dirichlet)
        return finite_diff_dirichlet_model(model.spatial_dim..., model.Δx, model.Δy, μ)
    else
        error("Not implemented")
    end
end



"""
$(SIGNATURES)

Integrate the 2D heat equation model.

## Arguments
- `A::Matrix{Float64}`: A matrix
- `B::Matrix{Float64}`: B matrix
- `U::Vector{Float64}`: input vector
- `tdata::Vector{Float64}`: time points
- `IC::Vector{Float64}`: initial condition

## Returns
- `state::Matrix{Float64}`: state matrix
"""
function integrate_model(A, B, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = Matrix{Float64}(undef, Xdim, Tdim)
    state[:,1] = IC
    @inbounds for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        state[:,j] = (I - Δt * A) \ (state[:,j-1] + B * U[:,j-1] * Δt)
    end
    return state
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
- `operators`: operators A and B
- `system_input::Bool=false`: system input flag
- `integrator_type::Symbol=:ForwardEuler`: integrator type

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
    @assert haskey(kwargs, :operators) "Keyword :operators not found"
    @assert haskey(kwargs, :system_input) "Keyword :system_input not found"
    @assert haskey(kwargs, :integrator_type) "Keyword :integrator_type not found"

    # Unpack the keyword arguments
    operators = kwargs[:operators]
    system_input = kwargs[:system_input]
    integrator_type = kwargs[:integrator_type]

    # Integration settings
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:,1] = u0

    # Adjust input dimensions if system_input is true
    if system_input
        A, B = operators
        input_dim = size(B, 2)  # Number of inputs

        # Adjust the input
        input = adjust_input(input, input_dim, tdim)
    else
        A = operators
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