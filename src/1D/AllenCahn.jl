"""
    Allen-Cahn equation PDE Model
"""
module AllenCahn

using DocStringExtensions
using LinearAlgebra
using SparseArrays
using UniqueKronecker

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export AllenCahnModel


"""
$(TYPEDEF)

Allen-Cahn equation Model
    
```math
\\frac{\\partial u}{\\partial t} =  \\mu\\frac{\\partial^2 u}{\\partial x^2} - \\epsilon(u^3 - u)
```

where ``u`` is the state variable, ``μ`` is the diffusion coefficient, ``ϵ`` is a nonlinear coefficient.

## Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Tuple{Real,Real}`: parameter domain (diffusion coeff)
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `params::Dict{Symbol,Union{Real,AbstractArray{<:Real}}}`: parameters
- `xspan::Vector{<:Real}`: spatial grid points
- `tspan::Vector{<:Real}`: temporal points
- `spatial_dim::Int`: spatial dimension
- `time_dim::Int`: temporal dimension
- `param_dim::Int`: parameter dimension
- `IC::AbstractArray{<:Real}`: initial condition
- `BC::Symbol`: boundary condition
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
"""
mutable struct AllenCahnModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Dict{Symbol,Tuple{Real,Real}}

    # Discritization grid
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size

    # Grid points
    xspan::Vector{<:Real}  # spatial grid points
    tspan::Vector{<:Real}  # temporal points
    params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}} # parameters

    # Dimensions
    spatial_dim::Int  # spatial dimension
    time_dim::Int  # temporal dimension
    param_dim::Dict{Symbol,<:Int}  # parameter dimension

    # Initial condition
    IC::AbstractArray{<:Real}  # initial condition

    # Boundary condition
    BC::Symbol  # boundary condition

    # Functions
    finite_diff_model::Function  # model using Finite Difference
    integrate_model::Function # integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
end


"""
$(SIGNATURES)

Constructor for the Allen-Cahn equation model.
"""
function AllenCahnModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                    params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}}, BC::Symbol=:periodic)
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
    param_dim = Dict([k => length(v) for (k, v) in params])
    param_domain = Dict([k => extrema(v) for (k,v) in params])

    AllenCahnModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, 
        xspan, tspan, params,
        spatial_dim, time_dim, param_dim, 
        IC, BC,
        finite_diff_model, integrate_model
    )
end


"""
$(SIGNATURES)

Create the matrices A (linear operator) and E (cubic operator) for the Allen-Cahn model.

## Arguments
- `model::AllenCahnModel`: Allen-Cahn model
- `params::Dict`: parameters dictionary
"""
function finite_diff_model(model::AllenCahnModel, params::Dict)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, params)
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, params)
    elseif model.BC == :mixed
        return finite_diff_mixed_model(model.spatial_dim, model.Δx, params)
    else
        error("Boundary condition not implemented")
    end
end


"""
$(SIGNATURES)

Create the matrices A (linear operator) and E (cubic operator) for the Chafee-Infante model.

## Arguments
- `N::Real`: spatial dimension
- `Δx::Real`: spatial grid size
- `params::Dict`: parameters

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `E::SparseMatrixCSC{Float64,Int}`: cubic operator
"""
function finite_diff_periodic_model(N::Real, Δx::Real, params::Dict)
    # parameters
    μ = params[:μ]
    ϵ = params[:ϵ]

    # Create A matrix
    A = spdiagm(0 => (ϵ-2*μ/Δx^2) * ones(N), 1 => (μ/Δx^2) * ones(N - 1), -1 => (μ/Δx^2) * ones(N - 1))
    A[1, end] = μ / Δx^2  # periodic boundary condition
    A[end, 1] = μ / Δx^2  

    # INFO: TAKES TOO LONG
    # # Create E matrix
    # indices = [(i,i,i,i) for i in 1:N]
    # values = [-ϵ for _ in 1:N]
    # E = makeCubicOp(N, indices, values, which_cubic_term='E')

    # Create E matrix
    S = Int(N * (N + 1) * (N + 2) / 6)
    cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:N-1])
    pushfirst!(cubic_idx_diff, 1)
    iii = collect(1:N)
    jjj = cumsum(cubic_idx_diff)  # indices for cubic terms
    vvv = -ϵ * ones(N)
    E = sparse(iii, jjj, vvv, N, S)

    return A, E
end


"""
$(SIGNATURES)
"""
function finite_diff_dirichlet_model(N::Real, Δx::Real, params::Dict)
    # Parameters 
    μ = params[:μ]
    ϵ = params[:ϵ]

    # Create A matrix
    A = spdiagm(0 => (ϵ-2*μ/Δx^2) * ones(N), 1 => (μ/Δx^2) * ones(N - 1), -1 => (μ/Δx^2) * ones(N - 1))

    # Create E matrix
    S = Int(N * (N + 1) * (N + 2) / 6)
    cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:N-1])
    pushfirst!(cubic_idx_diff, 1)
    iii = collect(1:N)
    jjj = cumsum(cubic_idx_diff)  # indices for cubic terms
    vvv = -ϵ * ones(N)
    E = sparse(iii, jjj, vvv, N, S)

    # Create B matrix
    B = spzeros(N,2)
    B[1,1] = μ / Δx^2  # from Dirichlet boundary condition
    B[end,2] = μ / Δx^2  # from Neumann boundary condition

    return A, E, B
end


"""
$(SIGNATURES)

Create the matrices A (linear operator), B (input operator), and E (cubic operator) for Chafee-Infante 
model using the mixed boundary condition. If the spatial domain is [0,1], then we assume u(0,t) to be 
homogeneous dirichlet boundary condition and u(1,t) to be Neumann boundary condition of some function h(t).

## Arguments
- `N::Real`: spatial dimension
- `Δx::Real`: spatial grid size
- `params::Dict`: parameters

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `B::SparseMatrixCSC{Float64,Int}`: input operator
- `E::SparseMatrixCSC{Float64,Int}`: cubic operator
"""
function finite_diff_mixed_model(N::Real, Δx::Real, params::Dict)
    # Parameters 
    μ = params[:μ]
    ϵ = params[:ϵ]

    # Create A matrix
    A = spdiagm(0 => (ϵ-2*μ/Δx^2) * ones(N), 1 => (μ/Δx^2) * ones(N - 1), -1 => (μ/Δx^2) * ones(N - 1))
    A[end,end] = ϵ - μ/Δx^2  # influence of Neumann boundary condition

    # Create E matrix
    S = Int(N * (N + 1) * (N + 2) / 6)
    cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:N-1])
    pushfirst!(cubic_idx_diff, 1)
    iii = collect(1:N)
    jjj = cumsum(cubic_idx_diff)  # indices for cubic terms
    vvv = -ϵ * ones(N)
    E = sparse(iii, jjj, vvv, N, S)

    # Create B matrix
    B = spzeros(N,2)
    B[1,1] = μ / Δx^2  # from Dirichlet boundary condition
    B[end,2] = μ / Δx  # from Neumann boundary condition

    return A, E, B
end


"""
$(SIGNATURES)
    
Integrate the Allen-Cahn model using the Crank-Nicholson (linear) Explicit (nonlinear) method.
Or, in other words, Semi-Implicit Crank-Nicholson (SICN) method without control.
"""
function integrate_model_without_control_SICN(tdata, u0; linear_matrix, cubic_matrix, const_stepsize=false)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0 

    A = linear_matrix
    E = cubic_matrix

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            u3 = ⊘(u[:, j-1], 3)
            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * Δt)
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u3 = ⊘(u[:, j-1], 3)
            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + E * u3 * Δt)
        end
    end
    return u
end


"""
$(SIGNATURES)
    
Integrate the Allen-Cahn model using the Crank-Nicholson (linear) Explicit (nonlinear) method.
Or Semi-Implicit Crank-Nicholson (SICN) method with control input.
"""
function integrate_model_with_control_SICN(tdata, u0, input; linear_matrix, cubic_matrix, 
                                           control_matrix, const_stepsize=false)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0 

    A = linear_matrix
    E = cubic_matrix
    B = control_matrix

    # Adjust the input
    input_dim = size(B, 2)  # Number of inputs
    input = adjust_input(input, input_dim, tdim)

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            u3 = ⊘(u[:, j-1], 3)
            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u3 = ⊘(u[:, j-1], 3)
            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + E * u3 * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Allen-Cahn model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method (CNAB) with control.
"""
function integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix, cubic_matrix, 
                                           control_matrix, const_stepsize=false, u3_jm1=nothing)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    E = cubic_matrix
    B = control_matrix

    # Adjust the input
    input_dim = size(B, 2)  # Number of inputs
    input = adjust_input(input, input_dim, tdim)

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim 
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && isnothing(u3_jm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * Δt + 0.5 * B * (input[:,j-1]+input[:,j]) * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2 + 0.5 * B * (input[:,j-1] + input[:,j]) * Δt)
            end
            u3_jm1 = u3
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && isnothing(u3_jm1)
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + E * u3 * Δt + 0.5 * B * (input[:,j-1]+input[:,j]) * Δt)
            else
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2 + 
                                             0.5 * B * (input[:,j-1] + input[:,j]) * Δt)
            end
            u3_jm1 = u3
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Allen-Cahn model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method (CNAB)
without control.
"""
function integrate_model_without_control_CNAB(tdata, u0; linear_matrix, cubic_matrix, 
                                              const_stepsize=false, u3_jm1=nothing)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    E = cubic_matrix

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim 
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && isnothing(u3_jm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2)
            end
            u3_jm1 = u3
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && isnothing(u3_jm1)
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + E * u3 * Δt)
            else
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2)
            end
            u3_jm1 = u3
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Allen-Cahn model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method (CNAB) or
Semi-Implicit Crank-Nicolson (SICN) method.

# Arguments
- `tdata::AbstractArray{T}`: time data
- `u0::AbstractArray{T}`: initial condition
- `input::AbstractArray{T}=[]`: input data

# Keyword Arguments
- `linear_matrix::AbstractArray{T,2}`: linear matrix
- `cubic_matrix::AbstractArray{T,2}`: cubic matrix
- `control_matrix::AbstractArray{T,2}`: control matrix
- `system_input::Bool=false`: system input flag
- `const_stepsize::Bool=false`: constant step size flag
- `u3_jm1::AbstractArray{T}=[]`: previous cubic term

# Returns
- `x::Array{T,2}`: integrated model states

# Notes
- If `system_input` is true, then `control_matrix` must be provided.
- If `const_stepsize` is true, then the time step size is assumed to be constant.
- If `u3_jm1` is provided, then the cubic term at the previous time step is used.
- `integrator_type` can be either `:SICN` for Semi-Implicit Crank-Nicolson or `:CNAB` for Crank-Nicolson Adam-Bashforth
"""
function integrate_model(tdata::AbstractArray{T}, u0::AbstractArray{T}, input::AbstractArray{T}=T[]; kwargs...) where {T<:Real}
    # Check that keyword exists in kwargs
    @assert haskey(kwargs, :linear_matrix) "Keyword :linear_matrix not found"
    @assert haskey(kwargs, :cubic_matrix) "Keyword :cubic_matrix not found"
    if !isempty(input)
        @assert haskey(kwargs, :control_matrix) "Keyword :control_matrix not found"
    end

    # Unpack the keyword arguments
    linear_matrix = kwargs[:linear_matrix]
    cubic_matrix = kwargs[:cubic_matrix]
    control_matrix = haskey(kwargs, :control_matrix) ? kwargs[:control_matrix] : nothing
    system_input = haskey(kwargs, :system_input) ? kwargs[:system_input] : !isempty(input)
    const_stepsize = haskey(kwargs, :const_stepsize) ? kwargs[:const_stepsize] : false
    u3_jm1 = haskey(kwargs, :u3_jm1) ? kwargs[:u3_jm1] : nothing
    if haskey(kwargs, :integrator_type)
        integrator_type = kwargs[:integrator_type]
        @assert integrator_type ∈ (:SICN, :CNAB) "Invalid integrator type. Choose from (:SICN, :CNAB), where SICN is Semi-Implicit Crank-Nicolson and CNAB is Crank-Nicolson Adam-Bashforth"
    else
        integrator_type = :CNAB
    end

    if system_input
        if integrator_type == :SICN
            integrate_model_with_control_SICN(tdata, u0, input; linear_matrix=linear_matrix, cubic_matrix=cubic_matrix, control_matrix=control_matrix)
        else
            integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix=linear_matrix, cubic_matrix=cubic_matrix, 
                                              control_matrix=control_matrix, const_stepsize=const_stepsize, u3_jm1=u3_jm1)
        end
    else
        if integrator_type == :SICN
            integrate_model_without_control_SICN(tdata, u0; linear_matrix=linear_matrix, cubic_matrix=cubic_matrix)
        else
            integrate_model_without_control_CNAB(tdata, u0; linear_matrix=linear_matrix, cubic_matrix=cubic_matrix,
                                                 const_stepsize=const_stepsize, u3_jm1=u3_jm1)
        end
    end
end

end