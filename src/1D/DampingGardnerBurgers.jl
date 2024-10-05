"""
    Damping Gardner-Burgers (DGB) PDE model
"""
module DampingGardnerBurgers

using DocStringExtensions
using LinearAlgebra
using SparseArrays
using UniqueKronecker

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export DampingGardnerBurgersModel

"""
$(TYPEDEF)

Gardner equation model

```math
\\frac{\\partial u}{\\partial t} = -\\alpha\\frac{\\partial^3 u}{\\partial x^3} + \beta u\\frac{\\partial u}{\\partial x} + \\gamma u^2\\frac{\\partial u}{\\partial x} + \\delta \\frac{\\partial^2 u}{\\partial x^2} + \\epsilon u
```

## Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Tuple{Real,Real}`: parameter domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `BC::Symbol`: boundary condition
- `IC::Array{Float64}`: initial condition
- `xspan::Vector{Float64}`: spatial grid points
- `tspan::Vector{Float64}`: temporal points
- `spatial_dim::Int64`: spatial dimension
- `time_dim::Int64`: temporal dimension
- `params::Union{Real,AbstractArray{<:Real}}`: parameter vector
- `param_dim::Int64`: parameter dimension
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: model integration
"""
mutable struct DampingGardnerBurgersModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Dict{Symbol,Tuple{Real,Real}}  # parameter domain

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
    params::Dict{Symbol,Union{Real,AbstractArray{<:Real}}} # parameters

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Dict{Symbol,Int64} # parameter dimension

    finite_diff_model::Function
    integrate_model::Function
end


"""
$(SIGNATURES)

Constructor for the Gardner equation model.
"""
function DampingGardnerBurgersModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                       params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}}, BC::Symbol=:dirichlet)
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

    DampingGardnerBurgersModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, BC, IC, xspan, tspan, params,
        spatial_dim, time_dim, param_dim,
        finite_diff_model, integrate_model
    )
end


"""
$(SIGNATURES)

Finite Difference Model for Gardner equation

## Arguments
- `model::DampingGardnerBurgersModel`: Gardner model
- `params::Real`: params including a, b, c

## Returns
- operators
"""
function finite_diff_model(model::DampingGardnerBurgersModel, params::Dict)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, params)
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, model.Δt, params)
    else
        error("Boundary condition not implemented")
    end
end


"""
$(SIGNATURES)

Generate A, F, E, B matrices for the Gardner equation for Dirichlet boundary condition.
"""
function finite_diff_dirichlet_model(N::Real, Δx::Real, Δt::Real, params::Dict)
    # Create A matrix
    α = params[:a]
    β = params[:b]
    γ = params[:c]
    δ = params[:d]
    ϵ = params[:e]

    A = spdiagm(
        -2 => (0.5 * α / Δx^3) * ones(N - 2),
        -1 => (δ / Δx^2 - α / Δx^3) * ones(N - 1),
        0 => (ϵ -2 * δ / Δx^2) * ones(N),
        1 => (α / Δx^3 + δ / Δx^2) * ones(N - 1), 
        2 => (-0.5 * α / Δx^3) * ones(N - 2)
    )
    A[1, 1:3] = [-1/Δt, 0, 0]
    A[2, 1:4] = [0, -1/Δt, 0, 0]
    A[end-1, end-3:end] = [0, 0, -1/Δt, 0]
    A[end, end-2:end] = [0, 0, -1/Δt]

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
    F *= β

    # Create E matrix
    S = Int(N * (N + 1) * (N + 2) / 6)
    if N >= 3
        cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:N-1])
        pushfirst!(cubic_idx_diff, 1)
        cubic_idx = cumsum(cubic_idx_diff)
        iii = repeat(3:(N-2), inner=2)
        jjjp = cubic_idx[3:end-2] .+ 1
        jjjm = cubic_idx[2:end-3] + range(N-1, step=-1, stop=4)
        jjj = reshape([jjjm'; jjjp'], 2N-8)
        vvv = reshape([ones(1,N-4); -ones(1,N-4)], 2N-8) * γ / 2 / Δx
        E = sparse(iii, jjj, vvv, N, S)
    else 
        E = spzeros(N, S)
    end

    # Create B matrix
    B = spzeros(N, 2)
    B[1,1] = 1 / Δt
    B[2,1] = 1 / Δt
    B[end,2] = 1 / Δt
    B[end-1,2] = 1 / Δt

    return A, F, E, B
end


"""
$(SIGNATURES)

Generate A, F, E matrices for the Gardner equation for periodic boundary condition (Non-conservative).
"""
function finite_diff_periodic_model(N::Real, Δx::Real, params::Dict)
    # Create A matrix
    α = params[:a]
    β = params[:b]
    γ = params[:c]
    δ = params[:d]
    ϵ = params[:e]

    A = spdiagm(
        -2 => (0.5 * α / Δx^3) * ones(N - 2),
        -1 => (δ / Δx^2 - α / Δx^3) * ones(N - 1),
        0 => (ϵ -2 * δ / Δx^2) * ones(N),
        1 => (α / Δx^3 + δ / Δx^2) * ones(N - 1), 
        2 => (-0.5 * α / Δx^3) * ones(N - 2)
    )
    # Periodic boundary condition
    A[1, end-1] = 0.5 * α / Δx^3
    A[1, end] = δ / Δx^2 - α / Δx^3 
    A[2, end] = 0.5 * α / Δx^3
    A[end-1, 1] = -0.5 * α / Δx^3
    A[end, 1] = α / Δx^3 + δ / Δx^2
    A[end, 2] = -0.5 * α / Δx^3

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
    F *= β

    # Create E matrix
    S = Int(N * (N + 1) * (N + 2) / 6)
    if N >= 3
        cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:N-1])
        pushfirst!(cubic_idx_diff, 1)
        cubic_idx = cumsum(cubic_idx_diff)
        iii = repeat(3:(N-2), inner=2)
        jjjp = cubic_idx[3:end-2] .+ 1
        jjjm = cubic_idx[2:end-3] + range(N-1, step=-1, stop=4)
        jjj = reshape([jjjm'; jjjp'], 2N-8)
        vvv = reshape([ones(1,N-4); -ones(1,N-4)], 2N-8) * γ / 2 / Δx
        E = sparse(iii, jjj, vvv, N, S)
    else 
        E = spzeros(N, S)
    end
    # BC
    E[1,2] = -β / 2 / Δx  # x_1^2 x_2
    E[1,N] = β / 2 / Δx  # x_1 x_N^2
    E[N,end-1] = β / 2 / Δx  # x_N^2 x_{N-1}
    E[N,Int(N*(N+1)÷2)] = -β / 2 / Δx  # x_N x_1^2

    return A, F, E
end


"""
$(SIGNATURES)

Semi-Implicit Euler scheme with control
"""
function integrate_model_with_control_SIE(tdata, u0, input; linear_matrix, quadratic_matrix, 
                                      cubic_matrix, control_matrix)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    F = quadratic_matrix
    E = cubic_matrix
    B = control_matrix

    # Adjust the input
    input_dim = size(B, 2)  # Number of inputs
    input = adjust_input(input, input_dim, tdim)

    for j in 2:tdim
        Δt = tdata[j] - tdata[j-1]
        u2 = u[:, j-1] ⊘ u[:, j-1]
        u3 = ⊘(u[:, j-1], 3)
        u[:, j] = (1.0I(xdim) - Δt * A) \ (u[:, j-1] + F * u2 * Δt + E * u3 * Δt + B * input[:,j-1] * Δt)
    end
    return u
end


"""
$(SIGNATURES)

Crank-Nicolson Adam-Bashforth (CNAB) scheme with control
"""
function integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix, quadratic_matrix, cubic_matrix, 
                                           control_matrix, const_stepsize=false, u2_jm1=nothing, u3_jm1=nothing)
    # Unpack matrices
    A = linear_matrix
    F = quadratic_matrix
    B = control_matrix
    E = cubic_matrix

    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    # Adjust the input
    input_dim = size(B, 2)  # Number of inputs
    input = adjust_input(input, input_dim, tdim)

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(1.0I(xdim) - Δt/2 * A) \ 1.0I(xdim) # |> sparse
        IpdtA = (1.0I(xdim) + Δt/2 * A)

        for j in 2:tdim
            u2 = ⊘(u[:, j-1], 2)
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && (isnothing(u3_jm1) || isnothing(u2_jm1))
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt +  E * u3 * Δt + 0.5 * Δt * B * (input[:,j-1] + input[:,j]))
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + 
                                       F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2 +
                                       E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2 + 
                                       0.5 * Δt * B * (input[:,j-1] + input[:,j]))
            end
            u2_jm1 = u2
            u3_jm1 = u3
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = ⊘(u[:, j-1], 2)
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && isnothing(u3_jm1)
                u[:, j] = (1.0I(xdim) - Δt/2 * A) \ ((1.0I(xdim) + Δt/2 * A) * u[:, j-1] + 
                                                     F * u2 * Δt + E * u3 * Δt + 0.5 * Δt * B * (input[:,j-1] + input[:,j]))
            else
                u[:, j] = (1.0I(xdim) - Δt/2 * A) \ ((1.0I(xdim) + Δt/2 * A) * u[:, j-1] + 
                                                     F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2 +
                                                     E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2 
                                                     + 0.5 * Δt * B * (input[:,j-1] + input[:,j]))
            end
            u2_jm1 = u2
            u3_jm1 = u3
        end
    end
    return u
end


"""
$(SIGNATURES)

Semi-Implicit Euler scheme without control
"""
function integrate_model_without_control_SIE(tdata, u0; linear_matrix, quadratic_matrix, cubic_matrix)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    F = quadratic_matrix
    E = cubic_matrix

    for j in 2:tdim
        Δt = tdata[j] - tdata[j-1]
        u2 = u[:, j-1] ⊘ u[:, j-1]
        u3 = ⊘(u[:, j-1], 3)
        u[:, j] = (1.0I(xdim) - Δt * A) \ (u[:, j-1] + F * u2 * Δt + E * u3 * Δt)
    end
    return u
end

"""
$(SIGNATURES)

Crank-Nicolson Adam-Bashforth (CNAB) scheme without control
"""
function integrate_model_without_control_CNAB(tdata, u0; linear_matrix, quadratic_matrix, cubic_matrix, 
                                              const_stepsize=false, u2_jm1=nothing, u3_jm1=nothing)
    # Unpack matrices
    A = linear_matrix
    F = quadratic_matrix
    E = cubic_matrix

    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(1.0I(xdim) - Δt/2 * A) \ 1.0I(xdim) # |> sparse
        IpdtA = (1.0I(xdim) + Δt/2 * A)

        for j in 2:tdim
            u2 = ⊘(u[:, j-1], 2)
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && (isnothing(u3_jm1) || isnothing(u2_jm1))
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt +  E * u3 * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + 
                                       F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2 +
                                       E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2)
            end
            u2_jm1 = u2
            u3_jm1 = u3
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = ⊘(u[:, j-1], 2)
            u3 = ⊘(u[:, j-1], 3)
            if j == 2 && isnothing(u3_jm1)
                u[:, j] = (1.0I(xdim) - Δt/2 * A) \ ((1.0I(xdim) + Δt/2 * A) * u[:, j-1] + F * u2 * Δt + E * u3 * Δt)
            else
                u[:, j] = (1.0I(xdim) - Δt/2 * A) \ ((1.0I(xdim) + Δt/2 * A) * u[:, j-1] + 
                                                     F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2 +
                                                     E * u3 * 3*Δt/2 - E * u3_jm1 * Δt/2)
            end
            u2_jm1 = u2
            u3_jm1 = u3
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Damping Gardner-Burgers model using 2 different methods:
- Semi-Implicit Euler (SIE)
- Crank-Nicolson Adam-Bashforth (CNAB)

# Arguments
- `tdata::AbstractArray{T}`: time data
- `u0::AbstractArray{T}`: initial condition
- `input::AbstractArray{T}=[]`: input data

# Keyword Arguments
- `linear_matrix::AbstractArray{T,2}`: linear matrix
- `quadratic_matrix::AbstractArray{T,2}`: quadratic matrix
- `cubic_matrix::AbstractArray{T,2}`: cubic matrix
- `control_matrix::AbstractArray{T,2}`: control matrix
- `system_input::Bool=false`: system input
- `const_stepsize::Bool=false`: constant step size
- `u2_jm1::Union{Nothing,AbstractArray{T}}=nothing`: u2 at j-1
- `u3_jm1::Union{Nothing,AbstractArray{T}}=nothing`: u3 at j-1
- `integrator_type::Symbol=:CNAB`: integrator type

# Returns
- `u::AbstractArray{T}`: solution

# Notes
- If `system_input=true`, then the control matrix must be provided
- If `const_stepsize=true`, then the time step size is assumed to be constant
- If `u2_jm1` and `u3_jm1` are provided, then the CNAB scheme is used
- If `integrator_type=:SIE`, then the Semi-Implicit Euler scheme is used
- If `integrator_type=:CNAB`, then the Crank-Nicolson Adam-Bashforth scheme is used
"""
function integrate_model(tdata::AbstractArray{T}, u0::AbstractArray{T}, input::AbstractArray{T}=T[]; kwargs...) where {T<:Real}
    # Check that keyword exists in kwargs
    @assert haskey(kwargs, :linear_matrix) "Keyword :linear_matrix not found"
    @assert haskey(kwargs, :quadratic_matrix) "Keyword :quadratic_matrix not found"
    @assert haskey(kwargs, :cubic_matrix) "Keyword :cubic_matrix not found"
    if !isempty(input)
        @assert haskey(kwargs, :control_matrix) "Keyword :control_matrix not found"
    end

    # Unpack the keyword arguments
    linear_matrix = kwargs[:linear_matrix]
    quadratic_matrix = kwargs[:quadratic_matrix]
    cubic_matrix = kwargs[:cubic_matrix]
    control_matrix = haskey(kwargs, :control_matrix) ? kwargs[:control_matrix] : nothing
    system_input = haskey(kwargs, :system_input) ? kwargs[:system_input] : !isempty(input)
    const_stepsize = haskey(kwargs, :const_stepsize) ? kwargs[:const_stepsize] : false
    u2_jm1 = haskey(kwargs, :u2_jm1) ? kwargs[:u2_jm1] : nothing
    u3_jm1 = haskey(kwargs, :u3_jm1) ? kwargs[:u3_jm1] : nothing
    if haskey(kwargs, :integrator_type)
        integrator_type = kwargs[:integrator_type]
        @assert integrator_type ∈ (:SIE, :CNAB) "Invalid integrator type. Choose from (:SIE, :CNAB), where SIE is Semi-Implicit Euler and CNAB is Crank-Nicolson Adam-Bashforth"
    else
        integrator_type = :CNAB
    end

    if system_input
        if integrator_type == :SIE
            integrate_model_with_control_SIE(tdata, u0, input; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix, 
                                             cubic_matrix=cubic_matrix, control_matrix=control_matrix)
        else
            integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix, cubic_matrix=cubic_matrix, 
                                              control_matrix=control_matrix, const_stepsize=const_stepsize, u2_jm1=u2_jm1, u3_jm1=u3_jm1)
        end
    else
        if integrator_type == :SIE
            integrate_model_without_control_SIE(tdata, u0; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix, cubic_matrix=cubic_matrix)
        else
            integrate_model_without_control_CNAB(tdata, u0; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix, cubic_matrix=cubic_matrix,
                                                 const_stepsize=const_stepsize, u2_jm1=u2_jm1, u3_jm1=u3_jm1)
        end
    end
end


end
