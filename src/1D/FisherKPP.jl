"""
    Fisher Kolmogorov-Petrovsky-Piskunov equation (Fisher-KPP) model
"""
module FisherKPP

using DocStringExtensions
using LinearAlgebra
using SparseArrays
using UniqueKronecker

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export FisherKPPModel


"""
$(TYPEDEF)

Fisher Kolmogorov-Petrovsky-Piskunov equation (Fisher-KPP) model is a reaction-diffusion equation or
logistic diffusion process in population dynamics. The model is given by the following PDE: 
    
```math
\\frac{\\partial u}{\\partial t} =  D\\frac{\\partial^2 u}{\\partial x^2} + ru(1-u)
```

where ``u`` is the state variable, ``D`` is the diffusion coefficient, and ``r`` is the growth rate.

## Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `diffusion_coeff_domain::Tuple{Real,Real}`: parameter domain (diffusion coeff)
- `growth_rate_domain::Tuple{Real,Real}`: parameter domain (growth rate)
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `xspan::Vector{<:Real}`: spatial grid points
- `tspan::Vector{<:Real}`: temporal points
- `spatial_dim::Int`: spatial dimension
- `time_dim::Int`: temporal dimension
- `diffusion_coeffs::Union{AbstractArray{<:Real},Real}`: diffusion coefficient
- `growth_rates::Union{AbstractArray{<:Real},Real}`: growth rate
- `param_dim::Dict{Symbol,<:Int}`: parameter dimension
- `IC::AbstractArray{<:Real}`: initial condition
- `BC::Symbol`: boundary condition
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
"""
mutable struct FisherKPPModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Dict{Symbol,Tuple{Real,Real}}

    # Discritization grid
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size

    xspan::Vector{<:Real}  # spatial grid points
    tspan::Vector{<:Real}  # temporal points
    params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}}  # parameters

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


function FisherKPPModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                        params::Dict{Symbol,<:Union{AbstractArray{<:Real},Real}}, BC::Symbol=:periodic)
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

    FisherKPPModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, xspan, tspan, params,
        spatial_dim, time_dim, param_dim, IC, BC,
        finite_diff_model, integrate_model
    )
end


"""
    finite_diff_model(model::fisherkpp, D::Real, r::Real)

Create the matrices A (linear operator) and F (quadratic operator) for the Fisher-KPP model. For different
boundary conditions, the matrices are created differently.

## Arguments
- `model::FisherKPPModel`: Fisher-KPP model
- `params::Dict`: parameters for the model
"""
function finite_diff_model(model::FisherKPPModel, params::Dict)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, params)
    elseif model.BC == :mixed
        return finite_diff_mixed_model(model.spatial_dim, model.Δx, params)
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, params)
    else
        error("Boundary condition not implemented")
    end
end


"""
$(SIGNATURES)
"""
function finite_diff_periodic_model(N::Real, Δx::Real, params::Dict)
    D = params[:D]
    r = params[:r]

    # Create A matrix
    A = spdiagm(0 => (r-2*D/Δx^2) * ones(N), 1 => (D/Δx^2) * ones(N - 1), -1 => (D/Δx^2) * ones(N - 1))
    A[1, end] = D / Δx^2  # periodic boundary condition
    A[end, 1] = D / Δx^2  

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    ii = 1:N  # row index
    jj = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 1:N])  # col index where the xi^2 term is
    vv = -r * ones(N);
    F = sparse(ii,jj,vv,N,S)

    return A, F
end


"""
$(SIGNATURES)
"""
function finite_diff_dirichlet_model(N::Real, Δx::Real, params::Dict)
    D = params[:D]
    r = params[:r]

    # Create A matrix
    A = spdiagm(0 => (r-2*D/Δx^2) * ones(N), 1 => (D/Δx^2) * ones(N - 1), -1 => (D/Δx^2) * ones(N - 1))

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    ii = 1:N  # row index
    jj = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 1:N])  # col index where the xi^2 term is
    vv = -r * ones(N);
    F = sparse(ii,jj,vv,N,S)

    # Create B matrix
    B = spzeros(N,2)
    B[1,1] = D / Δx^2  # from Dirichlet boundary condition
    B[end,2] = D / Δx^2  # from Neumann boundary condition

    return A, F, B
end


"""
$(SIGNATURES)
"""
function finite_diff_mixed_model(N::Real, Δx::Real, params::Dict)
    D = params[:D]
    r = params[:r]

    # Create A matrix
    A = spdiagm(0 => (r-2*D/Δx^2) * ones(N), 1 => (D/Δx^2) * ones(N - 1), -1 => (D/Δx^2) * ones(N - 1))
    A[end,end] = r - D/Δx^2  # influence of Neumann boundary condition

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    ii = 1:N  # row index
    jj = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 1:N])  # col index where the xi^2 term is
    vv = -r * ones(N);
    F = sparse(ii,jj,vv,N,S)

    # Create B matrix
    B = spzeros(N,2)
    B[1,1] = D / Δx^2  # from Dirichlet boundary condition
    B[end,2] = D / Δx  # from Neumann boundary condition

    return A, F, B
end


"""
$(SIGNATURES)
    
Integrate the Fisher-KPP model using the Crank-Nicholson (linear) Explicit (nonlinear) method.
Or Semi-Implicit Crank-Nicholson (SICN) method.
"""
function integrate_model_without_control_SICN(tdata, u0; linear_matrix, quadratic_matrix, const_stepsize=false)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0 

    A = linear_matrix
    F = quadratic_matrix

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            u2 = ⊘(u[:, j-1], 2)
            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = ⊘(u[:, j-1], 2)
            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * Δt)
        end
    end
    return u
end


"""
$(SIGNATURES)
    
Integrate the Fisher-KPP model using the Crank-Nicholson (linear) Explicit (nonlinear) method.
Or Semi-Implicit Crank-Nicholson (SICN) method with control input
"""
function integrate_model_with_control_SICN(tdata, u0, input; linear_matrix, quadratic_matrix, 
                                           control_matrix, const_stepsize=false)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0 

    A = linear_matrix
    E = quadratic_matrix
    B = control_matrix

    # Adjust the input
    input_dim = size(B, 2)  # Number of inputs
    input = adjust_input(input, input_dim, tdim)

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            u2 = ⊘(u[:, j-1], 2)
            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = ⊘(u[:, j-1], 2)
            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Fisher-KPP model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method (CNAB) with control.
"""
function integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix, quadratic_matrix, 
                                           control_matrix, const_stepsize=false, u2_jm1=nothing)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    F = quadratic_matrix
    B = control_matrix

    # Adjust the input
    input_dim = size(B, 2)  # Number of inputs
    input = adjust_input(input, input_dim, tdim)

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim 
            u2 = ⊘(u[:, j-1], 2)
            if j == 2 && isnothing(u2_jm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt + 0.5 * B * (input[:,j-1]+input[:,j]) * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2 + 0.5 * B * (input[:,j-1] + input[:,j]) * Δt)
            end
            u2_jm1 = u2
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = ⊘(u[:, j-1], 2)
            if j == 2 && isnothing(u2_jm1)
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * Δt + 0.5 * B * (input[:,j-1] + input[:,j]) * Δt)
            else
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2 + 
                                             0.5 * B * (input[:,j-1] + input[:,j]) * Δt)
            end
            u2_jm1 = u2
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Chafee-Infante model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method (CNAB)
without control.
"""
function integrate_model_without_control_CNAB(tdata, u0; linear_matrix, quadratic_matrix, 
                                              const_stepsize=false, u2_jm1=nothing)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    F = quadratic_matrix

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim 
            u2 = ⊘(u[:, j-1], 2)
            if j == 2 && isnothing(u2_jm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2)
            end
            u2_jm1 = u2
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = ⊘(u[:, j-1], 2)
            if j == 2 && isnothing(u2_jm1)
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2)
            end
            u2_jm1 = u2
        end
    end
    return u
end


function integrate_model(tdata::AbstractArray{T}, u0::AbstractArray{T}, input::AbstractArray{T}=T[]; kwargs...) where {T<:Real}
    # Check that keyword exists in kwargs
    @assert haskey(kwargs, :linear_matrix) "Keyword :linear_matrix not found"
    @assert haskey(kwargs, :quadratic_matrix) "Keyword :quadratic_matrix not found"
    if !isempty(input)
        @assert haskey(kwargs, :control_matrix) "Keyword :control_matrix not found"
    end

    # Unpack the keyword arguments
    linear_matrix = kwargs[:linear_matrix]
    quadratic_matrix = kwargs[:quadratic_matrix]
    control_matrix = haskey(kwargs, :control_matrix) ? kwargs[:control_matrix] : nothing
    system_input = haskey(kwargs, :system_input) ? kwargs[:system_input] : !isempty(input)
    const_stepsize = haskey(kwargs, :const_stepsize) ? kwargs[:const_stepsize] : false
    u2_jm1 = haskey(kwargs, :u2_jm1) ? kwargs[:u2_jm1] : nothing
    if haskey(kwargs, :integrator_type)
        integrator_type = kwargs[:integrator_type]
        @assert integrator_type ∈ (:SICN, :CNAB) "Invalid integrator type. Choose from (:SICN, :CNAB), where SICN is Semi-Implicit Crank-Nicolson and CNAB is Crank-Nicolson Adam-Bashforth"
    else
        integrator_type = :CNAB
    end

    if system_input
        if integrator_type == :SICN
            integrate_model_with_control_SICN(tdata, u0, input; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix, control_matrix=control_matrix)
        else
            integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix, 
                                              control_matrix=control_matrix, const_stepsize=const_stepsize, u2_jm1=u2_jm1)
        end
    else
        if integrator_type == :SICN
            integrate_model_without_control_SICN(tdata, u0; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix)
        else
            integrate_model_without_control_CNAB(tdata, u0; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix,
                                                 const_stepsize=const_stepsize, u2_jm1=u2_jm1)
        end
    end
end


end