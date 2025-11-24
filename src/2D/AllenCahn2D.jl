"""
    2D Allen-Cahn equation PDE Model
"""
module AllenCahn2D

using DocStringExtensions
using Kronecker: ⊗
using LinearAlgebra
using SparseArrays
using UniqueKronecker

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export AllenCahn2DModel


"""
$(TYPEDEF)

Allen-Cahn equation Model
    
```math
\\frac{\\partial u}{\\partial t} = \\mu\\left(\\frac{\\partial^2 u}{\\partial x^2} + \\frac{\\partial^2 u}{\\partial y^2}\\right) - \\epsilon(u^3 - u)
```

where ``u`` is the state variable, ``μ`` is the diffusion coefficient, ``ϵ`` is a nonlinear coefficient.

## Fields
- `spatial_domain::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}`: spatial domain (x, y)
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Tuple{Real,Real}`: parameter domain (diffusion coeff)
- `Δx::Real`: spatial grid size (x-axis)
- `Δy::Real`: spatial grid size (y-axis)
- `Δt::Real`: temporal step size
- `params::Dict{Symbol,Union{Real,AbstractArray{<:Real}}}`: parameters
- `xspan::Vector{<:Real}`: spatial grid points (x-axis)
- `yspan::Vector{<:Real}`: spatial grid points (y-axis)
- `tspan::Vector{<:Real}`: temporal points
- `spatial_dim::Int`: spatial dimension
- `time_dim::Int`: temporal dimension
- `param_dim::Int`: parameter dimension
- `IC::AbstractArray{<:Real}`: initial condition
- `BC::Tuple{Symbol,Symbol}`: boundary condition
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
"""
mutable struct AllenCahn2DModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}  # spatial domain (x, y)
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Dict{Symbol,Tuple{Real,Real}}

    # Discritization grid
    Δx::Real  # spatial grid size (x-axis)
    Δy::Real  # spatial grid size (y-axis)
    Δt::Real  # temporal step size

    # Grid points
    xspan::Vector{Float64}  # spatial grid points (x-axis)
    yspan::Vector{Float64}  # spatial grid points (y-axis)
    tspan::Vector{<:Real}  # temporal points
    params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}} # parameters

    # Dimensions
    spatial_dim::Tuple{Int64,Int64}  # spatial dimension x and y
    time_dim::Int  # temporal dimension
    param_dim::Dict{Symbol,<:Int}  # parameter dimension

    # Initial condition
    IC::AbstractArray{<:Real}  # initial condition

    # Boundary condition
    BC::Tuple{Symbol,Symbol}  # boundary condition

    # Functions
    finite_diff_model::Function  # model using Finite Difference
    integrate_model::Function # integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
end


"""
$(SIGNATURES)

Constructor for the Allen-Cahn equation model.
"""
function AllenCahn2DModel(;
    spatial_domain::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}},
    time_domain::Tuple{Real,Real}, 
    Δx::Real, Δy::Real, Δt::Real, 
    params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}}, 
    BC::Tuple{Symbol,Symbol})

    # Discritization grid info
    possible_BC = (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy, :flux)
    @assert all([BC[i] ∈ possible_BC for i in eachindex(BC)]) "Invalid boundary condition"

    # x-axis
    if BC[1] == :periodic
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2]-Δx)
    elseif BC[1] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2])
    end
    # y-axis
    if BC[2] == :periodic
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2]-Δy)
    elseif BC[2] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2])
    end

    tspan = collect(time_domain[1]:Δt:time_domain[2])
    spatial_dim = (length(xspan), length(yspan))
    time_dim = length(tspan)

    # Initial condition
    IC = zeros(prod(spatial_dim))

    # Parameter dimensions or number of parameters 
    param_dim = Dict([k => length(v) for (k, v) in params])
    param_domain = Dict([k => extrema(v) for (k,v) in params])

    AllenCahn2DModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δy, Δt, 
        xspan, yspan, tspan, params,
        spatial_dim, time_dim, param_dim, 
        IC, BC,
        finite_diff_model, integrate_model
    )
end


"""
$(SIGNATURES)

Create the matrices A (linear operator) and E (cubic operator) for the Allen-Cahn model.

## Arguments
- `model::AllenCahn2DModel`: Allen-Cahn model
- `params::Dict`: parameters dictionary
"""
function finite_diff_model(model::AllenCahn2DModel, params::Dict)
    if all(model.BC .== :periodic)
        return finite_diff_periodic_model(model.spatial_dim..., model.Δx, model.Δy, params)
    elseif all(model.BC .== :dirichlet)
        return finite_diff_dirichlet_model(model.spatial_dim..., model.Δx, model.Δy, params)
    else
        error("Boundary condition not implemented")
    end
end

"""
$(SIGNATURES)

Construct A and B matrices for 2D heat equation with periodic boundary conditions.
Returns (A, B) where B is empty (no boundary inputs for pure periodic BCs).
"""
function finite_diff_periodic_model(Nx::Integer, Ny::Integer, Δx::Real, Δy::Real, params::Dict)
    μ = params[:μ]
    ϵ = params[:ϵ]

    # 1D periodic second-derivative (circulant) operators
    Ax = spdiagm(
        0 => (ϵ - 2*μ/Δx^2) * ones(Nx),
        1 => μ/Δx^2 * ones(Nx-1), 
        -1 => μ/Δx^2 * ones(Nx-1)
    )
    Ax[1, Nx] = μ / Δx^2
    Ax[Nx, 1] = μ / Δx^2

    Ay = spdiagm(
        0 => (ϵ - 2*μ/Δy^2) * ones(Ny),
        1 => μ/Δy^2 * ones(Ny-1), 
        -1 => μ/Δy^2 * ones(Ny-1)
    )
    Ay[1, Ny] = μ / Δy^2
    Ay[Ny, 1] = μ / Δy^2

    # 2D Laplacian with periodic BC via Kronecker sums
    A = (Ay ⊗ I(Nx)) + (I(Ny) ⊗ Ax)

    # Construct cubic operator E for -ϵ * u.^3 term.
    # We use the same packed-monomial indexing as in the 1D implementation:
    Ntotal = Nx * Ny
    S = Int(Ntotal * (Ntotal + 1) * (Ntotal + 2) ÷ 6)  # number of unique cubic monomials
    cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:Ntotal-1])
    pushfirst!(cubic_idx_diff, 1)
    iii = collect(1:Ntotal)
    jjj = cumsum(cubic_idx_diff)  # column indices of the pure cubic (i,i,i) monomials
    vvv = -ϵ * ones(Ntotal)
    E = sparse(iii, jjj, vvv, Ntotal, S)

    return A, E
end


"""
$(SIGNATURES)
"""
function finite_diff_dirichlet_model(Nx::Real, Ny::Real, Δx::Real, Δy::Real, params::Dict)
    # Parameters 
    μ = params[:μ]
    ϵ = params[:ϵ]

    # Create A matrix
    Ax = spdiagm(
        0 => (ϵ - 2*μ/Δx^2) * ones(Nx),
        1 => μ/Δx^2 * ones(Nx-1), 
        -1 => μ/Δx^2 * ones(Nx-1)
    )
    Ay = spdiagm(
        0 => (ϵ - 2*μ/Δy^2) * ones(Ny),
        1 => μ/Δy^2 * ones(Ny-1), 
        -1 => μ/Δy^2 * ones(Ny-1)
    )
    A = (Ay ⊗ I(Nx)) + (I(Ny) ⊗ Ax)

    # Construct cubic operator E for -ϵ * u.^3 term.
    # We use the same packed-monomial indexing as in the 1D implementation:
    Ntotal = Nx * Ny
    S = Int(Ntotal * (Ntotal + 1) * (Ntotal + 2) ÷ 6)  # number of unique cubic monomials
    cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:Ntotal-1])
    pushfirst!(cubic_idx_diff, 1)
    iii = collect(1:Ntotal)
    jjj = cumsum(cubic_idx_diff)  # column indices of the pure cubic (i,i,i) monomials
    vvv = -ϵ * ones(Ntotal)
    E = sparse(iii, jjj, vvv, Ntotal, S)

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

    return A, E, B
end


"""
$(SIGNATURES)
    
Integrate the Allen-Cahn model using the Crank-Nicholson (linear) Explicit (nonlinear) method.
Or, in other words, Semi-Implicit Crank-Nicholson (SICN) method without control.
"""
function integrate_model_without_control_SICN(
    tdata, u0;
    linear_matrix, cubic_matrix, const_stepsize=false
    )

    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0 

    A = linear_matrix
    E = cubic_matrix

    # Cubic mapping function
    # cube = u -> prod(view(u, cidx), dims=1)[:]

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim

            u3 = ⊘(u[:, j-1], 3)
            # u3 = cube(u[:, j-1])

            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * Δt)
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]

            u3 = ⊘(u[:, j-1], 3)
            # u3 = cube(u[:, j-1])

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
function integrate_model_with_control_SICN(
    tdata, u0, input;
    linear_matrix, cubic_matrix, 
    control_matrix, const_stepsize=false
    )

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

    # Cubic mapping function
    # cube = u -> prod(view(u, cidx), dims=1)[:]

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim

            u3 = ⊘(u[:, j-1], 3)
            # u3 = cube(u[:, j-1])

            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + E * u3 * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]

            u3 = ⊘(u[:, j-1], 3)
            # u3 = cube(u[:, j-1])

            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + E * u3 * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Allen-Cahn model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method (CNAB) with control.
"""
function integrate_model_with_control_CNAB(
    tdata, u0, input;
    linear_matrix, cubic_matrix, 
    control_matrix, const_stepsize=false, u3_jm1=nothing
    )

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

    # Cubic mapping function
    # cube = u -> prod(view(u, cidx), dims=1)[:]

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim 

            u3 = ⊘(u[:, j-1], 3)
            # u3 = cube(u[:, j-1])

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
            # u3 = cube(u[:, j-1])

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
function integrate_model_without_control_CNAB(
    tdata, u0;
    linear_matrix, cubic_matrix, 
    const_stepsize=false, u3_jm1=nothing
    )

    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    E = cubic_matrix

    # Cubic mapping function
    # cube = u -> prod(view(u, cidx), dims=1)[:]

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim 

            u3 = ⊘(u[:, j-1], 3)
            # u3 = cube(u[:, j-1])

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
            # u3 = cube(u[:, j-1])

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

# """
# $(SIGNATURES)
# """
# function generate_cubic_indices(n::Integer)
#     S = div(n*(n+1)*(n+2), 6)
#     indices = Matrix{Int}(undef, 3, S)
    
#     idx = 1
#     @inbounds for i in 1:n
#         for j in i:n
#             len = n - j + 1
#             indices[1, idx:idx+len-1] .= i
#             indices[2, idx:idx+len-1] .= j
#             indices[3, idx:idx+len-1] .= j:n
#             idx += len
#         end
#     end
    
#     return indices
# end


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

    # cubic_indices = generate_cubic_indices(length(u0))

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
            integrate_model_with_control_SICN(
                tdata, u0, input; 
                # tdata, u0, input, cubic_indices; 
                linear_matrix=linear_matrix, cubic_matrix=cubic_matrix, 
                control_matrix=control_matrix)
        else
            integrate_model_with_control_CNAB(
                tdata, u0, input; 
                # tdata, u0, input, cubic_indices; 
                linear_matrix=linear_matrix, cubic_matrix=cubic_matrix, 
                control_matrix=control_matrix, 
                const_stepsize=const_stepsize, u3_jm1=u3_jm1)
        end
    else
        if integrator_type == :SICN
            integrate_model_without_control_SICN(
                tdata, u0; 
                # tdata, u0, cubic_indices; 
                linear_matrix=linear_matrix, cubic_matrix=cubic_matrix)
        else
            integrate_model_without_control_CNAB(
                tdata, u0; 
                # tdata, u0, cubic_indices; 
                linear_matrix=linear_matrix, cubic_matrix=cubic_matrix,
                const_stepsize=const_stepsize, u3_jm1=u3_jm1)
        end
    end
end

end
