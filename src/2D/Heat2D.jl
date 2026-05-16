"""
    2D Heat Equation Model
"""
module Heat2D

using DocStringExtensions
using Kronecker: ⊗
using LinearAlgebra
using SparseArrays

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input
using ..FastSolvers
import ..FastSolvers: build_fast_solver, integrate_model_fast,
                      linsolve!, mulIpA!, update_timestep!,
                      backward_euler_solve!,
                      FastKronSumSolver, FastFFT2DSolver, FastDenseSolver,
                      AbstractFastSolver

export Heat2DModel,
       FastDirichletSolver, FastPeriodicSolver, FastDenseSolver,
       AbstractFastBESolver,
       build_fast_be_solver, build_fast_solver,
       integrate_model_fast, update_timestep!,
       backward_euler_solve!,
       linsolve!, mulIpA!

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
    spatial_domain::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}
    time_domain::Tuple{Real,Real}
    param_domain::Tuple{Real,Real}

    # Grids
    Δx::Real
    Δy::Real
    Δt::Real

    # Dimensions
    spatial_dim::Tuple{Int64,Int64}
    time_dim::Int64
    param_dim::Int64

    # Boundary and Initial Conditions
    BC::Tuple{Symbol,Symbol}
    IC::Array{Float64}

    # Parameters
    diffusion_coeffs::Union{AbstractArray{<:Real},Real}

    # Data
    xspan::Vector{Float64}
    yspan::Vector{Float64}
    tspan::Vector{Float64}

    # Functions
    finite_diff_model::Function
    integrate_model::Function
end


function Heat2DModel(;spatial_domain::Tuple{Tuple{Real,Real},Tuple{Real,Real}}, time_domain::Tuple{Real,Real},
                 Δx::Real, Δy::Real, Δt::Real, diffusion_coeffs::Union{AbstractArray{<:Real},Real}, BC::Tuple{Symbol,Symbol})
    possible_BC = (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy, :flux)
    @assert all([BC[i] ∈ possible_BC for i in eachindex(BC)]) "Invalid boundary condition"
    if BC[1] == :periodic
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2]-Δx)
    elseif BC[1] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy)
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2])
    end
    if BC[2] == :periodic
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2]-Δy)
    elseif BC[2] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy)
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2])
    end
    tspan = collect(time_domain[1]:Δt:time_domain[2])
    spatial_dim = (length(xspan), length(yspan))
    time_dim = length(tspan)

    IC = zeros(prod(spatial_dim))

    param_dim = length(diffusion_coeffs)
    param_domain = extrema(diffusion_coeffs)

    Heat2DModel(spatial_domain, time_domain, param_domain, Δx, Δy, Δt,
           spatial_dim, time_dim, param_dim, BC, IC, diffusion_coeffs,
           xspan, yspan, tspan,
           finite_diff_model, integrate_model)
end


function finite_diff_dirichlet_model(Nx::Integer, Ny::Integer, Δx::Real, Δy::Real, μ::Real)
    Ax = spdiagm(0 => (-2)*ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1)) * μ / Δx^2
    Ay = spdiagm(0 => (-2)*ones(Ny), 1 => ones(Ny-1), -1 => ones(Ny-1)) * μ / Δy^2
    A = (Ay ⊗ I(Nx)) + (I(Ny) ⊗ Ax)

    Bx = spzeros(Nx*Ny, 2)
    left_indices  = [ (j - 1) * Nx + 1  for j in 1:Ny ]
    right_indices = [ (j - 1) * Nx + Nx for j in 1:Ny ]
    Bx[left_indices,  1] .= μ / Δx^2
    Bx[right_indices, 2] .= μ / Δx^2

    By = spzeros(Nx*Ny, 2)
    bottom_indices = [ i for i in 1:Nx ]
    top_indices    = [ (Ny - 1) * Nx + i for i in 1:Nx ]
    By[bottom_indices, 1] .= μ / Δy^2
    By[top_indices,    2] .= μ / Δy^2

    B = hcat(Bx, By)
    return A, B
end

function finite_diff_periodic_model(Nx::Integer, Ny::Integer, Δx::Real, Δy::Real, μ::Real)
    Ax = spdiagm(0 => (-2)*ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1))
    Ax = sparse(Ax); Ax[1, Nx] = 1; Ax[Nx, 1] = 1
    Ax *= μ / Δx^2

    Ay = spdiagm(0 => (-2)*ones(Ny), 1 => ones(Ny-1), -1 => ones(Ny-1))
    Ay = sparse(Ay); Ay[1, Ny] = 1; Ay[Ny, 1] = 1
    Ay *= μ / Δy^2

    A = (Ay ⊗ I(Nx)) + (I(Ny) ⊗ Ax)
    return A
end

function finite_diff_model(model::Heat2DModel, μ::Real)
    if all(model.BC .== :dirichlet)
        return finite_diff_dirichlet_model(model.spatial_dim..., model.Δx, model.Δy, μ)
    elseif all(model.BC .== :periodic)
        return finite_diff_periodic_model(model.spatial_dim..., model.Δx, model.Δy, μ)
    else
        error("Not implemented")
    end
end


# ============================================================================
# Original integrate_model functions (kept for backward compatibility)
# ============================================================================

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


function integrate_model(tdata::AbstractVector{T}, u0::AbstractVector{T},
                         input::AbstractArray{T}=T[]; kwargs...) where {T<:Real}
    @assert haskey(kwargs, :linear_matrix) "Keyword :linear_matrix not found"
    @assert haskey(kwargs, :system_input) "Keyword :system_input not found"
    if kwargs[:system_input]
        @assert haskey(kwargs, :control_matrix) "Keyword :control_matrix not found"
    end
    @assert haskey(kwargs, :integrator_type) "Keyword :integrator_type not found"

    system_input    = kwargs[:system_input]
    integrator_type = kwargs[:integrator_type]

    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:,1] = u0

    if system_input
        A = kwargs[:linear_matrix]
        B = kwargs[:control_matrix]
        input_dim = size(B, 2)
        input = adjust_input(input, input_dim, tdim)
    else
        A = kwargs[:linear_matrix]
    end

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


# ============================================================================
# Fast solvers
# ----------------------------------------------------------------------------
# The structured solver types and the per-step kernels (linsolve!, mulIpA!,
# update_timestep!, …) live in the shared `FastSolvers` submodule. This
# section provides Heat2D-specific aliases and convenience constructors so the
# original Heat2D API (`FastDirichletSolver`, `FastPeriodicSolver`,
# `build_fast_be_solver`, `integrate_model_fast`) keeps working unchanged.
#
# The 2D heat operator has Kronecker-sum structure
#       A = (Ay ⊗ I_Nx) + (I_Ny ⊗ Ax)
# so its 1D factors are simultaneously diagonalized by:
#   * eigendecomposition (Dirichlet) → `FastKronSumSolver`
#   * 2D FFT             (periodic)  → `FastFFT2DSolver`
# ============================================================================

"""
    AbstractFastBESolver

Legacy alias for `FastSolvers.AbstractFastSolver`. Retained so that downstream
code with `solver isa AbstractFastBESolver` checks continues to work.
"""
const AbstractFastBESolver = AbstractFastSolver

"""
    FastDirichletSolver

Type alias for the structured Heat2D Dirichlet solver, which is a
`FastKronSumSolver{Float64}`. Construct it via either

```julia
FastDirichletSolver(Nx, Ny, Δx, Δy, μ, Δt)
```

or via the model-aware dispatcher `build_fast_be_solver(model, μ)`.
"""
const FastDirichletSolver = FastKronSumSolver{Float64}

"""
    FastPeriodicSolver

Type alias for the FFT-based 2D periodic Heat2D solver.
"""
const FastPeriodicSolver = FastFFT2DSolver


"""
$(SIGNATURES)

Build the fast Backward Euler solver for `model`. This is the original
Heat2D API; equivalent to `build_fast_solver(model, μ; scheme=:BE, Δt=Δt)`.
"""
function build_fast_be_solver(model::Heat2DModel, μ::Real, Δt::Real=model.Δt)
    return build_fast_solver(model, μ; scheme=:BE, Δt=Δt)
end


"""
$(SIGNATURES)

Build a fast implicit solver for a 2-D heat model. `scheme=:BE` bakes in
`α = Δt` (backward Euler); `scheme=:CN` bakes in `α = Δt/2` (Crank–Nicolson).

Currently supports `(:dirichlet, :dirichlet)` and `(:periodic, :periodic)`
boundary conditions.
"""
function build_fast_solver(model::Heat2DModel, μ::Real;
                            scheme::Symbol=:BE, Δt::Real=model.Δt)
    α = scheme === :BE ? Float64(Δt) :
        scheme === :CN ? Float64(Δt)/2 :
        error("scheme must be :BE or :CN, got $scheme")
    Nx, Ny = model.spatial_dim
    if all(model.BC .== :dirichlet)
        return FastKronSumSolver(Nx, Ny, model.Δx, model.Δy, μ, α)
    elseif all(model.BC .== :periodic)
        return FastFFT2DSolver(Nx, Ny, model.Δx, model.Δy, μ, α)
    else
        error("Fast solver not implemented for BC = $(model.BC). " *
              "Currently supports (:dirichlet, :dirichlet) and (:periodic, :periodic).")
    end
end


"""
$(SIGNATURES)

Fast backward Euler integrator. Same signature as the original
`integrate_model(A, B, U, tdata, IC)` except it takes a precomputed
`solver::AbstractFastSolver` in place of the assembled matrix `A`.
Pass `B` and `U` as empty matrices (or skip the entries) when there are no
boundary inputs (e.g. periodic BCs).

Assumes a uniform time step (`tdata[i] - tdata[i-1]` constant) matching the
Δt used when the solver was built.
"""
function integrate_model_fast(solver::AbstractFastSolver,
                              B::AbstractMatrix, U::AbstractMatrix,
                              tdata::AbstractVector, IC::AbstractVector)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = Matrix{Float64}(undef, Xdim, Tdim)
    state[:, 1] .= IC

    rhs = Vector{Float64}(undef, Xdim)
    has_input = size(B, 1) > 0 && size(B, 2) > 0 && !isempty(U)
    Δt = tdata[2] - tdata[1]

    if has_input
        Bu = Vector{Float64}(undef, Xdim)
        @inbounds for j in 2:Tdim
            mul!(Bu, B, view(U, :, j-1))
            @. rhs = state[:, j-1] + Δt * Bu
            linsolve!(view(state, :, j), solver, rhs)
        end
    else
        @inbounds for j in 2:Tdim
            @. rhs = state[:, j-1]
            linsolve!(view(state, :, j), solver, rhs)
        end
    end
    return state
end

# Convenience overload: build solver and integrate in one call.
function integrate_model_fast(model::Heat2DModel, μ::Real,
                              B::AbstractMatrix, U::AbstractMatrix,
                              tdata::AbstractVector, IC::AbstractVector)
    Δt = tdata[2] - tdata[1]
    solver = build_fast_solver(model, μ; scheme=:BE, Δt=Δt)
    return integrate_model_fast(solver, B, U, tdata, IC)
end

# Convenience overload for the periodic / no-input case.
function integrate_model_fast(model::Heat2DModel, μ::Real,
                              tdata::AbstractVector, IC::AbstractVector)
    Δt = tdata[2] - tdata[1]
    solver = build_fast_solver(model, μ; scheme=:BE, Δt=Δt)
    Xdim = length(IC)
    return integrate_model_fast(solver,
                                zeros(Xdim, 0), zeros(0, length(tdata)),
                                tdata, IC)
end


# ============================================================================
# Reduced-order / dense use case: FastDenseSolver pass-through
# ============================================================================

"""
$(SIGNATURES)

Integrate a reduced-order system `du/dt = A u + B f` using backward Euler
with a precomputed `FastDenseSolver`.
"""
function integrate_model_fast(solver::FastDenseSolver,
                              tdata::AbstractVector, u0::AbstractVector,
                              B::AbstractMatrix, input::AbstractMatrix)
    r = solver.r
    Tdim = length(tdata)
    u = Matrix{Float64}(undef, r, Tdim)
    u[:, 1] .= u0

    Δt = tdata[2] - tdata[1]
    has_input = size(B, 2) > 0 && !isempty(input)

    rhs = Vector{Float64}(undef, r)

    if has_input
        Bu = Vector{Float64}(undef, r)
        @inbounds for j in 2:Tdim
            mul!(Bu, B, view(input, :, j-1))
            @. rhs = u[:, j-1] + Δt * Bu
            linsolve!(view(u, :, j), solver, rhs)
        end
    else
        @inbounds for j in 2:Tdim
            @. rhs = u[:, j-1]
            linsolve!(view(u, :, j), solver, rhs)
        end
    end
    return u
end

# Convenience: no-input overload
function integrate_model_fast(solver::FastDenseSolver,
                              tdata::AbstractVector, u0::AbstractVector)
    return integrate_model_fast(solver, tdata, u0,
                                zeros(solver.r, 0), zeros(0, length(tdata)))
end

end
