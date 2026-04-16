"""
    2D Heat Equation Model
"""
module Heat2D

using DocStringExtensions
using FFTW
using Kronecker: ⊗
using LinearAlgebra
using SparseArrays

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input

export Heat2DModel,
       FastDirichletSolver, FastPeriodicSolver, FastDenseSolver,
       build_fast_be_solver, integrate_model_fast, update_timestep!

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
# Fast backward Euler solvers
# ----------------------------------------------------------------------------
# The 2D heat operator has Kronecker-sum structure
#       A = (Ay ⊗ I_Nx) + (I_Ny ⊗ Ax)
# which simultaneously diagonalizes Ax and Ay. For backward Euler we need to
# solve (I - Δt*A) u_new = rhs at every step. Reshaping rhs into an Nx × Ny
# matrix R, the system decouples in the eigenbasis (or Fourier basis):
#
#       û_{ij} = r̂_{ij} / (1 - Δt*(λx_i + λy_j))
#
# For Dirichlet BCs we eigendecompose the 1D symmetric tridiagonal Ax, Ay
# once and apply the change of basis as four dense matmuls per step
# (BLAS-3, very cache friendly):
#
# U_new = Vx * ( (Vxᵀ * RHS_mat * Vy) ./ (1 .- Δt .* (λx .+ λy')) ) * Vyᵀ
#
# For periodic BCs both Ax, Ay are circulant, diagonalized by the DFT, so we
# use a 2D in-place FFT plus an elementwise divide.
#
# In both cases there is no factorization or sparse triangular solve at all,
# and the per-step cost drops dramatically:
#   - sparse LU + back-solve : O(N^{3/2}) factor + O(N log N) per step
#   - fast diagonalization   : O(N_x N_y (N_x + N_y)) per step, BLAS-3
#   - FFT (periodic)         : O(N log N) per step
# ============================================================================

abstract type AbstractFastBESolver end

"""
$(TYPEDEF)

Fast backward Euler solver for the 2D heat equation with Dirichlet BCs,
using fast diagonalization of the Kronecker-sum operator.

Built once for a given (Nx, Ny, Δx, Δy, μ, Δt). All allocations happen at
construction; subsequent solves are allocation-free.
"""
struct FastDirichletSolver <: AbstractFastBESolver
    Vx::Matrix{Float64}
    Vy::Matrix{Float64}
    Vxt::Matrix{Float64}             # Vx' materialized for BLAS efficiency
    Vyt::Matrix{Float64}
    inv_denom::Matrix{Float64}       # 1 ./ (1 .- Δt .* (λx .+ λy'))
    Nx::Int
    Ny::Int
    tmp1::Matrix{Float64}            # workspace
    tmp2::Matrix{Float64}
end

function FastDirichletSolver(Nx::Integer, Ny::Integer, Δx::Real, Δy::Real,
                              μ::Real, Δt::Real)
    Ax = SymTridiagonal(fill(-2μ/Δx^2, Nx), fill(μ/Δx^2, Nx-1))
    Ay = SymTridiagonal(fill(-2μ/Δy^2, Ny), fill(μ/Δy^2, Ny-1))
    Ex = eigen(Ax)
    Ey = eigen(Ay)
    inv_denom = 1.0 ./ (1.0 .- Δt .* (Ex.values .+ Ey.values'))
    return FastDirichletSolver(
        Ex.vectors, Ey.vectors,
        Matrix(Ex.vectors'), Matrix(Ey.vectors'),
        inv_denom, Int(Nx), Int(Ny),
        zeros(Nx, Ny), zeros(Nx, Ny),
    )
end

"""
$(SIGNATURES)

In-place backward Euler step: solves `(I - Δt*A) * unew = rhs` and writes
the result into `unew`. The Δt baked into `F` must match the time step used
to construct it.
"""
function backward_euler_solve!(unew::AbstractVector, F::FastDirichletSolver,
                                rhs::AbstractVector)
    R = reshape(rhs,  F.Nx, F.Ny)
    U = reshape(unew, F.Nx, F.Ny)
    mul!(F.tmp1, F.Vxt, R)              # tmp1 = Vxᵀ R
    mul!(F.tmp2, F.tmp1, F.Vy)          # tmp2 = Vxᵀ R Vy   (= R̂)
    @inbounds @. F.tmp2 = F.tmp2 * F.inv_denom
    mul!(F.tmp1, F.Vx, F.tmp2)          # tmp1 = Vx Û
    mul!(U,       F.tmp1, F.Vyt)        # U    = Vx Û Vyᵀ
    return unew
end


"""
$(TYPEDEF)

Fast backward Euler solver for the 2D heat equation with periodic BCs,
using FFT diagonalization of the circulant Laplacians.
"""
struct FastPeriodicSolver{P,IP} <: AbstractFastBESolver
    plan_f::P
    plan_if::IP
    inv_denom::Matrix{Float64}
    Nx::Int
    Ny::Int
    buffer::Matrix{ComplexF64}
end

function FastPeriodicSolver(Nx::Integer, Ny::Integer, Δx::Real, Δy::Real,
                             μ::Real, Δt::Real)
    # Eigenvalues of the periodic 1D second-difference operator
    λx = [μ/Δx^2 * (2cos(2π*(k-1)/Nx) - 2) for k in 1:Nx]
    λy = [μ/Δy^2 * (2cos(2π*(k-1)/Ny) - 2) for k in 1:Ny]
    inv_denom = 1.0 ./ (1.0 .- Δt .* (λx .+ λy'))

    buffer  = zeros(ComplexF64, Nx, Ny)
    plan_f  = plan_fft!(buffer;  flags=FFTW.MEASURE)
    plan_if = plan_ifft!(buffer; flags=FFTW.MEASURE)
    return FastPeriodicSolver(plan_f, plan_if, inv_denom, Int(Nx), Int(Ny), buffer)
end

function backward_euler_solve!(unew::AbstractVector, F::FastPeriodicSolver,
                                rhs::AbstractVector)
    @inbounds for k in eachindex(rhs)
        F.buffer[k] = rhs[k]
    end
    F.plan_f  * F.buffer                                # in-place forward FFT
    @inbounds @. F.buffer = F.buffer * F.inv_denom
    F.plan_if * F.buffer                                # in-place inverse FFT
    @inbounds for k in eachindex(unew)
        unew[k] = real(F.buffer[k])
    end
    return unew
end


"""
$(SIGNATURES)

Build the appropriate fast backward Euler solver for `model`. Currently
supports `(:dirichlet, :dirichlet)` and `(:periodic, :periodic)` BCs.
"""
function build_fast_be_solver(model::Heat2DModel, μ::Real, Δt::Real=model.Δt)
    Nx, Ny = model.spatial_dim
    if all(model.BC .== :dirichlet)
        return FastDirichletSolver(Nx, Ny, model.Δx, model.Δy, μ, Δt)
    elseif all(model.BC .== :periodic)
        return FastPeriodicSolver(Nx, Ny, model.Δx, model.Δy, μ, Δt)
    else
        error("Fast backward Euler solver not implemented for BC = $(model.BC). " *
              "Currently supports (:dirichlet, :dirichlet) and (:periodic, :periodic).")
    end
end


"""
$(SIGNATURES)

Fast backward Euler integrator. Same signature as the original
`integrate_model(A, B, U, tdata, IC)` except it takes a precomputed
`solver::AbstractFastBESolver` in place of the assembled matrix `A`.
Pass `B` and `U` as empty matrices (or skip the entries) when there are no
boundary inputs (e.g. periodic BCs).

Assumes a uniform time step (`tdata[i] - tdata[i-1]` constant) matching the
Δt used when the solver was built.
"""
function integrate_model_fast(solver::AbstractFastBESolver,
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
            backward_euler_solve!(view(state, :, j), solver, rhs)
        end
    else
        @inbounds for j in 2:Tdim
            @. rhs = state[:, j-1]
            backward_euler_solve!(view(state, :, j), solver, rhs)
        end
    end
    return state
end

# Convenience overload: build solver and integrate in one call.
function integrate_model_fast(model::Heat2DModel, μ::Real,
                              B::AbstractMatrix, U::AbstractMatrix,
                              tdata::AbstractVector, IC::AbstractVector)
    Δt = tdata[2] - tdata[1]
    solver = build_fast_be_solver(model, μ, Δt)
    return integrate_model_fast(solver, B, U, tdata, IC)
end

# Convenience overload for the periodic / no-input case.
function integrate_model_fast(model::Heat2DModel, μ::Real,
                              tdata::AbstractVector, IC::AbstractVector)
    Δt = tdata[2] - tdata[1]
    solver = build_fast_be_solver(model, μ, Δt)
    Xdim = length(IC)
    return integrate_model_fast(solver,
                                zeros(Xdim, 0), zeros(0, length(tdata)),
                                tdata, IC)
end


# ============================================================================
# Fast backward Euler for unstructured dense A (e.g. from a reduced-order model)
# ----------------------------------------------------------------------------
# Given a dense r×r matrix A (no Kronecker or sparsity structure), we
# eigendecompose once:
#       A = V Λ V⁻¹
# Then:
#       (I - Δt A)⁻¹ = V diag(1 / (1 - Δt λ_i)) V⁻¹
#
# We precompute M_inv = real(V D V⁻¹) as a single dense r×r matrix so that
# each backward Euler step is a single BLAS-2 mul!(unew, M_inv, rhs).
#
# If Δt changes (e.g. adaptive stepping or parameter sweep), call
# update_timestep!(solver, Δt_new) to rebuild M_inv in O(r²) without
# repeating the O(r³) eigendecomposition.
#
# A may be non-symmetric; complex eigenvalues are handled transparently.
# If A is nearly defective (κ(V) ≫ 1), the constructor issues a warning and
# falls back to a direct LU-based inverse for robustness.
# ============================================================================
 
"""
$(TYPEDEF)
 
Fast backward Euler solver for a dense, unstructured matrix `A` (typically
from a reduced-order model). Precomputes `(I - Δt A)⁻¹` via eigendecomposition
so that each time step is a single dense matrix-vector multiply.
 
## Fields
$(TYPEDFIELDS)
"""
struct FastDenseSolver <: AbstractFastBESolver
    "Precomputed (I - Δt A)⁻¹, real r×r matrix applied via mul! each step"
    M_inv::Matrix{Float64}
    "Eigenvectors of A (complex, stored for update_timestep!)"
    V::Matrix{ComplexF64}
    "Inverse of V (complex)"
    Vinv::Matrix{ComplexF64}
    "Eigenvalues of A (complex)"
    λ::Vector{ComplexF64}
    "Dimension of the system"
    r::Int
    "Whether the solver was constructed via eigendecomposition (false = LU fallback)"
    eigen_based::Bool
end
 
 
"""
$(SIGNATURES)
 
Construct a fast backward Euler solver for a dense matrix `A`.
 
Eigendecomposes `A` once and precomputes the full inverse
`M_inv = real(V diag(1/(1 - Δt λ_i)) V⁻¹)`. If `A` is nearly defective
(condition number of `V` exceeds `cond_threshold`), falls back to a direct
`inv(I - Δt * A)` and prints a warning.
 
## Arguments
- `A::AbstractMatrix{<:Real}`: the system matrix (r × r)
- `Δt::Real`: time step size
 
## Keyword Arguments
- `cond_threshold::Real=1e12`: condition number threshold for V; above this,
  fall back to direct inverse
"""
function FastDenseSolver(A::AbstractMatrix{<:Real}, Δt::Real;
                          cond_threshold::Real=1e12)
    r = size(A, 1)
    @assert size(A, 2) == r "A must be square, got size $(size(A))"
 
    F = eigen(A)
    V    = ComplexF64.(F.vectors)
    Vinv = inv(V)
    λ    = ComplexF64.(F.values)
 
    κ = opnorm(V, 2) * opnorm(Vinv, 2)  # cond(V)
 
    if κ > cond_threshold
        @warn "Eigenvector matrix is ill-conditioned (κ(V) = $(round(κ; sigdigits=3))). " *
              "Falling back to direct inverse of (I - Δt A) for robustness."
        M_inv = real.(inv(I - Δt * A))
        return FastDenseSolver(M_inv, V, Vinv, λ, r, false)
    end
 
    M_inv = _build_M_inv(V, Vinv, λ, Δt)
    return FastDenseSolver(M_inv, V, Vinv, λ, r, true)
end
 
# Internal: compute real(V * Diag(d) * Vinv) with sanity check.
function _build_M_inv(V::Matrix{ComplexF64}, Vinv::Matrix{ComplexF64},
                       λ::Vector{ComplexF64}, Δt::Real)
    d = 1.0 ./ (1.0 .- Δt .* λ)
    M_inv_c = V * Diagonal(d) * Vinv
    imag_norm = norm(imag.(M_inv_c))
    real_norm = max(norm(real.(M_inv_c)), 1.0)
    if imag_norm / real_norm > 1e-10
        @warn "Precomputed inverse has unexpectedly large imaginary part " *
              "(relative: $(round(imag_norm/real_norm; sigdigits=3))). " *
              "Proceeding with real part only."
    end
    return real.(M_inv_c)
end
 
 
"""
$(SIGNATURES)
 
In-place backward Euler step for a dense unstructured system. Applies the
precomputed `M_inv` as a single matrix-vector multiply.
"""
function backward_euler_solve!(unew::AbstractVector, F::FastDenseSolver,
                                rhs::AbstractVector)
    mul!(unew, F.M_inv, rhs)
    return unew
end
 
 
"""
$(SIGNATURES)
 
Rebuild `M_inv` for a new time step `Δt_new` without repeating the
eigendecomposition of `A`. Cost: O(r²).
 
If the solver was constructed via the LU fallback (nearly defective `A`),
this recomputes `inv(I - Δt_new A)` from the stored eigendecomposition
anyway, which may be inaccurate; a warning is issued.
"""
function update_timestep!(solver::FastDenseSolver, Δt_new::Real)
    if !solver.eigen_based
        @warn "Solver was built via LU fallback due to ill-conditioned " *
              "eigenvectors. update_timestep! uses the eigendecomposition " *
              "regardless; results may be inaccurate."
    end
    solver.M_inv .= _build_M_inv(solver.V, solver.Vinv, solver.λ, Δt_new)
    return solver
end
 
 
"""
$(SIGNATURES)
 
Integrate a reduced-order system `du/dt = A u + B f` using backward Euler
with a precomputed `FastDenseSolver`.
 
## Arguments
- `solver::FastDenseSolver`: precomputed solver (from `FastDenseSolver(A, Δt)`)
- `tdata::AbstractVector`: time points (uniform spacing must match solver Δt)
- `u0::AbstractVector`: initial condition (length r)
- `B::AbstractMatrix`: input matrix (r × m); pass `zeros(r,0)` if no input
- `input::AbstractMatrix`: input signals (m × Tdim); pass `zeros(0,Tdim)` if no input
 
## Returns
- `u::Matrix{Float64}`: state trajectory (r × Tdim)
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
            backward_euler_solve!(view(u, :, j), solver, rhs)
        end
    else
        @inbounds for j in 2:Tdim
            @. rhs = u[:, j-1]
            backward_euler_solve!(view(u, :, j), solver, rhs)
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