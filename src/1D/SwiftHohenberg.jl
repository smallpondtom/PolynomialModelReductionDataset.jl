"""
    Subcritical Swift-Hohenberg (SH23) equation PDE Model
"""
module SwiftHohenberg

using DocStringExtensions
using LinearAlgebra
using SparseArrays
using UniqueKronecker

import ..PolynomialModelReductionDataset: AbstractModel, adjust_input
using ..FastSolvers
import ..FastSolvers: build_fast_solver, integrate_model_fast,
                      linsolve!, mulIpA!,
                      FastCirculant1DSolver, FastDenseSolver,
                      FactorizedSolver, AbstractFastSolver

export SwiftHohenbergModel, build_fast_solver, integrate_model_fast


"""
$(TYPEDEF)

Subcritical (quadratic-cubic) Swift-Hohenberg equation model, commonly called SH23:

```math
\\frac{\\partial u}{\\partial t} = r u - \\left(1 + \\frac{\\partial^2}{\\partial x^2}\\right)^2 u + b u^2 - u^3
```

where ``u`` is the state variable, ``r`` is the bifurcation (control) parameter, and
``b`` is the quadratic coefficient. Expanding the linear operator,

```math
\\frac{\\partial u}{\\partial t} = (r-1) u - 2\\frac{\\partial^2 u}{\\partial x^2} - \\frac{\\partial^4 u}{\\partial x^4} + b u^2 - u^3,
```

so the semi-discrete model is linear-quadratic-cubic in the state. The linear growth
rate of a Fourier mode ``e^{ikx}`` is ``r - (1-k^2)^2``, so the trivial state is
linearly stable for ``r<0``. For ``b^2 > 27/38`` the pattern-forming bifurcation at
``r=0`` is subcritical: a finite-amplitude periodic pattern coexists with the stable
trivial state for ``r_{\\mathrm{sn}} < r < 0``, and the basin of ``u \\equiv 0`` is
bounded by the stable manifolds of the unstable (lower-branch) states. For ``b=1.8``
the saddle-node of the ``k=1`` periodic branch is at ``r_{\\mathrm{sn}} \\approx -0.37``.

## Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Dict{Symbol,Tuple{Real,Real}}`: parameter domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `xspan::Vector{<:Real}`: spatial grid points
- `tspan::Vector{<:Real}`: temporal points
- `params::Dict{Symbol,Union{Real,AbstractArray{<:Real}}}`: parameters (`:r` bifurcation parameter, `:b` quadratic coefficient)
- `spatial_dim::Int`: spatial dimension
- `time_dim::Int`: temporal dimension
- `param_dim::Dict{Symbol,Int}`: parameter dimension
- `IC::AbstractArray{<:Real}`: initial condition
- `BC::Symbol`: boundary condition (`:periodic` or `:dirichlet`)
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: integrator using Crank-Nicolson (linear) Explicit (nonlinear) method
"""
mutable struct SwiftHohenbergModel <: AbstractModel
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
    integrate_model::Function # integrator using Crank-Nicolson (linear) Explicit (nonlinear) method
end


"""
$(SIGNATURES)

Constructor for the subcritical Swift-Hohenberg (SH23) equation model.

## Keyword Arguments
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `params::Dict`: parameters with keys `:r` (bifurcation parameter) and `:b` (quadratic coefficient)
- `BC::Symbol=:periodic`: boundary condition (`:periodic` or `:dirichlet`)

## Note
- `:dirichlet` imposes the hinged (simply supported) conditions ``u = u_{xx} = 0``
  at the boundaries. Inhomogeneous boundary values of ``u`` enter through the input
  matrix `B` returned by `finite_diff_model`.
"""
function SwiftHohenbergModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real,
                             params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}}, BC::Symbol=:periodic)
    # Discritization grid info
    @assert BC ∈ (:periodic, :dirichlet) "Invalid boundary condition. Choose from (:periodic, :dirichlet)"
    @assert haskey(params, :r) && haskey(params, :b) "params must contain :r (bifurcation parameter) and :b (quadratic coefficient)"
    if BC == :periodic
        xspan = collect(spatial_domain[1]:Δx:spatial_domain[2]-Δx)
    else
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

    SwiftHohenbergModel(
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

Create the matrices A (linear operator), F (quadratic operator), E (cubic
operator), and, for Dirichlet boundary conditions, B (input operator) for the
Swift-Hohenberg model.

## Arguments
- `model::SwiftHohenbergModel`: Swift-Hohenberg model
- `params::Dict`: parameters dictionary

## Returns
- periodic: `A, F, E`
- Dirichlet: `A, F, E, B`
"""
function finite_diff_model(model::SwiftHohenbergModel, params::Dict)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, params)
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, params)
    else
        error("Boundary condition not implemented")
    end
end


"""
$(SIGNATURES)

Quadratic operator F (N × N(N+1)/2) and cubic operator E (N × N(N+1)(N+2)/6)
of the reaction term ``b u^2 - u^3`` in the non-redundant (unique Kronecker)
coordinates. Only the ``u_i^2`` and ``u_i^3`` columns are nonzero.
"""
function reaction_operators(N::Int, b::Real)
    # Quadratic: b u_i^2
    S2 = Int(N * (N + 1) / 2)
    ii = 1:N  # row index
    jj = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 1:N])  # col index where the u_i^2 term is
    vv = b * ones(N)
    F = sparse(ii, jj, vv, N, S2)

    # Cubic: - u_i^3
    S3 = Int(N * (N + 1) * (N + 2) / 6)
    cubic_idx_diff = reverse([Int(3 + 2*(i-1) + 0.5*i*(i-1)) for i in 1:N-1])
    pushfirst!(cubic_idx_diff, 1)
    iii = collect(1:N)
    jjj = cumsum(cubic_idx_diff)  # col index where the u_i^3 term is
    vvv = -ones(N)
    E = sparse(iii, jjj, vvv, N, S3)

    return F, E
end


"""
$(SIGNATURES)

Stencil coefficients of the linear operator ``(r-1) - 2\\partial_x^2 - \\partial_x^4``
discretized with second-order central differences. Returns the diagonal, first
off-diagonal, and second off-diagonal values.
"""
function linear_stencil(Δx::Real, r::Real)
    d0 = (r - 1) + 4/Δx^2 - 6/Δx^4
    d1 = -2/Δx^2 + 4/Δx^4
    d2 = -1/Δx^4
    return d0, d1, d2
end


"""
$(SIGNATURES)

Create the matrices A (linear operator), F (quadratic operator), and E (cubic
operator) for the Swift-Hohenberg model with periodic boundary condition. The
linear operator is a symmetric pentadiagonal circulant matrix.

## Arguments
- `N::Real`: spatial dimension
- `Δx::Real`: spatial grid size
- `params::Dict`: parameters

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `F::SparseMatrixCSC{Float64,Int}`: quadratic operator
- `E::SparseMatrixCSC{Float64,Int}`: cubic operator
"""
function finite_diff_periodic_model(N::Real, Δx::Real, params::Dict)
    r = params[:r]
    b = params[:b]
    d0, d1, d2 = linear_stencil(Δx, r)

    # Create A matrix
    A = spdiagm(
        0 => d0 * ones(N),
        1 => d1 * ones(N - 1), -1 => d1 * ones(N - 1),
        2 => d2 * ones(N - 2), -2 => d2 * ones(N - 2)
    )
    # Periodicity for the first and final few indices
    A[1, end-1:end] = [d2, d1]
    A[2, end] = d2
    A[end-1, 1] = d2
    A[end, 1:2] = [d1, d2]

    F, E = reaction_operators(Int(N), b)
    return A, F, E
end


"""
$(SIGNATURES)

Create the matrices A (linear operator), F (quadratic operator), E (cubic
operator), and B (input operator) for the Swift-Hohenberg model with hinged
Dirichlet boundary conditions ``u = g`` and ``u_{xx} = 0`` at both ends.

The boundary values ``g_L, g_R`` sit at the ghost nodes one grid spacing
outside `xspan` (the same convention as the other Dirichlet models in this
package), and the condition ``u_{xx}=0`` determines the second ghost node by
reflection, ``u_{-1} = 2 g_L - u_1``. For homogeneous boundary values the
operator A is symmetric and diagonalized by the discrete sine transform.

## Arguments
- `N::Real`: spatial dimension
- `Δx::Real`: spatial grid size
- `params::Dict`: parameters

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `F::SparseMatrixCSC{Float64,Int}`: quadratic operator
- `E::SparseMatrixCSC{Float64,Int}`: cubic operator
- `B::SparseMatrixCSC{Float64,Int}`: input operator (columns: left and right boundary values)
"""
function finite_diff_dirichlet_model(N::Real, Δx::Real, params::Dict)
    r = params[:r]
    b = params[:b]
    d0, d1, d2 = linear_stencil(Δx, r)

    # Create A matrix
    A = spdiagm(
        0 => d0 * ones(N),
        1 => d1 * ones(N - 1), -1 => d1 * ones(N - 1),
        2 => d2 * ones(N - 2), -2 => d2 * ones(N - 2)
    )
    # Reflection of the second ghost node (u_xx = 0) changes the first and last diagonal entries
    A[1, 1] = (r - 1) + 4/Δx^2 - 5/Δx^4
    A[end, end] = (r - 1) + 4/Δx^2 - 5/Δx^4

    F, E = reaction_operators(Int(N), b)

    # Create B matrix
    B = spzeros(N, 2)
    B[1, 1] = -2/Δx^2 + 2/Δx^4  # left boundary value in the first row
    B[2, 1] = -1/Δx^4           # left boundary value in the second row
    B[end, 2] = -2/Δx^2 + 2/Δx^4  # right boundary value in the last row
    B[end-1, 2] = -1/Δx^4         # right boundary value in the second-to-last row

    return A, F, E, B
end


"""
$(SIGNATURES)

Evaluate the polynomial reaction term ``F u^{⟨2⟩} + E u^{⟨3⟩}`` at a state
vector `u`.
"""
function reaction_term(u::AbstractVector, F::AbstractMatrix, E::AbstractMatrix)
    return F * ⊘(u, 2) + E * ⊘(u, 3)
end


"""
$(SIGNATURES)

Integrate the Swift-Hohenberg model using the Crank-Nicolson (linear) Explicit (nonlinear) method,
or, in other words, the Semi-Implicit Crank-Nicolson (SICN) method without control.
"""
function integrate_model_without_control_SICN(tdata, u0; linear_matrix, quadratic_matrix, cubic_matrix,
                                              const_stepsize=false)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    F = quadratic_matrix
    E = cubic_matrix

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            nl = reaction_term(u[:, j-1], F, E)
            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + nl * Δt)
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            nl = reaction_term(u[:, j-1], F, E)
            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + nl * Δt)
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Swift-Hohenberg model using the Semi-Implicit Crank-Nicolson (SICN) method with control input.
"""
function integrate_model_with_control_SICN(tdata, u0, input; linear_matrix, quadratic_matrix, cubic_matrix,
                                           control_matrix, const_stepsize=false)
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

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            nl = reaction_term(u[:, j-1], F, E)
            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + nl * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            nl = reaction_term(u[:, j-1], F, E)
            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + nl * Δt + 0.5 * B * Δt * (input[:,j-1] + input[:,j]))
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Swift-Hohenberg model using the Crank-Nicolson (linear) Adams-Bashforth (nonlinear) method (CNAB)
with control.
"""
function integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix, quadratic_matrix, cubic_matrix,
                                           control_matrix, const_stepsize=false, nl_jm1=nothing)
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

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            nl = reaction_term(u[:, j-1], F, E)
            if j == 2 && isnothing(nl_jm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + nl * Δt + 0.5 * B * (input[:,j-1]+input[:,j]) * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + nl * 3*Δt/2 - nl_jm1 * Δt/2 + 0.5 * B * (input[:,j-1] + input[:,j]) * Δt)
            end
            nl_jm1 = nl
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            nl = reaction_term(u[:, j-1], F, E)
            if j == 2 && isnothing(nl_jm1)
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + nl * Δt + 0.5 * B * (input[:,j-1]+input[:,j]) * Δt)
            else
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + nl * 3*Δt/2 - nl_jm1 * Δt/2 +
                                             0.5 * B * (input[:,j-1] + input[:,j]) * Δt)
            end
            nl_jm1 = nl
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Swift-Hohenberg model using the Crank-Nicolson (linear) Adams-Bashforth (nonlinear) method (CNAB)
without control.
"""
function integrate_model_without_control_CNAB(tdata, u0; linear_matrix, quadratic_matrix, cubic_matrix,
                                              const_stepsize=false, nl_jm1=nothing)
    xdim = length(u0)
    tdim = length(tdata)
    u = zeros(xdim, tdim)
    u[:, 1] = u0

    A = linear_matrix
    F = quadratic_matrix
    E = cubic_matrix

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:tdim
            nl = reaction_term(u[:, j-1], F, E)
            if j == 2 && isnothing(nl_jm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + nl * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + nl * 3*Δt/2 - nl_jm1 * Δt/2)
            end
            nl_jm1 = nl
        end
    else
        for j in 2:tdim
            Δt = tdata[j] - tdata[j-1]
            nl = reaction_term(u[:, j-1], F, E)
            if j == 2 && isnothing(nl_jm1)
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + nl * Δt)
            else
                u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + nl * 3*Δt/2 - nl_jm1 * Δt/2)
            end
            nl_jm1 = nl
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrate the Swift-Hohenberg model using the Crank-Nicolson Adams-Bashforth (CNAB) or
the Semi-Implicit Crank-Nicolson (SICN) method.

# Arguments
- `tdata::AbstractArray{T}`: time data
- `u0::AbstractArray{T}`: initial condition
- `input::AbstractArray{T}=[]`: input data (Dirichlet boundary values)

# Keyword Arguments
- `linear_matrix::AbstractArray{T,2}`: linear matrix A
- `quadratic_matrix::AbstractArray{T,2}`: quadratic matrix F
- `cubic_matrix::AbstractArray{T,2}`: cubic matrix E
- `control_matrix::AbstractArray{T,2}`: control matrix B
- `system_input::Bool=false`: system input flag
- `const_stepsize::Bool=false`: constant step size flag
- `nl_jm1::AbstractArray{T}=nothing`: reaction term at the previous time step (CNAB only)
- `integrator_type::Symbol=:CNAB`: `:SICN` or `:CNAB`

# Returns
- `u::Array{T,2}`: integrated model states

# Notes
- If `system_input` is true, then `control_matrix` must be provided.
- If `const_stepsize` is true, then the time step size is assumed to be constant.
- If `nl_jm1` is provided, then the reaction term at the previous time step is used to start AB-2.
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
    nl_jm1 = haskey(kwargs, :nl_jm1) ? kwargs[:nl_jm1] : nothing
    if haskey(kwargs, :integrator_type)
        integrator_type = kwargs[:integrator_type]
        @assert integrator_type ∈ (:SICN, :CNAB) "Invalid integrator type. Choose from (:SICN, :CNAB), where SICN is Semi-Implicit Crank-Nicolson and CNAB is Crank-Nicolson Adams-Bashforth"
    else
        integrator_type = :CNAB
    end

    if system_input
        if integrator_type == :SICN
            integrate_model_with_control_SICN(tdata, u0, input; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix,
                                              cubic_matrix=cubic_matrix, control_matrix=control_matrix, const_stepsize=const_stepsize)
        else
            integrate_model_with_control_CNAB(tdata, u0, input; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix,
                                              cubic_matrix=cubic_matrix, control_matrix=control_matrix,
                                              const_stepsize=const_stepsize, nl_jm1=nl_jm1)
        end
    else
        if integrator_type == :SICN
            integrate_model_without_control_SICN(tdata, u0; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix,
                                                 cubic_matrix=cubic_matrix, const_stepsize=const_stepsize)
        else
            integrate_model_without_control_CNAB(tdata, u0; linear_matrix=linear_matrix, quadratic_matrix=quadratic_matrix,
                                                 cubic_matrix=cubic_matrix, const_stepsize=const_stepsize, nl_jm1=nl_jm1)
        end
    end
end


# ============================================================================
# Fast SICN / CNAB integrator (Crank-Nicolson on the linear A,
# explicit / Adams-Bashforth on the reaction term `F u² + E u³`).
# A is pentadiagonal, so the tridiagonal solver does not apply:
#   * periodic  → circulant → FFT diagonalization (FastCirculant1DSolver)
#   * Dirichlet → symmetric pentadiagonal → eigendecomposition with a cached
#                 inverse (FastDenseSolver), O(N²) per step
# ============================================================================

"""
$(SIGNATURES)

Build a fast Crank-Nicolson-flavoured solver for the Swift-Hohenberg model
(bakes `α = Δt/2`). Periodic BCs use the FFT-based circulant solver; hinged
Dirichlet BCs give a symmetric pentadiagonal operator, which is
eigendecomposed once (`FastDenseSolver`).
"""
function build_fast_solver(model::SwiftHohenbergModel, params::Dict;
                           scheme::Symbol=:CN, Δt::Real=model.Δt)
    @assert scheme === :CN "Swift-Hohenberg fast integrators use Crank-Nicolson (scheme=:CN)"
    α = Float64(Δt) / 2
    out = finite_diff_model(model, params)
    A = out[1]
    if model.BC === :periodic
        return FastCirculant1DSolver(A, α)
    elseif model.BC === :dirichlet
        return FastDenseSolver(Matrix(A), α)
    else
        return FactorizedSolver(sparse(A), α)
    end
end


"""
$(SIGNATURES)

Fast SICN or CNAB integrator for the 1D Swift-Hohenberg model.

## Arguments
- `model::SwiftHohenbergModel`
- `solver::AbstractFastSolver`: from [`build_fast_solver`](@ref) with `scheme=:CN`
- `tdata::AbstractVector`: time grid (uniform spacing assumed)
- `u0::AbstractVector`: initial condition
- `input::AbstractArray=Float64[]`: boundary input matrix (Dirichlet only)

## Keyword Arguments
- `quadratic_matrix`: F from `finite_diff_model`
- `cubic_matrix`: E from `finite_diff_model`
- `control_matrix=nothing`: B from `finite_diff_model`
- `integrator_type::Symbol=:CNAB`: `:SICN` or `:CNAB`
- `nl_jm1=nothing`: reaction term at the previous step, seed for AB-2 (CNAB only)
"""
function integrate_model_fast(model::SwiftHohenbergModel, solver::AbstractFastSolver,
                              tdata::AbstractVector, u0::AbstractVector,
                              input::AbstractArray=Float64[];
                              quadratic_matrix, cubic_matrix, control_matrix=nothing,
                              integrator_type::Symbol=:CNAB,
                              nl_jm1=nothing)
    @assert integrator_type ∈ (:SICN, :CNAB) "integrator_type must be :SICN or :CNAB"
    Xdim = length(u0)
    Tdim = length(tdata)
    u = zeros(Xdim, Tdim)
    u[:, 1] = u0
    Δt = tdata[2] - tdata[1]
    F = quadratic_matrix
    E = cubic_matrix

    has_input = control_matrix !== nothing && !isempty(input)
    if has_input
        input = adjust_input(input, size(control_matrix, 2), Tdim)
    end

    rhs = Vector{Float64}(undef, Xdim)
    tmp = Vector{Float64}(undef, Xdim)
    nl  = Vector{Float64}(undef, Xdim)

    @inbounds for j in 2:Tdim
        uold = u[:, j-1]
        mulIpA!(rhs, solver, uold)                 # rhs = (I + Δt/2 A) u_old

        # Reaction term  nl = F u² + E u³
        mul!(nl, F, ⊘(uold, 2))
        mul!(tmp, E, ⊘(uold, 3))
        @. nl = nl + tmp

        if integrator_type === :SICN || (j == 2 && nl_jm1 === nothing)
            @. rhs = rhs + Δt * nl
        else
            @. rhs = rhs + (3*Δt/2) * nl - (Δt/2) * nl_jm1
        end

        # Boundary input
        if has_input
            @views begin
                mul!(tmp, control_matrix, input[:, j-1])
                @. rhs = rhs + 0.5 * Δt * tmp
                mul!(tmp, control_matrix, input[:, j])
                @. rhs = rhs + 0.5 * Δt * tmp
            end
        end

        linsolve!(view(u, :, j), solver, rhs)
        nl_jm1 = copy(nl)
    end
    return u
end

end
