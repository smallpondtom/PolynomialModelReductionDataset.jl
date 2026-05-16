"""
    Fast implicit linear solvers for PDE time integration.

This submodule provides reusable, allocation-free solvers for systems of the
form `(I - α A) u_new = rhs` (and the companion `(I + α A) x` product) that
arise repeatedly in implicit / semi-implicit time stepping. The factor `α`
is baked into the solver at construction time:

  * Backward Euler  :  α = Δt
  * Crank–Nicolson  :  α = Δt / 2

A typical workflow is therefore

```julia
solver = FastSymTridiagSolver(A, Δt/2)        # CN-flavoured
linsolve!(u_new, solver, rhs)                  # (I - α A) u_new = rhs
mulIpA!(out,    solver, u_old)                 # out = (I + α A) u_old
update_timestep!(solver, Δt_new/2)             # rebuild cached factor in O(N)
```

Available concrete solver types and the structure they exploit:

| Type                    | Structure of A                                 | Per-step cost      |
|-------------------------|------------------------------------------------|--------------------|
| `FastSymTridiagSolver`  | 1D symmetric tridiagonal (Dirichlet/Neumann)   | O(N²) (BLAS-2)     |
| `FastCirculant1DSolver` | 1D circulant (periodic)                        | O(N log N) (FFT)   |
| `FastKronSumSolver`     | 2D `(Ay ⊗ I) + (I ⊗ Ax)` with sym-tridiag      | O(Nx Ny (Nx+Ny))   |
| `FastFFT2DSolver`       | 2D circulant Kronecker sum (periodic, periodic)| O(N log N) (FFT)   |
| `FastDenseSolver`       | dense unstructured (e.g. reduced-order)        | O(r²) (BLAS-2)     |
| `FactorizedSolver`      | any sparse A (LU factorization once)           | O(nnz) per solve   |
"""
module FastSolvers

using DocStringExtensions
using FFTW
using LinearAlgebra
using SparseArrays

export AbstractFastSolver,
       FastSymTridiagSolver, FastCirculant1DSolver,
       FastKronSumSolver, FastFFT2DSolver,
       FastDenseSolver, FactorizedSolver,
       linsolve!, mulIpA!, update_timestep!,
       backward_euler_solve!,
       build_fast_solver, integrate_model_fast


"""
$(SIGNATURES)

Build a fast implicit linear solver for `model`. Each concrete model adds its
own method that selects the appropriate structured solver based on its
boundary conditions and discretization. Keyword `scheme` chooses the
flavour of α to bake in:

  * `:BE`   ⇒ α = `Δt`        (backward Euler)
  * `:CN`   ⇒ α = `Δt / 2`    (Crank–Nicolson, also used by SICN / CNAB)

Additional keyword arguments are forwarded to the model-specific builder.
"""
function build_fast_solver end


"""
$(SIGNATURES)

Integrate a model using a pre-built fast implicit solver. Per-PDE methods
live in each PDE module; see e.g. [`Heat2D.integrate_model_fast`](@ref).
"""
function integrate_model_fast end


"""
$(TYPEDEF)

Abstract supertype for all fast linear solvers that pre-process `A` once so
that each time step reduces to a structured solve / multiply.
"""
abstract type AbstractFastSolver end


# ============================================================================
# 1D symmetric tridiagonal solver
# ============================================================================

"""
$(TYPEDEF)

Fast implicit solver for a 1-D symmetric tridiagonal operator `A`. Eigen-
decomposes `A = V Λ Vᵀ` once and caches both `1/(1 − α λ_i)` and
`1 + α λ_i` so that each solve / multiply is a pair of matrix-vector products
plus an elementwise scaling.

## Fields
$(TYPEDFIELDS)
"""
struct FastSymTridiagSolver{T<:Real} <: AbstractFastSolver
    "Eigenvectors of A (real, N×N)"
    V::Matrix{T}
    "Transpose of V, materialized for BLAS"
    Vt::Matrix{T}
    "Eigenvalues of A (length N)"
    λ::Vector{T}
    "1 / (1 - α λ_i) — cached inverse"
    inv_denom::Vector{T}
    "1 + α λ_i — cached forward (CN) factor"
    mul_factor::Vector{T}
    "Spatial dimension"
    N::Int
    "Workspace (length N)"
    tmp::Vector{T}
end


"""
$(SIGNATURES)

Build a solver from a `SymTridiagonal` operator and α (= Δt for BE, Δt/2 for CN).
"""
function FastSymTridiagSolver(A::SymTridiagonal{T}, α::Real) where {T<:Real}
    N = size(A, 1)
    Ed = eigen(A)
    λ = Ed.values
    α_T = convert(T, α)
    inv_denom = T(1) ./ (T(1) .- α_T .* λ)
    mul_factor = T(1) .+ α_T .* λ
    return FastSymTridiagSolver{T}(
        Ed.vectors, Matrix(Ed.vectors'),
        λ, inv_denom, mul_factor, N, zeros(T, N),
    )
end

"""
$(SIGNATURES)

Build a solver from any matrix whose symmetric tridiagonal part captures `A`
exactly (e.g. a sparse Dirichlet/Neumann Laplacian).
"""
function FastSymTridiagSolver(A::AbstractMatrix, α::Real)
    return FastSymTridiagSolver(SymTridiagonal(Matrix(A)), α)
end


# ============================================================================
# 1D circulant solver (FFT-based)
# ============================================================================

"""
$(TYPEDEF)

Fast implicit solver for a 1-D circulant operator `A` (periodic BCs). The
eigenvalues are `λ = fft(first column of A)`; the time loop reduces to a pair
of in-place FFTs plus an elementwise scaling.

## Fields
$(TYPEDFIELDS)
"""
struct FastCirculant1DSolver{P,IP} <: AbstractFastSolver
    "In-place forward FFT plan"
    plan_f::P
    "In-place inverse FFT plan"
    plan_if::IP
    "Eigenvalues of A (complex, length N)"
    λ::Vector{ComplexF64}
    "1 / (1 - α λ) — cached inverse"
    inv_denom::Vector{ComplexF64}
    "1 + α λ — cached forward (CN) factor"
    mul_factor::Vector{ComplexF64}
    "Spatial dimension"
    N::Int
    "Complex workspace of length N (FFT scratchpad)"
    buffer::Vector{ComplexF64}
end


"""
$(SIGNATURES)

Build a solver for a circulant matrix `A` by computing its eigenvalues as
`fft(A[:, 1])`.
"""
function FastCirculant1DSolver(A::AbstractMatrix, α::Real)
    N = size(A, 1)
    c = Vector{ComplexF64}(A[:, 1])
    λ = fft(c)
    inv_denom = 1.0 ./ (1.0 .- α .* λ)
    mul_factor = 1.0 .+ α .* λ
    buffer = zeros(ComplexF64, N)
    plan_f = plan_fft!(buffer; flags=FFTW.MEASURE)
    plan_if = plan_ifft!(buffer; flags=FFTW.MEASURE)
    return FastCirculant1DSolver(plan_f, plan_if, λ,
                                  inv_denom, mul_factor, N, buffer)
end


# ============================================================================
# 2D Kronecker-sum solver:  A = (Ay ⊗ I_Nx) + (I_Ny ⊗ Ax)
# ============================================================================

"""
$(TYPEDEF)

Fast implicit solver for a 2-D operator with Kronecker-sum structure

    A = (Ay ⊗ I_Nx) + (I_Ny ⊗ Ax)

where both 1-D factors are symmetric tridiagonal. Eigendecomposing Ax and Ay
once turns each solve / multiply into four dense matrix-matrix products
(BLAS-3, very cache friendly) plus an elementwise scaling. Total per-step
cost is `O(Nx Ny (Nx + Ny))`.

## Fields
$(TYPEDFIELDS)
"""
struct FastKronSumSolver{T<:Real} <: AbstractFastSolver
    "Eigenvectors of Ax"
    Vx::Matrix{T}
    "Eigenvectors of Ay"
    Vy::Matrix{T}
    "Vxᵀ materialized for BLAS"
    Vxt::Matrix{T}
    "Vyᵀ materialized for BLAS"
    Vyt::Matrix{T}
    "Eigenvalues of Ax (length Nx)"
    λx::Vector{T}
    "Eigenvalues of Ay (length Ny)"
    λy::Vector{T}
    "1 ./ (1 .- α .* (λx .+ λy'))  — cached inverse"
    inv_denom::Matrix{T}
    "1 .+ α .* (λx .+ λy')  — cached forward (CN) factor"
    mul_factor::Matrix{T}
    "x-dimension"
    Nx::Int
    "y-dimension"
    Ny::Int
    "Workspace 1 (Nx × Ny)"
    tmp1::Matrix{T}
    "Workspace 2 (Nx × Ny)"
    tmp2::Matrix{T}
end


"""
$(SIGNATURES)

Build a 2-D Kronecker-sum solver from the two 1-D `SymTridiagonal` factors
and α (= Δt for BE, Δt/2 for CN).
"""
function FastKronSumSolver(Ax::SymTridiagonal{T}, Ay::SymTridiagonal{T},
                            α::Real) where {T<:Real}
    Nx = size(Ax, 1); Ny = size(Ay, 1)
    Ex = eigen(Ax)
    Ey = eigen(Ay)
    α_T = convert(T, α)
    inv_denom = T(1) ./ (T(1) .- α_T .* (Ex.values .+ Ey.values'))
    mul_factor = T(1) .+ α_T .* (Ex.values .+ Ey.values')
    return FastKronSumSolver{T}(
        Ex.vectors, Ey.vectors,
        Matrix(Ex.vectors'), Matrix(Ey.vectors'),
        Ex.values, Ey.values,
        inv_denom, mul_factor,
        Nx, Ny,
        zeros(T, Nx, Ny), zeros(T, Nx, Ny),
    )
end

"""
$(SIGNATURES)

Convenience: accept any matrices `Ax, Ay` and treat their symmetric
tridiagonal parts as the 1-D operators.
"""
function FastKronSumSolver(Ax::AbstractMatrix, Ay::AbstractMatrix, α::Real)
    return FastKronSumSolver(SymTridiagonal(Matrix(Ax)),
                              SymTridiagonal(Matrix(Ay)), α)
end

"""
$(SIGNATURES)

Heat2D-style convenience constructor. Builds the standard second-difference
1-D factors `Ax, Ay` for the 2-D heat operator with grid spacings `Δx, Δy`,
diffusion coefficient `μ`, and bakes in α = `Δt` (backward Euler) by default.
"""
function FastKronSumSolver(Nx::Integer, Ny::Integer,
                            Δx::Real, Δy::Real, μ::Real, Δt::Real)
    Ax = SymTridiagonal(fill(-2μ/Δx^2, Nx), fill(μ/Δx^2, Nx-1))
    Ay = SymTridiagonal(fill(-2μ/Δy^2, Ny), fill(μ/Δy^2, Ny-1))
    return FastKronSumSolver(Ax, Ay, Δt)
end


# ============================================================================
# 2D FFT-based solver (periodic Kronecker sum of circulants)
# ============================================================================

"""
$(TYPEDEF)

Fast implicit solver for a 2-D doubly-periodic operator (Kronecker sum of two
1-D circulants). Each step is a single in-place 2-D FFT, an elementwise
scaling, and an inverse FFT.

## Fields
$(TYPEDFIELDS)
"""
struct FastFFT2DSolver{P,IP} <: AbstractFastSolver
    "In-place 2-D forward FFT plan"
    plan_f::P
    "In-place 2-D inverse FFT plan"
    plan_if::IP
    "1-D eigenvalues in x"
    λx::Vector{ComplexF64}
    "1-D eigenvalues in y"
    λy::Vector{ComplexF64}
    "1 ./ (1 .- α .* (λx .+ λy'))  — cached inverse"
    inv_denom::Matrix{ComplexF64}
    "1 .+ α .* (λx .+ λy')  — cached forward (CN) factor"
    mul_factor::Matrix{ComplexF64}
    "x-dimension"
    Nx::Int
    "y-dimension"
    Ny::Int
    "Complex (Nx × Ny) workspace"
    buffer::Matrix{ComplexF64}
end


"""
$(SIGNATURES)

Build a 2-D FFT-based solver from the 1-D eigenvalue vectors `λx, λy` of the
two periodic factor matrices. These can be either real or complex; for real
circulants, the eigenvalues of `A = (Ay ⊗ I) + (I ⊗ Ax)` are the outer sum
`λx + λy'`.
"""
function FastFFT2DSolver(λx::AbstractVector, λy::AbstractVector, α::Real)
    Nx, Ny = length(λx), length(λy)
    λxc = Vector{ComplexF64}(λx)
    λyc = Vector{ComplexF64}(λy)
    inv_denom = 1.0 ./ (1.0 .- α .* (λxc .+ transpose(λyc)))
    mul_factor = 1.0 .+ α .* (λxc .+ transpose(λyc))
    buffer = zeros(ComplexF64, Nx, Ny)
    plan_f = plan_fft!(buffer; flags=FFTW.MEASURE)
    plan_if = plan_ifft!(buffer; flags=FFTW.MEASURE)
    return FastFFT2DSolver(plan_f, plan_if, λxc, λyc,
                            inv_denom, mul_factor, Nx, Ny, buffer)
end

"""
$(SIGNATURES)

Heat2D-style convenience constructor for periodic BCs. Builds the
second-difference eigenvalues `λ_k = (μ/Δx²)(2 cos(2π(k-1)/N) - 2)` for both
axes.
"""
function FastFFT2DSolver(Nx::Integer, Ny::Integer,
                          Δx::Real, Δy::Real, μ::Real, Δt::Real)
    λx = [μ/Δx^2 * (2cos(2π*(k-1)/Nx) - 2) for k in 1:Nx]
    λy = [μ/Δy^2 * (2cos(2π*(k-1)/Ny) - 2) for k in 1:Ny]
    return FastFFT2DSolver(λx, λy, Δt)
end


# ============================================================================
# Dense unstructured solver (ROM use case)
# ============================================================================

"""
$(TYPEDEF)

Fast implicit solver for a dense, unstructured matrix `A` (typically from a
reduced-order model). Eigendecomposes `A` once and pre-computes the full
inverse `(I - α A)⁻¹` (and `I + α A` for CN-style steps) so that each step
is a single dense matrix-vector multiply.

If `A` is nearly defective (cond(V) > `cond_threshold`), falls back to a
direct LU-based inverse for robustness; a warning is issued.

## Fields
$(TYPEDFIELDS)
"""
struct FastDenseSolver <: AbstractFastSolver
    "(I - α A)⁻¹, real r×r matrix applied via mul! each step"
    M_inv::Matrix{Float64}
    "I + α A, real r×r matrix used by mulIpA!"
    IpA::Matrix{Float64}
    "Eigenvectors of A (complex, stored for update_timestep!)"
    V::Matrix{ComplexF64}
    "Inverse of V (complex)"
    Vinv::Matrix{ComplexF64}
    "Eigenvalues of A (complex)"
    λ::Vector{ComplexF64}
    "Stored copy of A (real, for update_timestep! rebuilding IpA)"
    A::Matrix{Float64}
    "Dimension of the system"
    r::Int
    "Whether the solver was constructed via eigendecomposition (false = LU fallback)"
    eigen_based::Bool
end


"""
$(SIGNATURES)

Construct a fast solver for a dense matrix `A` with baked-in α.

## Keyword Arguments
- `cond_threshold::Real=1e12`: condition number of V above which the
  eigendecomposition is considered too ill-conditioned and the LU fallback
  is used.
"""
function FastDenseSolver(A::AbstractMatrix{<:Real}, α::Real;
                          cond_threshold::Real=1e12)
    r = size(A, 1)
    @assert size(A, 2) == r "A must be square, got size $(size(A))"
    A_d = Matrix{Float64}(A)

    F = eigen(A_d)
    V    = ComplexF64.(F.vectors)
    Vinv = inv(V)
    λ    = ComplexF64.(F.values)

    κ = opnorm(V, 2) * opnorm(Vinv, 2)
    IpA = Matrix(I(r) + α * A_d)

    if κ > cond_threshold
        @warn "Eigenvector matrix is ill-conditioned (κ(V) = $(round(κ; sigdigits=3))). " *
              "Falling back to direct inverse of (I - α A) for robustness."
        M_inv = real.(inv(I - α * A_d))
        return FastDenseSolver(M_inv, IpA, V, Vinv, λ, A_d, r, false)
    end

    M_inv = _build_M_inv(V, Vinv, λ, α)
    return FastDenseSolver(M_inv, IpA, V, Vinv, λ, A_d, r, true)
end


function _build_M_inv(V::Matrix{ComplexF64}, Vinv::Matrix{ComplexF64},
                       λ::Vector{ComplexF64}, α::Real)
    d = 1.0 ./ (1.0 .- α .* λ)
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


# ============================================================================
# Generic LU-factorized fallback (any sparse / dense A)
# ============================================================================

"""
$(TYPEDEF)

Fallback fast solver for arbitrary `A`. Forms `M = I - α A` once and stores
its LU factorization; each `linsolve!` is a single triangular back-solve.
Use this when none of the structured solvers apply (e.g. asymmetric tridiag,
matrices with Δt baked into boundary rows).

## Fields
$(TYPEDFIELDS)
"""
mutable struct FactorizedSolver{TF,TA<:AbstractMatrix} <: AbstractFastSolver
    "LU factorization of (I - α A)"
    fact::TF
    "Stored A (used for mulIpA! and update_timestep!)"
    A::TA
    "Baked-in α"
    α::Float64
    "System dimension"
    N::Int
end


"""
$(SIGNATURES)
"""
function FactorizedSolver(A::AbstractMatrix, α::Real)
    N = size(A, 1)
    @assert size(A, 2) == N "A must be square"
    M = I(N) - α * A
    fact = lu(M)
    return FactorizedSolver(fact, A, Float64(α), N)
end


# ============================================================================
# Common operations:  linsolve!  /  mulIpA!  /  update_timestep!
# ============================================================================

# ---------- FastSymTridiagSolver ----------

function linsolve!(out::AbstractVector, F::FastSymTridiagSolver, rhs::AbstractVector)
    mul!(F.tmp, F.Vt, rhs)
    @inbounds @. F.tmp = F.tmp * F.inv_denom
    mul!(out, F.V, F.tmp)
    return out
end

function mulIpA!(out::AbstractVector, F::FastSymTridiagSolver, x::AbstractVector)
    mul!(F.tmp, F.Vt, x)
    @inbounds @. F.tmp = F.tmp * F.mul_factor
    mul!(out, F.V, F.tmp)
    return out
end

function update_timestep!(F::FastSymTridiagSolver, α_new::Real)
    @inbounds @. F.inv_denom  = 1 / (1 - α_new * F.λ)
    @inbounds @. F.mul_factor = 1 + α_new * F.λ
    return F
end

# ---------- FastCirculant1DSolver ----------

function linsolve!(out::AbstractVector, F::FastCirculant1DSolver, rhs::AbstractVector)
    @inbounds for k in 1:F.N
        F.buffer[k] = rhs[k]
    end
    F.plan_f * F.buffer
    @inbounds @. F.buffer = F.buffer * F.inv_denom
    F.plan_if * F.buffer
    @inbounds for k in 1:F.N
        out[k] = real(F.buffer[k])
    end
    return out
end

function mulIpA!(out::AbstractVector, F::FastCirculant1DSolver, x::AbstractVector)
    @inbounds for k in 1:F.N
        F.buffer[k] = x[k]
    end
    F.plan_f * F.buffer
    @inbounds @. F.buffer = F.buffer * F.mul_factor
    F.plan_if * F.buffer
    @inbounds for k in 1:F.N
        out[k] = real(F.buffer[k])
    end
    return out
end

function update_timestep!(F::FastCirculant1DSolver, α_new::Real)
    @inbounds @. F.inv_denom  = 1 / (1 - α_new * F.λ)
    @inbounds @. F.mul_factor = 1 + α_new * F.λ
    return F
end

# ---------- FastKronSumSolver ----------

function linsolve!(out::AbstractVector, F::FastKronSumSolver, rhs::AbstractVector)
    R = reshape(rhs, F.Nx, F.Ny)
    U = reshape(out, F.Nx, F.Ny)
    mul!(F.tmp1, F.Vxt, R)
    mul!(F.tmp2, F.tmp1, F.Vy)
    @inbounds @. F.tmp2 = F.tmp2 * F.inv_denom
    mul!(F.tmp1, F.Vx, F.tmp2)
    mul!(U,       F.tmp1, F.Vyt)
    return out
end

function mulIpA!(out::AbstractVector, F::FastKronSumSolver, x::AbstractVector)
    X = reshape(x,   F.Nx, F.Ny)
    U = reshape(out, F.Nx, F.Ny)
    mul!(F.tmp1, F.Vxt, X)
    mul!(F.tmp2, F.tmp1, F.Vy)
    @inbounds @. F.tmp2 = F.tmp2 * F.mul_factor
    mul!(F.tmp1, F.Vx, F.tmp2)
    mul!(U,       F.tmp1, F.Vyt)
    return out
end

function update_timestep!(F::FastKronSumSolver, α_new::Real)
    @inbounds @. F.inv_denom  = 1 / (1 - α_new * (F.λx + F.λy'))
    @inbounds @. F.mul_factor = 1 + α_new * (F.λx + F.λy')
    return F
end

# ---------- FastFFT2DSolver ----------

function linsolve!(out::AbstractVector, F::FastFFT2DSolver, rhs::AbstractVector)
    @inbounds for k in eachindex(rhs)
        F.buffer[k] = rhs[k]
    end
    F.plan_f * F.buffer
    @inbounds @. F.buffer = F.buffer * F.inv_denom
    F.plan_if * F.buffer
    @inbounds for k in eachindex(out)
        out[k] = real(F.buffer[k])
    end
    return out
end

function mulIpA!(out::AbstractVector, F::FastFFT2DSolver, x::AbstractVector)
    @inbounds for k in eachindex(x)
        F.buffer[k] = x[k]
    end
    F.plan_f * F.buffer
    @inbounds @. F.buffer = F.buffer * F.mul_factor
    F.plan_if * F.buffer
    @inbounds for k in eachindex(out)
        out[k] = real(F.buffer[k])
    end
    return out
end

function update_timestep!(F::FastFFT2DSolver, α_new::Real)
    @inbounds @. F.inv_denom  = 1 / (1 - α_new * (F.λx + F.λy'))
    @inbounds @. F.mul_factor = 1 + α_new * (F.λx + F.λy')
    return F
end

# ---------- FastDenseSolver ----------

function linsolve!(out::AbstractVector, F::FastDenseSolver, rhs::AbstractVector)
    mul!(out, F.M_inv, rhs)
    return out
end

function mulIpA!(out::AbstractVector, F::FastDenseSolver, x::AbstractVector)
    mul!(out, F.IpA, x)
    return out
end

function update_timestep!(solver::FastDenseSolver, α_new::Real)
    if !solver.eigen_based
        @warn "Solver was built via LU fallback due to ill-conditioned " *
              "eigenvectors. update_timestep! uses the eigendecomposition " *
              "regardless; results may be inaccurate."
    end
    solver.M_inv .= _build_M_inv(solver.V, solver.Vinv, solver.λ, α_new)
    solver.IpA  .= I(solver.r) + α_new * solver.A
    return solver
end

# ---------- FactorizedSolver ----------

function linsolve!(out::AbstractVector, F::FactorizedSolver, rhs::AbstractVector)
    ldiv!(out, F.fact, rhs)
    return out
end

function mulIpA!(out::AbstractVector, F::FactorizedSolver, x::AbstractVector)
    mul!(out, F.A, x)
    @inbounds @. out = x + F.α * out
    return out
end

function update_timestep!(F::FactorizedSolver, α_new::Real)
    M = I(F.N) - α_new * F.A
    F.fact = lu(M)
    F.α = Float64(α_new)
    return F
end


# ============================================================================
# Legacy alias retained for backward compatibility with the original Heat2D
# fast Backward-Euler API (`backward_euler_solve!`).
# ============================================================================

"""
$(SIGNATURES)

Backward compatibility alias for [`linsolve!`](@ref). Identical behaviour: it
applies the cached factor of `(I - α A)⁻¹` baked into the solver. Prefer
`linsolve!` in new code.
"""
backward_euler_solve!(unew, F::AbstractFastSolver, rhs) = linsolve!(unew, F, rhs)

end # module
