using PolynomialModelReductionDataset
using LinearAlgebra
using Test

const pomoreda = PolynomialModelReductionDataset

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

# Helper: compare two state trajectories that may go unstable late in the
# simulation. Returns true if every entry up to the first column where either
# matrix becomes "non-finite or excessively large" matches within tolerance.
# Useful for stiff / chaotic PDEs (e.g. Gardner, DGB on long horizons) where
# the slow and fast paths agree until both lose stability — they may then
# diverge wildly even when the underlying scheme is mathematically identical.
# `blowup` defaults to 10× the larger of the two ICs' max abs (or 10 if both
# ICs are small), which catches the explosion well before round-off errors
# blow up the relative comparison.
function agrees_until_nan(A::AbstractMatrix, B::AbstractMatrix;
                          rtol::Real=1e-6, atol::Real=1e-9,
                          blowup::Real=max(10.0,
                                            10 * max(maximum(abs, @view(A[:, 1])),
                                                     maximum(abs, @view(B[:, 1])))))
    @assert size(A) == size(B)
    col_unstable(j) = begin
        a = @view A[:, j]; b = @view B[:, j]
        any(!isfinite, a) || any(!isfinite, b) ||
            maximum(abs, a) > blowup || maximum(abs, b) > blowup
    end
    nancol = findfirst(col_unstable, axes(A, 2))
    j_last = nancol === nothing ? size(A, 2) : nancol - 1
    j_last == 0 && return true   # both matrices unstable from the start (vacuously equal)
    return isapprox(@view(A[:, 1:j_last]), @view(B[:, 1:j_last]); rtol, atol)
end

@testset "PolynomialModelReductionDataset" begin
    # Utilities
    testfile("utilities.jl")

    # 1D models
    testfile("1D/heat1d.jl")
    testfile("1D/burgers.jl")
    testfile("1D/allencahn.jl")
    testfile("1D/dgb.jl")
    testfile("1D/fhn.jl")
    testfile("1D/fisherkpp.jl")
    testfile("1D/gardner.jl")
    testfile("1D/kawahara.jl")
    testfile("1D/kse.jl")
    testfile("1D/mKdV.jl")
    testfile("1D/mKdVB.jl")

    # 2D models
    testfile("2D/heat2d.jl")
    testfile("2D/allencahn2d.jl")
end