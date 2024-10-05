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
    testfile("1D/kse.jl")
    testfile("1D/mKdV.jl")
    testfile("1D/mKdVB.jl")

    # 2D models
    testfile("2D/heat2d.jl")
end