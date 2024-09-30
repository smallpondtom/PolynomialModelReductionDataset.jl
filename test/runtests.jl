using PolynomialModelReductionDataset
using Test

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

@testset "PolynomialModelReductionDataset" begin
    testfile("utilities.jl")
end