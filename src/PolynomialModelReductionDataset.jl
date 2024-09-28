module PolynomialModelReductionDataset

using DocStringExtensions
using LinearAlgebra
using SparseArrays
using UniqueKronecker

export AbstractModel

"""
    AbstractModel

Abstract type for the model.
"""
abstract type AbstractModel end

# 1D models
include("1D/AllenCahn.jl")
include("1D/Burgers.jl")
include("1D/ChafeeInfante.jl")
include("1D/FisherKPP.jl")
include("1D/FitzHughNagumo.jl")
include("1D/Gardner.jl")
include("1D/Heat1D.jl")
include("1D/KuramotoSivashinsky.jl")

# 2D models
include("2D/Heat2D.jl")

end