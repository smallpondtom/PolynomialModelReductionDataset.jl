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

include("models/AllenCahn.jl")
include("models/Burgers.jl")
include("models/ChafeeInfante.jl")
include("models/FisherKPP.jl")
include("models/FitzHughNagumo.jl")
include("models/Gardner.jl")
include("models/Heat1D.jl")
include("models/Heat2D.jl")
include("models/KuramotoSivashinsky.jl")

end