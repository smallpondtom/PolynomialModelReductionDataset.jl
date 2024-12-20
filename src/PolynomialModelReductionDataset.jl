module PolynomialModelReductionDataset

using DocStringExtensions
using FFTW
using Kronecker: ⊗
using LinearAlgebra
using SparseArrays
using UniqueKronecker

export AbstractModel

"""
    AbstractModel

Abstract type for the model.
"""
abstract type AbstractModel end

# Utility functions
include("utilities/adjust_input.jl")

# 1D models
include("1D/AllenCahn.jl")
include("1D/Burgers.jl")
include("1D/DampedGardnerBurgers.jl")
include("1D/FisherKPP.jl")
include("1D/FitzHughNagumo.jl")
include("1D/Gardner.jl")
include("1D/Heat1D.jl")
include("1D/KuramotoSivashinsky.jl")
include("1D/ModifiedKortewegDeVries.jl")
include("1D/ModifiedKortewegDeVriesBurgers.jl")
using .AllenCahn: AllenCahnModel
using .Burgers: BurgersModel
using .DampedGardnerBurgers: DampedGardnerBurgersModel
using .Heat1D: Heat1DModel
using .FisherKPP: FisherKPPModel
using .FitzHughNagumo: FitzHughNagumoModel
using .Gardner: GardnerModel
using .KuramotoSivashinsky: KuramotoSivashinskyModel
using .ModifiedKortewegDeVries: ModifiedKortewegDeVriesModel
using .ModifiedKortewegDeVriesBurgers: ModifiedKortewegDeVriesBurgersModel

# 2D models
include("2D/Heat2D.jl")
using .Heat2D: Heat2DModel

end