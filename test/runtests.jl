using RaschModels
using Test
using AbstractItemResponseModels
using AbstractItemResponseModels.Tests
using MCMCChains
using Turing
using Optim

Turing.setprogress!(false)

@testset "RaschModels.jl" begin
    include("utils.jl")
    include("test_interface.jl")
    include("models/RaschModel.jl")
    include("models/PolytomousRaschModel.jl")
    include("models/RatingScaleModel.jl")
    include("models/PartialCreditModel.jl")
end
