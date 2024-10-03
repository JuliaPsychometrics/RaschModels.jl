using RaschModels
using Test
using AbstractItemResponseModels
using AbstractItemResponseModels.Tests
using MCMCChains
using Turing
using Optim
using LinearAlgebra

Turing.setprogress!(false)

@testset "RaschModels.jl" begin
    include("utils.jl")
    include("test_interface.jl")
    include("algorithms.jl")
    # include("models/models.jl")
end
