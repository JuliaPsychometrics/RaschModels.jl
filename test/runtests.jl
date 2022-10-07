using RaschModels
using Test
using AbstractItemResponseModels.Tests
using MCMCChains
using Turing

Turing.setprogress!(false)

@testset "RaschModels.jl" begin
    include("test_interface.jl")
    include("utils.jl")
    include("fit.jl")
end
