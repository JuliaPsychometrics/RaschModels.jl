using RaschModels
using Test
using AbstractItemResponseModels.Tests

@testset "RaschModels.jl" begin
    @testset "AbstractItemResponseModels.jl Interface" begin
        data = rand(0:1, 10, 2)
        test_interface(RaschModel, data, type=:optim)
        test_interface(RaschModel, data, type=:mcmc)
    end
end
