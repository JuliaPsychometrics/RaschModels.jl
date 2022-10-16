@testset "AbstractItemResponseModels.jl Interface" begin
    data = rand(0:1, 10, 2)

    # MCMC algorithms
    test_interface(RaschModel, data, MH())
end
