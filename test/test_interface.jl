@testset "AbstractItemResponseModels.jl Interface" begin
    data_dichotomous = rand(0:1, 10, 2)
    data_ordinal = rand(1:4, 10, 3)

    # MCMC algorithms
    test_interface(RaschModel, data_dichotomous, MH(), 100)
    test_interface(RatingScaleModel, data_ordinal, MH(), 100)

    # point estimation
    test_interface(RaschModel, data_dichotomous, MLE())
    test_interface(RatingScaleModel, data_ordinal, MLE())
end
