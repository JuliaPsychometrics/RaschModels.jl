@testset "AbstractItemResponseModels.jl Interface" begin
    # available_models = [RaschModel, PartialCreditModel, RatingScaleModel]
    available_models = [RaschModel]

    for model in available_models
        @testset "$model" begin
            if AbstractItemResponseModels.response_type(model) ==
               AbstractItemResponseModels.Dichotomous
                data = rand(0:1, 100, 5)
            else
                data = rand(1:4, 100, 5)
            end

            @testset "SamplingEstimate" test_interface(model, data, MH(), 100)
            @testset "PointEstimate" test_interface(model, data, CML())
        end
    end
end
