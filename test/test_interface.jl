@testset "AbstractItemResponseModels.jl Interface" begin
    available_models = [RaschModel, PartialCreditModel]

    for model in available_models
        @testset "$model" begin
            if AbstractItemResponseModels.response_type(model) == AbstractItemResponseModels.Dichotomous
                data = rand(0:1, 10, 2)
            else
                data = rand(1:4, 10, 2)
            end

            test_interface(model, data, MH(), 100)
            test_interface(model, data, MLE())
        end
    end
end
