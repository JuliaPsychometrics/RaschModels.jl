struct FrequentistRaschModel{T<:AbstractMatrix,U<:AbstractDimArray,V} <:
       RaschModel{PointEstimate}
    "the original response data matrix"
    data::T
    "raw estimates"
    estimates::Any
    "An array of item parameters"
    item_parameters::U
    "An array of person parameters"
    person_parameters::V
end

function FrequentistRaschModel(data, estimate; alg_pp = WLE())
    item_parameters = make_rasch_item_parameters(estimate)
    person_pars = person_parameters(OnePL, data, item_parameters, alg_pp)
    return FrequentistRaschModel(data, estimate, item_parameters, person_pars)
end

response_type(::Type{<:FrequentistRaschModel}) = AbstractItemResponseModels.Dichotomous
model_type(::Type{<:FrequentistRaschModel}) = OnePL

function make_rasch_item_parameters(estimate::StatisticalModel)
    item_pars = [ItemParameters(OnePL; b) for b in estimate.values]
    return DimArray(item_pars, :item)
end
