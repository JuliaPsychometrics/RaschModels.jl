mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: ItemResponseModel
    data::DT
    pars::PT
end

AbstractItemResponseModels.response_type(::Type{<:RaschModel}) = Dichotomous
AbstractItemResponseModels.person_dimensionality(::Type{<:RaschModel}) = Univariate
AbstractItemResponseModels.item_dimensionality(::Type{<:RaschModel}) = Univariate
AbstractItemResponseModels.estimation_type(::Type{<:RaschModel{ET,PT}}) where {ET,PT} = ET
