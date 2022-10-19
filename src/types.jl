mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: ItemResponseModel
    data::DT
    pars::PT
end

response_type(::Type{<:RaschModel}) = AbstractItemResponseModels.Dichotomous
person_dimensionality(::Type{<:RaschModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:RaschModel}) = AbstractItemResponseModels.Univariate
estimation_type(::Type{<:RaschModel{ET,PT}}) where {ET,PT} = ET

"""
    getitempars(model::RaschModel, i)

Fetch the item parameters of `model` for item `i`.
"""
function getitempars(model::RaschModel{ET,DT,PT}, i::Int) where {ET,DT,PT<:Chains}
    parname = "beta[" * string(i) * "]"
    betas = getindex(model.pars, parname)
    return vec(betas)
end

function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    pars = coef(model.pars)
    parname = Symbol("beta[" * string(i) * "]")
    return getindex(pars, parname)
end
