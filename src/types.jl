mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: ItemResponseModel
    data::DT
    pars::PT
end

AbstractItemResponseModels.response_type(::Type{<:RaschModel}) = Dichotomous
AbstractItemResponseModels.person_dimensionality(::Type{<:RaschModel}) = Univariate
AbstractItemResponseModels.item_dimensionality(::Type{<:RaschModel}) = Univariate
AbstractItemResponseModels.estimation_type(::Type{<:RaschModel{ET,PT}}) where {ET,PT} = ET

function AbstractItemResponseModels.irf(model::RaschModel{PointEstimate,PT}, theta::Real, i, y::Real) where {PT}
    checkresponsetype(response_type(model), y)
    beta = getindex(model.pars, i)
    return _irf(theta, beta, y)
end

function AbstractItemResponseModels.irf(model::RaschModel{SamplingEstimate,PT}, theta::Real, i, y::Real) where {PT}
    checkresponsetype(response_type(model), y)
    beta = getindex(model.pars, i)
    return _irf(theta, beta, y)
end

function _irf(theta, beta, y)
    exp_linpred = exp.(theta .- beta)
    prob = @. exp_linpred / (1 + exp_linpred)
    return ifelse(y == 1, prob, 1 .- prob)
end
