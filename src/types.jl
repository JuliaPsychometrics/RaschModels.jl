abstract type AbstractRaschModel <: ItemResponseModel end

mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
end

response_type(::Type{<:RaschModel}) = AbstractItemResponseModels.Dichotomous
person_dimensionality(::Type{<:RaschModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:RaschModel}) = AbstractItemResponseModels.Univariate
estimation_type(::Type{<:RaschModel{ET,PT}}) where {ET,PT} = ET

# TODO: rename!
struct RatingScaleModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
end

response_type(::Type{<:RatingScaleModel}) = AbstractItemResponseModels.Ordinal
person_dimensionality(::Type{<:RatingScaleModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:RatingScaleModel}) = AbstractItemResponseModels.Univariate
estimation_type(::Type{<:RatingScaleModel{ET,DT,PT}}) where {ET,DT,PT} = ET

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

function getitempars(model::RatingScaleModel{ET,DT,PT}, i::Int) where {ET,DT,PT<:Chains}
    beta_name = "beta[" * string(i) * "]"
    betas = getindex(model.pars, beta_name)

    threshold_names = namesingroup(model.pars, :tau)
    thresholds = getindex(model.pars, threshold_names)

    return vec(betas), Array(thresholds)
end

function getitempars(model::RatingScaleModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    pars = coef(model.pars)

    beta_name = Symbol("beta[" * string(i) * "]")
    beta = getindex(pars, beta_name)

    threshold_names = filter(x -> occursin("tau", x), string.(params(model.pars)))
    thresholds = getindex(pars, Symbol.(threshold_names))
    return beta, vec(thresholds)
end
