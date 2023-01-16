struct RatingScaleModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
end

response_type(::Type{<:RatingScaleModel}) = AbstractItemResponseModels.Ordinal
person_dimensionality(::Type{<:RatingScaleModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:RatingScaleModel}) = AbstractItemResponseModels.Univariate
estimation_type(::Type{<:RatingScaleModel{ET,DT,PT}}) where {ET,DT,PT} = ET

"""
    getitempars
"""
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

"""
    irf
"""
function irf(model::RatingScaleModel{ET,DT,PT}, theta::Real, i, y::Real) where {ET<:SamplingEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    beta, tau = getitempars(model, i)
    eta = theta .- (beta .+ tau)
    p = probs.(PartialCredit.(eachrow(eta)))
    return getindex.(p, Int(y))
end

function irf(model::RatingScaleModel{ET,DT,PT}, theta::Real, i, y::Real) where {ET<:PointEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    beta, tau = getitempars(model, i)
    eta = theta .- (beta .+ tau)
    p = probs(PartialCredit(eta))
    return getindex(p, Int(y))
end

"""
    iif
"""
function iif(model::RatingScaleModel, theta::Real, i, y::Real)
    category_prob = irf(model, theta, i, y)
    return category_prob .* (1 .- category_prob)
end

"""
    expected_score
"""

"""
    information
"""

# Turing implementation
@model function ratingscale(y, i, p)
    I = maximum(i)
    P = maximum(p)
    K = maximum(y) - 1

    theta ~ filldist(Normal(), P)
    mu_beta ~ Normal()
    sigma_beta ~ InverseGamma(3, 2)
    beta ~ filldist(Normal(mu_beta, sigma_beta), I)
    tau ~ filldist(Normal(), K)

    eta = [theta[p[n]] .- (beta[i[n]] .+ tau) for n in eachindex(y)]
    Turing.@addlogprob! sum(logpdf.(PartialCredit.(eta), y))
end
