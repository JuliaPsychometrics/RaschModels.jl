mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
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

"""
    irf
"""
function irf(model::RaschModel, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _irf(theta, beta, y)
end

function _irf(theta, beta, y)
    exp_linpred = exp.(theta .- beta)
    prob = @. exp_linpred / (1 + exp_linpred)
    return ifelse(y == 1, prob, 1 .- prob)
end

"""
    iif
"""
function iif(model::RaschModel, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _iif(theta, beta, y)
end

_iif(theta, beta, y) = _irf(theta, beta, y) .* _irf(theta, beta, 1 - y)

"""
    expected_score
"""
function expected_score(model::RaschModel{SamplingEstimate}, theta::Real, is)
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)
    for i in is
        score .+= irf(model, theta, i, 1)
    end
    return score
end

function expected_score(model::RaschModel{PointEstimate}, theta::Real, is)
    score = zero(Float64)
    for i in is
        score += irf(model, theta, i, 1)
    end
    return score
end

expected_score(model::RaschModel, theta::Real) = expected_score(model, theta, 1:size(model.data, 2))

"""
    information
"""
function information(model::RaschModel{SamplingEstimate}, theta::Real, is)
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)
    for i in is
        info .+= iif(model, theta, i, 1)
    end
    return info
end

function information(model::RaschModel{PointEstimate}, theta::Real, is)
    info = zero(Float64)
    for i in is
        info += iif(model, theta, i, 1)
    end
    return info
end

information(model::RaschModel, theta::Real) = information(model, theta, 1:size(model.data, 2))

# Turing implementation
function _turing_model(::Type{RaschModel}; priors)
    @model function rasch_model(y, i, p; I=maximum(i), P=maximum(p), priors=priors)
        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta
        beta ~ filldist(mu_beta + priors.beta_norm * sigma_beta, I)
        Turing.@addlogprob! sum(logpdf.(BernoulliLogit.(theta[p] .- beta[i]), y))
    end
end
