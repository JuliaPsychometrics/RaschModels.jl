mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
end

response_type(::Type{<:RaschModel}) = AbstractItemResponseModels.Dichotomous
person_dimensionality(::Type{<:RaschModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:RaschModel}) = AbstractItemResponseModels.Univariate
estimation_type(::Type{<:RaschModel{ET,DT,PT}}) where {ET,DT,PT} = ET

"""
    getitempars(model::RaschModel, i)

Fetch the item parameters of `model` for item `i`.
"""
function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:Chains}
    parname = Symbol("beta[", i, "]")
    betas = model.pars.value[var=parname]
    return vec(betas)
end

function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    parname = Symbol("beta[", i, "]")
    betas = coef(model.pars)
    return getindex(betas, parname)
end

"""
    irf
"""
function irf(model::RaschModel, theta, i, y=1)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _irf.(RaschModel, theta, beta, y)
end

function _irf(::Type{RaschModel}, theta, beta, y)
    exp_linpred = exp(theta - beta)
    prob = exp_linpred / (1 + exp_linpred)
    return ifelse(y == 1, prob, 1 - prob)
end

"""
    iif
"""
function iif(model::RaschModel, theta, i, y=1)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _iif.(RaschModel, theta, beta, y)
end

function _iif(::Type{RaschModel}, theta, beta, y)
    prob = _irf(RaschModel, theta, beta, y)
    info = prob * (1 - prob)
    return info
end

"""
    expected_score
"""
function expected_score(model::RaschModel{SamplingEstimate}, theta, is)
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)
    for i in is
        score .+= irf(model, theta, i, 1)
    end
    return score
end

function expected_score(model::RaschModel{PointEstimate}, theta, is)
    score = zero(Float64)
    for i in is
        score += irf(model, theta, i, 1)
    end
    return score
end

function expected_score(model::RaschModel, theta)
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items)
    return score
end

"""
    information
"""
function information(model::RaschModel{SamplingEstimate}, theta, is)
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)
    for i in is
        info .+= iif(model, theta, i, 1)
    end
    return info
end

function information(model::RaschModel{PointEstimate}, theta, is)
    info = zero(Float64)
    for i in is
        info += iif(model, theta, i, 1)
    end
    return info
end

function information(model::RaschModel, theta)
    items = 1:size(model.data, 2)
    info = information(model, theta, items)
    return info
end

# Turing implementation
function _turing_model(::Type{RaschModel}; priors)
    @model function rasch_model(y, i, p; I=maximum(i), P=maximum(p), priors=priors)
        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta
        beta ~ filldist(mu_beta + priors.beta_norm * sigma_beta, I)

        eta = theta[p] .- beta[i]
        Turing.@addlogprob! sum(logpdf.(Rasch.(eta), y))
    end
end

function Rasch(eta::Real)
    return BernoulliLogit(eta)
end
