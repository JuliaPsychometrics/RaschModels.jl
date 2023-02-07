mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
    parnames::Vector{Symbol}
end

response_type(::Type{<:RaschModel}) = AbstractItemResponseModels.Dichotomous
estimation_type(::Type{<:RaschModel{ET,DT,PT}}) where {ET,DT,PT} = ET

"""
    getitempars(model::RaschModel, i)

Fetch the item parameters of `model` for item `i`.
"""
function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:Chains}
    parname = Symbol("beta[", i, "]")
    betas = model.pars.value[var = parname]
    return vec(betas)
end

function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    parname = Symbol("beta[", i, "]")
    betas = coef(model.pars)
    return getindex(betas, parname)
end

"""
    getpersonpars(model::RaschModel, p)

Fetch the person parameters of `model` for person `p`.
"""
function getpersonpars(model::RaschModel{ET,DT,PT}, p) where {ET,DT,PT<:Chains}
    parname = Symbol("theta[", p, "]")
    thetas = model.pars.value[var = parname]
    return vec(thetas)
end

function getpersonpars(model::RaschModel{ET,DT,PT}, p) where {ET,DT,PT<:StatisticalModel}
    parname = Symbol("theta[", p, "]")
    thetas = coef(model.pars)
    return getindex(thetas, parname)
end

"""
    irf(model::RaschModel, theta, i, y)
    irf(model::RaschModel, theta, i)

Evaluate the item response function for a dichotomous Rasch model for item `i` at the ability
value `theta`.

If the response value `y` is omitted, the item response probability for a correct response
`y = 1` is returned.
"""
function irf(model::RaschModel, theta, i, y = 1)
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
    iif(model::RaschModel, theta, i, y)
    iif(model::RaschModel, theta, i)

Evaluate the item information function for a dichotomous Rasch model for item `i` at the
ability value `theta`.

If the response value `y` is omitted, the item information for a correct response `y = 1` is
returned.
"""
function iif(model::RaschModel, theta, i, y = 1)
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
    expected_score(model::RaschModel, theta, is; scoring_function)
    expected_score(model::RaschModel, theta; scoring_function)

Calculate the expected score for a dichotomous Rasch model at ability value `theta` for a
set of items `is`.

`is` can either be a single item index, an array of item indices, or a range of values.
If `is` is omitted, the expected score for the whole test is calculated.
"""
function expected_score(
    model::RaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function = identity,
)
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)
    for i in is
        score .+= scoring_function.(irf(model, theta, i, 1))
    end
    return score
end

function expected_score(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function = identity,
)
    score = zero(Float64)
    for i in is
        score += scoring_function(irf(model, theta, i, 1))
    end
    return score
end

function expected_score(model::RaschModel, theta; scoring_function = identity)
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items; scoring_function)
    return score
end

"""
    information(model::RaschModel, theta, is; scoring_function)
    information(model::RaschModel, theta; scoring_function)

Calculate the information for a dichotomous Rasch model at the ability value `theta` for a
set of items `is`.

`is` can either be a single item index, an array of item indices, or a range of values.
If `is` is omitted, the information for the whole test is calculated.
"""
function information(
    model::RaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function = identity,
)
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)
    for i in is
        info .+= scoring_function.(iif(model, theta, i, 1))
    end
    return info
end

function information(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function = identity,
)
    info = zero(Float64)
    for i in is
        info += scoring_function(iif(model, theta, i, 1))
    end
    return info
end

function information(model::RaschModel, theta; scoring_function = identity)
    items = 1:size(model.data, 2)
    info = information(model, theta, items; scoring_function)
    return info
end

# Turing implementation
function _turing_model(::Type{RaschModel}; priors)
    @model function rasch_model(y, i, p; I = maximum(i), P = maximum(p), priors = priors)
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
