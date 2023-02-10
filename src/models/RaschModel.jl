mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
    parnames_beta::Vector{Symbol}
end

response_type(::Type{<:RaschModel}) = AbstractItemResponseModels.Dichotomous
estimation_type(::Type{<:RaschModel{ET,DT,PT}}) where {ET,DT,PT} = ET

"""
    getitempars(model::RaschModel, i)

Fetch the item parameters of `model` for item `i`.
"""
function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:Chains}
    parname = model.parnames_beta[i]
    betas = vec(view(model.pars.value, var = parname))
    return betas
end

function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    parname = model.parnames_beta[i]
    betas = coef(model.pars)
    return getindex(betas, parname)
end

"""
    getpersonpars(model::RaschModel, p)

Fetch the person parameters of `model` for person `p`.
"""
function getpersonpars(model::RaschModel{ET,DT,PT}, p) where {ET,DT,PT<:Chains}
    parname = Symbol("theta[", p, "]")
    thetas = vec(view(model.pars.value, var = parname))
    return thetas
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
function irf(model::RaschModel{SamplingEstimate}, theta, i, y = 1)
    n_iter = length(getitempars(model, i))
    probs = zeros(Float64, n_iter)
    add_irf!(model, probs, theta, i, y)
    return probs
end

function irf(model::RaschModel{PointEstimate}, theta, i, y = 1)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _irf(RaschModel, theta, beta, y)
end

function add_irf!(
    model::RaschModel{SamplingEstimate},
    probs,
    theta,
    i,
    y;
    scoring_function::F = identity,
) where {F}
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)

    for j in eachindex(beta)
        probs[j] += _irf(RaschModel, theta, beta[j], y) * scoring_function(y)
    end

    return nothing
end

function _irf(::Type{RaschModel}, theta, beta, y)
    prob = logistic(theta - beta)
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
function iif(model::RaschModel{SamplingEstimate}, theta, i, y = 1)
    n_iter = length(getitempars(model, i))
    info = zeros(Float64, n_iter)
    add_iif!(model, info, theta, i, y)
    return info
end

function iif(model::RaschModel{PointEstimate}, theta, i, y = 1)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _iif(RaschModel, theta, beta)
end

function add_iif!(
    model::RaschModel{SamplingEstimate},
    info,
    theta,
    i,
    y;
    scoring_function::F = identity,
) where {F}
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)

    for j in eachindex(beta)
        info[j] += _iif(RaschModel, theta, beta[j]; scoring_function)
    end

    return nothing
end

function _iif(::Type{RaschModel}, theta, beta; scoring_function::F = identity) where {F}
    expected = _irf(RaschModel, theta, beta, 1) * scoring_function(1)
    info = zero(Float64)

    for y in 0:1
        prob = _irf(RaschModel, theta, beta, y)
        info += (scoring_function(y) - expected)^2 * prob
    end

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
    scoring_function::F = identity,
) where {F}
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)

    for i in is
        add_irf!(model, score, theta, i, 1; scoring_function)
    end

    return score
end

function expected_score(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    score = zero(Float64)
    for i in is
        score += irf(model, theta, i, 1) * scoring_function(1)
    end
    return score
end

function expected_score(model::RaschModel, theta; scoring_function::F = identity) where {F}
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
    scoring_function::F = identity,
) where {F}
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)

    for i in is
        add_iif!(model, info, theta, i, 1; scoring_function)
    end

    return info
end

function information(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    info = zero(Float64)
    for i in is
        beta = getitempars(model, i)
        info += _iif(RaschModel, theta, beta; scoring_function)
    end
    return info
end

function information(model::RaschModel, theta; scoring_function::F = identity) where {F}
    items = 1:size(model.data, 2)
    info = information(model, theta, items; scoring_function)
    return info
end

function _information(
    modeltype::Type{RaschModel},
    theta::AbstractFloat,
    betas::AbstractVector{<:AbstractFloat};
    scoring_function::F = identity,
) where {F}
    info = zero(Float64)
    for beta in betas
        info += _iif(modeltype, theta, beta; scoring_function)
    end
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
