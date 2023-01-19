"""
    PolytomousRaschModel

A type representing Rasch Models with polytomous responses.
"""
abstract type PolytomousRaschModel{ET<:EstimationType,PT} <: AbstractRaschModel end

response_type(::Type{<:PolytomousRaschModel}) = AbstractItemResponseModels.Ordinal
estimation_type(::Type{<:PolytomousRaschModel{ET,PT}}) where {ET,PT} = ET

"""
    getitempars(model::PolytomousRaschModel, i)

Fetch item parameters from a fitted model.

## Return value
For polytomous Rasch Models a tuple with item parameters `beta` and threshold parameteres `tau`
is returned.
"""
function getitempars(model::PolytomousRaschModel, i)
    beta = _get_item_parameter(model, i)
    tau = _get_item_thresholds(model, i)
    return (; beta, tau)
end

# MCMCChains
function _get_item_parameter(model::PolytomousRaschModel{ET,PT}, i) where {ET,PT<:Chains}
    parname = Symbol("beta[", i, "]")
    betas = model.pars.value[var=parname]
    return vec(betas)
end

# StatisticalModel
function _get_item_parameter(model::PolytomousRaschModel{ET,PT}, i) where {ET,PT<:StatisticalModel}
    parname = Symbol("beta[", i, "]")
    pars = coef(model.pars)
    return getindex(pars, parname)
end

"""
    irf(model::PolytomousRaschModel, theta, i, y)
    irf(model::PolytomousRaschModel, theta, i)

Evaluate the item response function for a polytomous Rasch model (Partial Credit Model or
Rating Scale Model) for item `i` at the ability value `theta`.

If the response value `y` is omitted, the item response probabilities for each category are
returned. To calculate expected scores for an item, see [`expected_score`](@ref).
"""
function irf(model::PolytomousRaschModel{SamplingEstimate}, theta, i)
    beta, thresholds = getitempars(model, i)
    n_samples, n_thresholds = size(thresholds)

    probs = similar(thresholds, n_samples, n_thresholds + 1)

    for i in 1:n_samples
        threshold_difficulty = view(beta, i) .+ view(thresholds, i, :)
        probs[i, :] = _irf(PolytomousRaschModel, theta, threshold_difficulty)
    end

    return probs
end

function irf(model::PolytomousRaschModel{SamplingEstimate}, theta, i, y)
    checkresponsetype(response_type(model), y)
    probs = irf(model, theta, i)
    return probs[:, Int(y)]
end

function irf(model::PolytomousRaschModel{PointEstimate}, theta, i)
    beta, thresholds = getitempars(model, i)
    threshold_difficulties = beta .+ thresholds
    probs = _irf(PolytomousRaschModel, theta, threshold_difficulties)
    return probs
end

function irf(model::PolytomousRaschModel{PointEstimate}, theta, i, y)
    checkresponsetype(response_type(model), y)
    probs = irf(model, theta, i)
    return probs[Int(y)]
end

function _irf(::Type{PolytomousRaschModel}, theta, beta)
    extended = vcat(zero(eltype(beta)), beta)
    cumsum!(extended, extended)
    softmax!(extended, extended)
    return extended
end

"""
    iif(model::PolytomousRaschModel, theta, i, y)
    iif(model::PolytomousRaschModel, theta, i)

Evaluate the item (category) information function for a polytomous Rasch model (Partial Credit
Model or Rating Scale Model) for item `i` at the ability value `theta`.

If the response value `y` is omitted, the item information for each category is returned.
To calculate the total information of an item, see [`@ref`](information).
"""
function iif(model::PolytomousRaschModel, theta, i, y)
    checkresponsetype(response_type(model), y)
    category_prob = irf(model, theta, i, y)
    item_information = _iif(model, theta, i)
    category_information = _icif.(PolytomousRaschModel, category_prob, item_information)
    return category_information
end

function iif(model::PolytomousRaschModel{SamplingEstimate}, theta, i)
    category_probs = irf(model, theta, i)
    n_samples = size(category_probs, 1)

    item_information = _iif(model, theta, i)
    category_information = similar(category_probs)

    for i in 1:n_samples
        category_information[i, :] = _icif(PolytomousRaschModel, category_probs[i, :], item_information[i])
    end

    return category_information
end

function iif(model::PolytomousRaschModel{PointEstimate}, theta, i)
    category_probs = irf(model, theta, i)
    item_information = _iif(model, theta, i)
    category_information = _icif.(PolytomousRaschModel, category_probs, item_information)
    return category_information
end

function _iif(model::PolytomousRaschModel{SamplingEstimate}, theta, i)
    category_probs = irf(model, theta, i)
    score = expected_score(model, theta, i)

    info = similar(score)

    for i in eachindex(info)
        info[i] = _iif(PolytomousRaschModel, category_probs[i, :], score[i])
    end

    return info
end

function _iif(model::PolytomousRaschModel{PointEstimate}, theta, i)
    category_probs = irf(model, theta, i)
    score = expected_score(model, theta, i)
    _iif(PolytomousRaschModel, category_probs, score)
end

function _iif(::Type{PolytomousRaschModel}, probs, score)
    info = zero(Float64)
    for (category, prob) in enumerate(probs)
        info += (category - score)^2 * prob
    end
    return info
end

function _icif(::Type{PolytomousRaschModel}, prob, item_information)
    return prob * item_information
end

"""
    expected_score(model::PolytomousRaschModel, theta, is)
    expected_score(model::PolytomousRaschModel, theta)

Calculate the expected score for a polytomous Rasch model at `theta` for a set of items `is`.

`is` can either be a single item index, an array of item indices, or a range of values.
If `is` is omitted, the expected score for the whole test is calculated.
"""
function expected_score(model::PolytomousRaschModel{SamplingEstimate}, theta, is)
    n_samples = size(model.pars, 1)
    score = zeros(Float64, n_samples)

    for i in is
        category_probs = irf(model, theta, i)
        categories = 1:size(category_probs, 2)
        category_scores = category_probs * categories
        score += category_scores
    end

    return score
end

function expected_score(model::PolytomousRaschModel{PointEstimate}, theta, is)
    score = zero(Float64)

    for i in is
        category_probs = irf(model, theta, i)
        categories = 1:length(category_probs)
        category_scores = category_probs .* categories
        score += sum(category_scores)
    end

    return score
end

function expected_score(model::PolytomousRaschModel, theta)
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items)
    return score
end

"""
    information(model::PolytomousRaschModel, theta, is)
    information(model::PolytomousRaschModel, theta)

Calculate the information for a polytomous Rasch model at `theta` for a set of items `is`.

`is` can either be a single item index, an array of item indices, or a range of values.
If `is` is omitted, the expected score for the whole test is calculated.
"""
function information(model::PolytomousRaschModel{SamplingEstimate}, theta, is)
    n_samples = size(model.pars, 1)
    info = zeros(Float64, n_samples)

    for i in is
        item_information = _iif(model, theta, i)
        info += item_information
    end

    return info
end

function information(model::PolytomousRaschModel{PointEstimate}, theta, is)
    info = zero(Float64)

    for i in is
        item_information = _iif(model, theta, i)
        info += item_information
    end

    return info
end

function information(model::PolytomousRaschModel, theta)
    items = 1:size(model.data, 2)
    info = information(model, theta, items)
    return info
end
