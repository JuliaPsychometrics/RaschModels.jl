"""
    PolytomousRaschModel

A type representing Rasch Models with polytomous responses.
"""
abstract type PolytomousRaschModel{ET<:EstimationType,PT} <: AbstractRaschModel end

response_type(::Type{<:PolytomousRaschModel}) = AbstractItemResponseModels.Ordinal
estimation_type(::Type{<:PolytomousRaschModel{ET,PT}}) where {ET,PT} = ET

"""
    getpersonlocations(model::PolytomousRaschModel, p)

Fetch the person parameters of `model` for person `p`.
"""
function getpersonlocations(model::PolytomousRaschModel{SamplingEstimate}, p)
    parname = Symbol("theta[", p, "]")
    thetas = model.pars.value[var = parname]
    return vec(thetas)
end

function getpersonlocations(
    model::PolytomousRaschModel{ET,PT},
    p,
) where {ET,PT<:StatisticalModel}
    parname = Symbol("theta[", p, "]")
    thetas = coef(model.pars)
    return getindex(thetas, parname)
end

"""
    getitemlocations(model::PolytomousRaschModel, i, y)

Fetch item parameters from a fitted `model`.
"""
function getitemlocations(model::PolytomousRaschModel, i, y)
    difficulty = getitemdifficulty(model, i)
    if y == 1
        return difficulty
    else
        thresholds = getthresholds(model, i, y - 1)
        return difficulty + thresholds
    end
end

function getitemdifficulty(model::PolytomousRaschModel{SamplingEstimate}, i)
    parname = model.parnames_beta[i]
    difficulty = vec(view(model.pars.value, var = parname))
    return difficulty
end

function getitemdifficulty(
    model::PolytomousRaschModel{ET,PT},
    i,
) where {ET,PT<:StatisticalModel}
    parname = model.parnames_beta[i]
    difficulty = model.pars.values[parname]
    return difficulty
end

function getthresholds(model::PolytomousRaschModel{SamplingEstimate}, i, c)
    parname = getthresholdnames(model, i, c)
    thresholds = vec(view(model.pars.value, var = parname))
    return thresholds
end

function getthresholds(model::PolytomousRaschModel{SamplingEstimate}, i)
    threshold_names = getthresholdnames(model, i)
    thresholds = view(model.pars.value, var = threshold_names)
    n_iter, n_pars, n_chains = size(thresholds)
    thresholds_permuted = permutedims(thresholds, (1, 3, 2))
    threshold_mat = Matrix(reshape(thresholds_permuted, n_iter * n_chains, n_pars))
    return threshold_mat
end

function getthresholds(
    model::PolytomousRaschModel{ET,PT},
    i,
    c,
) where {ET,PT<:StatisticalModel}
    parname = getthresholdnames(model, i, c)
    threshold = model.pars.values[parname]
    return threshold
end

function getthresholds(
    model::PolytomousRaschModel{ET,PT},
    i,
) where {ET,PT<:StatisticalModel}
    threshold_names = getthresholdnames(model, i)
    thresholds = vec(view(model.pars.values, threshold_names))
    return thresholds
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
    beta = getitemdifficulty(model, i)
    thresholds = getthresholds(model, i)

    n_samples, n_thresholds = size(thresholds)

    probs = similar(thresholds, n_samples, n_thresholds + 1)
    extended = zeros(eltype(probs), n_thresholds + 1)

    eta = @. theta - (beta + thresholds)

    for (i, x) in enumerate(eachrow(eta))
        probs[i, :] = _irf(PolytomousRaschModel, extended, x)
    end

    return probs
end

function irf(model::PolytomousRaschModel{SamplingEstimate}, theta, i, y)
    checkresponsetype(response_type(model), y)
    probs = irf(model, theta, i)
    return probs[:, Int(y)]
end

function irf(model::PolytomousRaschModel{PointEstimate}, theta, i)
    beta = getitemdifficulty(model, i)
    thresholds = getthresholds(model, i)

    n_thresholds = length(thresholds)
    extended = zeros(eltype(thresholds), n_thresholds + 1)

    eta = @. theta - (beta + thresholds)

    probs = _irf(PolytomousRaschModel, extended, eta)
    return probs
end

function irf(model::PolytomousRaschModel{PointEstimate}, theta, i, y)
    checkresponsetype(response_type(model), y)
    probs = irf(model, theta, i)
    return probs[Int(y)]
end

function _irf(::Type{PolytomousRaschModel}, extended, eta)
    extended .= 0.0
    extended[2:end] = eta
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
        category_information[i, :] =
            _icif(PolytomousRaschModel, category_probs[i, :], item_information[i])
    end

    return category_information
end

function iif(model::PolytomousRaschModel{PointEstimate}, theta, i)
    category_probs = irf(model, theta, i)
    item_information = _iif(model, theta, i)
    category_information = _icif.(PolytomousRaschModel, category_probs, item_information)
    return category_information
end

function _iif(
    model::PolytomousRaschModel{SamplingEstimate},
    theta,
    i;
    scoring_function::F = identity,
) where {F}
    category_probs = irf(model, theta, i)
    score = expected_score(model, theta, i; scoring_function)

    info = similar(score)

    for i in eachindex(info)
        probs = vec(view(category_probs, i, :))
        info[i] = _iif(PolytomousRaschModel, probs, score[i]; scoring_function)
    end

    return info
end

function _iif(
    model::PolytomousRaschModel{PointEstimate},
    theta,
    i;
    scoring_function::F = identity,
) where {F}
    category_probs = irf(model, theta, i)
    score = expected_score(model, theta, i; scoring_function)
    info = _iif(PolytomousRaschModel, category_probs, score; scoring_function)
    return info
end

function _iif(
    ::Type{PolytomousRaschModel},
    probs,
    score;
    scoring_function::F = identity,
) where {F}
    info = zero(Float64)
    for (category, prob) in enumerate(probs)
        info += (scoring_function(category) - score)^2 * prob
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
function expected_score(
    model::PolytomousRaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    n_samples = size(model.pars, 1)
    score = zeros(Float64, n_samples)

    for i in is
        probs = irf(model, theta, i)

        for (category, prob) in enumerate(eachcol(probs))
            @. score += prob * scoring_function(category)
        end
    end

    return score
end

function expected_score(
    model::PolytomousRaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    score = zero(Float64)

    for i in is
        probs = irf(model, theta, i)

        for (category, prob) in enumerate(probs)
            score += prob * scoring_function(category)
        end
    end

    return score
end

function expected_score(
    model::PolytomousRaschModel,
    theta;
    scoring_function::F = identity,
) where {F}
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items; scoring_function)
    return score
end

"""
    information(model::PolytomousRaschModel, theta, is)
    information(model::PolytomousRaschModel, theta)

Calculate the information for a polytomous Rasch model at `theta` for a set of items `is`.

`is` can either be a single item index, an array of item indices, or a range of values.
If `is` is omitted, the information for the whole test is calculated.
"""
function information(
    model::PolytomousRaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function = identity,
)
    n_samples = size(model.pars, 1)
    info = zeros(Float64, n_samples)

    for i in is
        item_information = _iif(model, theta, i; scoring_function)
        info += item_information
    end

    return info
end

function information(
    model::PolytomousRaschModel{PointEstimate},
    theta,
    is;
    scoring_function = identity,
)
    info = zero(Float64)

    for i in is
        item_information = _iif(model, theta, i; scoring_function)
        info += item_information
    end

    return info
end

function information(model::PolytomousRaschModel, theta; scoring_function = identity)
    items = 1:size(model.data, 2)
    info = information(model, theta, items; scoring_function)
    return info
end
