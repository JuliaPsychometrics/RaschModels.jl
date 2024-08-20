# all frequentist models can reuse irf, iif, expected_score, and information from the
# implementation in ItemResponseFunctions.jl
function irf(model::AbstractRaschModel{PointEstimate}, theta, i, y)
    M = model_type(model)
    beta = model.item_parameters[i]
    return irf(M, theta, beta, y)
end

function irf(model::AbstractRaschModel{PointEstimate}, theta, i)
    M = model_type(model)
    beta = model.item_parameters[i]
    return irf(M, theta, beta)
end

function iif(model::AbstractRaschModel{PointEstimate}, theta, i, y)
    M = model_type(model)
    beta = model.item_parameters[i]
    return iif(M, theta, beta, y)
end

function iif(model::AbstractRaschModel{PointEstimate}, theta, i)
    M = model_type(model)
    beta = model.item_parameters[i]
    return iif(M, theta, beta)
end

function expected_score(
    model::AbstractRaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    M = model_type(model)
    betas = view(model.item_parameters, is)
    score = expected_score(M, theta, betas; scoring_function)
    return score
end

function information(
    model::AbstractRaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = one,
) where {F}
    M = model_type(model)
    betas = view(model.item_parameters, is)
    info = information(M, theta, betas; scoring_function)
    return info
end

include("FrequentistRaschModel.jl")

# include("RaschModel.jl")
# include("PolytomousRaschModel.jl")
# include("PartialCreditModel.jl")
# include("RatingScaleModel.jl")
