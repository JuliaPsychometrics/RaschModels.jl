# interface
"""
    $(SIGNATURES)

Extract the prior distributions (a [`Prior`](@ref) object) from a Bayesian Rasch Model.
"""
prior(model::AbstractRaschModel{SamplingEstimate}) = model.prior

"""
    $(SIGNATURES)

Extract the posterior distribution (a `MCMCChains.Chains` object) from a Bayesian Rasch Model.
"""
posterior(model::AbstractRaschModel{SamplingEstimate}) = model.posterior

# all bayesian/sampling based models can reuse irf, iif, expected_score, and information
# from the implementation in ItemResponseFunctions.jl
function irf(model::AbstractRaschModel{SamplingEstimate}, theta, i, y)
    M = model_type(model)
    betas = view(model.item_parameters, :, i)
    probs = [irf(M, theta, beta, y) for beta in betas]
    return parent(probs)
end

function irf(model::AbstractRaschModel{SamplingEstimate}, theta, i)
    M = model_type(model)
    betas = view(model.item_parameters, :, i)
    probs = [irf(M, theta, beta) for beta in betas]
    return parent(probs)
end

function iif(model::AbstractRaschModel{SamplingEstimate}, theta, i, y)
    M = model_type(model)
    betas = view(model.item_parameters, :, i)
    info = [iif(M, theta, beta, y) for beta in betas]
    return parent(info)
end

function iif(model::AbstractRaschModel{SamplingEstimate}, theta, i)
    M = model_type(model)
    betas = view(model.item_parameters, :, i)
    info = [iif(M, theta, beta) for beta in betas]
    return parent(info)
end

function expected_score(
    model::AbstractRaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    M = model_type(model)
    betas = view(model.item_parameters, :, is)
    scores = [expected_score(M, theta, beta; scoring_function) for beta in eachrow(betas)]
    return parent(scores)
end

function information(
    model::AbstractRaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function::F = one,
) where {F}
    M = model_type(model)
    betas = view(model.item_parameters, :, is)
    info = [information(M, theta, beta; scoring_function) for beta in eachrow(betas)]
    return parent(info)
end

include("BayesianRaschModel.jl")
include("BayesianRatingScaleModel.jl")
include("BayesianPartialCreditModel.jl")
