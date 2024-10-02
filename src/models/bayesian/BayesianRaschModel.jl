"""
    $(TYPEDEF)

A struct representing a fitted Bayesian Rasch Model.

## Fields
$(FIELDS)
"""
struct BayesianRaschModel{
    T<:AbstractMatrix,
    U<:Prior,
    V<:Chains,
    W<:AbstractDimArray,
    X<:AbstractDimArray,
} <: RaschModel{SamplingEstimate}
    "the original response data matrix"
    data::T
    "The prior distributions used for fitting the model"
    prior::U
    "The posterior distribution of the fitted model"
    posterior::V
    "An array of item parameters. The first dimension represents iterations and the second dimension represents the item"
    item_parameters::W
    "An array of person parameters. The first dimension represents iterations and the second dimension represents the person"
    person_parameters::X
end

function BayesianRaschModel(data, prior, chain)
    item_parameters = make_rasch_item_parameters(chain)
    person_parameters = make_person_parameters(chain)
    return BayesianRaschModel(data, prior, chain, item_parameters, person_parameters)
end

response_type(::Type{<:BayesianRaschModel}) = AbstractItemResponseModels.Dichotomous
model_type(::Type{<:BayesianRaschModel}) = OnePL

function make_rasch_item_parameters(chain::Chains)
    parnames = namesingroup(chain, :beta)
    arr = Array(chain[parnames])
    pars = [ItemParameters(OnePL; b) for b in arr]
    res = DimArray(pars, (:iteration, :item))
    return res
end

function make_person_parameters(chain::Chains)
    parnames = namesingroup(chain, :theta)
    arr = Array(chain[parnames])
    pars = DimArray(arr, (:iteration, :person))
    return pars
end

function turing_model(::Type{BayesianRaschModel}; priors = Prior())
    @model function rasch_model(y, i, p; I = maximum(i), P = maximum(p), priors = priors)
        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta
        beta ~ filldist(mu_beta + priors.beta_norm * sigma_beta, I)

        eta = theta[p] .- beta[i]
        @addlogprob! sum(logpdf.(Rasch.(eta), y))
    end
end

function Rasch(eta::Real)
    return BernoulliLogit(eta)
end
