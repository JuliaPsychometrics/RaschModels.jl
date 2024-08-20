"""
    $(TYPEDEF)

A struct representing a fitted Bayesian Rating Scale Model.

## Fields
$(FIELDS)
"""
struct BayesianRatingScaleModel{
    T<:AbstractMatrix,
    U<:Prior,
    V<:Chains,
    W<:AbstractDimArray,
    X<:AbstractDimArray,
} <: AbstractRaschModel{SamplingEstimate}
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

function BayesianRatingScaleModel(data, prior, chain)
    item_parameters = make_rsm_item_parameters(chain)
    person_parameters = make_person_parameters(chain)
    return BayesianRatingScaleModel(data, prior, chain, item_parameters, person_parameters)
end

response_type(::Type{<:BayesianRatingScaleModel}) = AbstractItemResponseModels.Ordinal
model_type(::Type{<:BayesianRatingScaleModel}) = RSM

function make_rsm_item_parameters(chain::Chains)
    parnames_beta = namesingroup(chain, :beta)
    betas = Array(chain[parnames_beta])

    parnames_tau = namesingroup(chain, :tau)
    thresholds = Array(chain[parnames_tau])

    pars = broadcast(betas, eachrow(thresholds)) do b, t
        return ItemParameters(RSM, (; b, t))
    end

    return DimArray(pars, (:iteration, :item))
end

function turing_model(::Type{BayesianRatingScaleModel}; priors)
    @model function rating_scale_model(
        y,
        i,
        p;
        I = maximum(i),
        P = maximum(p),
        K = maximum(y) - 1,
        priors = priors,
    )
        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta
        beta ~ filldist(mu_beta + priors.beta_norm * sigma_beta, I)
        tau ~ filldist(priors.tau, K)

        betas = [ItemParameters(RSM, b = beta[i], t = tau) for i in 1:I]
        @addlogprob! sum(logpdf.(RatingScale.(theta[p], betas[i]), y))
    end
end

function RatingScale(theta::Real, beta::ItemParameters; check_args = false)
    probs = irf(RSM, theta, beta)
    return Categorical(probs; check_args)
end
