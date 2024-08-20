"""
    $(TYPEDEF)

A struct representing a fitted Bayesian Partial Credit Model.

## Fields
$(FIELDS)
"""
struct BayesianPartialCreditModel{
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

function BayesianPartialCreditModel(data, prior, chain)
    item_parameters = make_pcm_item_parameters(chain)
    person_parameters = make_person_parameters(chain)
    return BayesianPartialCreditModel(
        data,
        prior,
        chain,
        item_parameters,
        person_parameters,
    )
end

response_type(::Type{<:BayesianPartialCreditModel}) = AbstractItemResponseModels.Ordinal
model_type(::Type{<:BayesianPartialCreditModel}) = PCM

function make_pcm_item_parameters(chain::Chains)
    parnames_beta = namesingroup(chain, :beta)
    parnames_tau = namesingroup(chain, :tau)

    pars = map(eachindex(parnames_beta)) do i
        betas = Array(chain[parnames_beta[i]])
        tnames = filter(x -> occursin("tau[$i]", string(x)), parnames_tau)
        thresholds = Array(chain[tnames])

        pars = map(betas, eachrow(thresholds)) do b, t
            return ItemParameters(PCM, (; b, t))
        end

        return pars
    end

    return DimArray(hcat(pars...), (:iteration, :item))
end

function turing_model(::Type{BayesianPartialCreditModel}; priors = Prior())
    @model function partial_credit_model(
        y,
        i,
        p,
        ::Type{T} = Float64;
        priors = priors,
    ) where {T}
        I = maximum(i)
        P = maximum(p)
        K = [maximum(y[i.==item]) - 1 for item in 1:I]

        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta

        beta ~ filldist(mu_beta + priors.beta_norm * sigma_beta, I)
        tau = Vector{T}.(undef, K)

        for item in eachindex(tau)
            tau[item] ~ filldist(Normal(), K[item])
        end

        betas = [ItemParameters(PCM; b = beta[i], t = tau[i]) for i in 1:I]

        @addlogprob! sum(logpdf.(PartialCredit.(theta[p], betas[i]), y))
    end
end

function PartialCredit(theta::Real, beta::ItemParameters; check_args = false)
    probs = irf(PCM, theta, beta)
    return Categorical(probs; check_args)
end
