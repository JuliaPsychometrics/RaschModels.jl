"""
    RatingScaleModel <: PolytomousRaschModel

A type representing a Rating Scale Model
"""
struct RatingScaleModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <:
       PolytomousRaschModel{ET,PT}
    data::DT
    pars::PT
    parnames_beta::Vector{Symbol}
    parnames_tau::Vector{Symbol}
end

getthresholdnames(model::RatingScaleModel, i, c = :) = getindex(model.parnames_tau, c)

function _turing_model(::Type{RatingScaleModel}; priors)
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

        eta = [theta[p[n]] .- (beta[i[n]] .+ tau) for n in eachindex(y)]
        Turing.@addlogprob! sum(logpdf.(RatingScale.(eta), y))
    end
end

function RatingScale(eta; check_args = false)
    return PartialCredit(eta; check_args)
end
