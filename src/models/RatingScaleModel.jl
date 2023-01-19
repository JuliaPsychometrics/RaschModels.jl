"""
    RatingScaleModel

A type representing a Rating Scale Model
"""
struct RatingScaleModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: PolytomousRaschModel{ET,PT}
    data::DT
    pars::PT
end

function _get_item_thresholds(model::RatingScaleModel{ET,DT,PT}, i)::Matrix{Float64} where {ET,DT,PT<:Chains}
    threshold_names = namesingroup(model.pars, :tau)
    thresholds = Array(model.pars[threshold_names])
    return thresholds
end

function _get_item_thresholds(model::RatingScaleModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    parnames = string.(params(model.pars))
    pars = coef(model.pars)
    threshold_names = filter(x -> occursin("tau", x), parnames)
    thresholds = getindex(pars, Symbol.(threshold_names))
    return vec(thresholds)
end

function _turing_model(::Type{RatingScaleModel}; priors)
    @model function rating_scale_model(y, i, p; I=maximum(i), P=maximum(p), K=maximum(y) - 1, priors=priors)
        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta
        beta ~ filldist(mu_beta + priors.beta_norm * sigma_beta, I)
        tau ~ filldist(priors.tau, K)

        eta = [theta[p[n]] .- (beta[i[n]] .+ tau) for n in eachindex(y)]
        Turing.@addlogprob! sum(logpdf.(RatingScale.(eta), y))
    end
end

function RatingScale(eta; check_args=false)
    return PartialCredit(eta; check_args)
end
