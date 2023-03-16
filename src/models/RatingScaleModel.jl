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

function getthresholds(model::RatingScaleModel{SamplingEstimate}, i, y)
    if y == 1
        nsamples, _, nchains = size(model.pars)
        return zeros(nsamples * nchains)
    end

    parname = model.parnames_tau[y-1]
    thresholds = vec(view(model.pars.value, var = parname))
    return thresholds
end

function getthresholds(
    model::RatingScaleModel{ET,DT,PT},
    i,
    y,
) where {ET,DT,PT<:StatisticalModel}
    if y == 1
        return 0.0
    else
        parname = model.parnames_tau[y-1]
        threshold = model.pars.values[parname]
        return threshold
    end
end

function _get_item_thresholds(model::RatingScaleModel{ET,DT,PT}, i) where {ET,DT,PT<:Chains}
    threshold_names = model.parnames_tau
    thresholds = view(model.pars.value, var = threshold_names)
    n_iter, n_pars, n_chains = size(thresholds)
    thresholds_permuted = permutedims(thresholds, (1, 3, 2))
    threshold_mat = Matrix(reshape(thresholds_permuted, n_iter * n_chains, n_pars))
    return threshold_mat
end

function _get_item_thresholds(
    model::RatingScaleModel{ET,DT,PT},
    i,
) where {ET,DT,PT<:StatisticalModel}
    pars = coef(model.pars)
    threshold_index = getindex.(pars.dicts, model.parnames_tau)
    thresholds = view(pars.array, threshold_index)
    return vec(thresholds)
end

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
