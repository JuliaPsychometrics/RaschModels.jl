"""
    PartialCreditModel <: PolytomousRaschModel

A type representing a Partial Credit Model.
"""
struct PartialCreditModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <:
       PolytomousRaschModel{ET,PT}
    data::DT
    pars::PT
    parnames_beta::Vector{Symbol}
    parnames_tau::Vector{Vector{Symbol}}
end

function _get_item_thresholds(
    model::PartialCreditModel{ET,DT,PT},
    i,
)::Matrix{Float64} where {ET,DT,PT<:Chains}
    parnames = string.(namesingroup(model.pars, :tau))
    threshold_names = Symbol.(filter(x -> occursin("tau[$i]", x), parnames))
    thresholds = Array(model.pars[threshold_names])
    return thresholds
end

function _get_item_thresholds(
    model::PartialCreditModel{ET,DT,PT},
    i,
) where {ET,DT,PT<:StatisticalModel}
    parnames = string.(params(model.pars))
    pars = coef(model.pars)
    threshold_names = filter(x -> occursin("tau[$i]", x), parnames)
    thresholds = getindex(pars, Symbol.(threshold_names))
    return vec(thresholds)
end

function _turing_model(::Type{PartialCreditModel}; priors)
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

        eta = [theta[p[n]] .- (beta[i[n]] .+ tau[i[n]]) for n in eachindex(y)]

        Turing.@addlogprob! sum(logpdf.(PartialCredit.(eta), y))
    end
end

function PartialCredit(eta::AbstractVector{<:Real}; check_args = false)
    extended = vcat(0.0, eta)
    probs = softmax(cumsum(extended))
    return Categorical(probs; check_args)
end
