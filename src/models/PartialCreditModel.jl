struct PartialCreditModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
end

response_type(::Type{<:PartialCreditModel}) = AbstractItemResponseModels.Ordinal
person_dimensionality(::Type{<:PartialCreditModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:PartialCreditModel}) = AbstractItemResponseModels.Univariate
estimation_type(::Type{<:PartialCreditModel{ET,DT,PT}}) where {ET,DT,PT} = ET

"""
    getitempars
"""
function getitempars(model::PartialCreditModel{ET,DT,PT}, i) where {ET,DT,PT<:Chains}
    parnames = string.(namesingroup(model.pars, :beta))
    beta_names = filter(x -> occursin("beta[$i]", x), parnames)
    betas = getindex(model.pars, beta_names)
    return Array(betas)
end

function getitempars(model::PartialCreditModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    pars = coef(model.pars)
    parnames = string.(params(model.pars))
    beta_names = filter(x -> occursin("beta[$i]", x), parnames)
    betas = getindex(pars, Symbol.(beta_names))
    return vec(betas)
end

"""
    irf
"""
function irf(model::PartialCreditModel{ET,DT,PT}, theta::Real, i, y::Real) where {ET<:SamplingEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    eta = theta .- beta
    p = probs.(PartialCredit.(eachrow(eta)))
    return getindex.(p, Int(y))
end

function irf(model::PartialCreditModel{ET,DT,PT}, theta::Real, i, y::Real) where {ET<:PointEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    eta = theta .- beta
    p = probs(PartialCredit(eta))
    return getindex(p, Int(y))
end

"""
    iif
"""

"""
    expected_score
"""

"""
    information
"""

# Turing implementation
@model function partialcredit(y, i, p, ::Type{T}=Float64) where {T}
    I = maximum(i)
    P = maximum(p)
    K = [maximum(y[i.==item]) - 1 for item in 1:I]

    theta ~ filldist(Normal(), P)
    mu_beta ~ Normal()
    sigma_beta ~ InverseGamma(3, 2)

    beta = Vector{T}.(undef, K)

    for item in eachindex(beta)
        beta[item] ~ filldist(Normal(mu_beta, sigma_beta), K[item])
    end

    eta = [theta[p[n]] .- beta[i[n]] for n in eachindex(y)]
    Turing.@addlogprob! sum(logpdf.(PartialCredit.(eta), y))
end

function PartialCredit(eta::AbstractVector{<:Real}; check_args=false)
    extended = vcat(0.0, eta)
    probs = softmax(cumsum(extended))
    return Categorical(probs; check_args)
end
