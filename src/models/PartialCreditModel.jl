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
function getitempars(model::PartialCreditModel{ET,DT,PT}, i)::Matrix{Float64} where {ET,DT,PT<:Chains}
    parnames = string.(namesingroup(model.pars, :beta))
    beta_names = Symbol.(filter(x -> occursin("beta[$i]", x), parnames))
    betas = Array(model.pars[beta_names])
    return betas
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
function irf(model::PartialCreditModel{ET,DT,PT}, theta, i) where {ET<:SamplingEstimate,DT,PT}
    betas = getitempars(model, i)
    categories = 1:size(betas, 2)
    probs = _irf.(PartialCreditModel, theta, eachrow(betas), Ref(categories))
    return probs
end

function irf(model::PartialCreditModel{ET,DT,PT}, theta, i, y) where {ET<:SamplingEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    probs = irf(model, theta, i)
    return getindex.(probs, Int(y))
end

function irf(model::PartialCreditModel{ET,DT,PT}, theta, i) where {ET<:PointEstimate,DT,PT}
    betas = getitempars(model, i)
    categories = 1:length(betas)
    probs = _irf(PartialCreditModel, theta, betas, categories)
    return probs
end

function irf(model::PartialCreditModel{ET,DT,PT}, theta, i, y) where {ET<:PointEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    probs = irf(model, theta, i)
    return getindex(probs, Int(y))
end

function _irf(::Type{PartialCreditModel}, theta, betas, y)
    extended = vcat(zero(eltype(betas)), betas)
    cumsum!(extended, extended)
    softmax!(extended, extended)
    return extended
end

"""
    iif
"""
function iif(model::PartialCreditModel{SamplingEstimate}, theta, i)
    category_probs = irf(model, theta, i)
    score = expected_score(model, theta, i)
    return _iif.(PartialCreditModel, category_probs, score)
end

function iif(model::PartialCreditModel{SamplingEstimate}, theta, i, y)
    checkresponsetype(response_type(model), y)
    category_prob = irf(model, theta, i, y)
    item_information = iif(model, theta, i)
    return category_prob ./ item_information
end

function iif(model::PartialCreditModel{PointEstimate}, theta, i)
    category_probs = irf(model, theta, i)
    score = expected_score(model, theta, i)
    return _iif(PartialCreditModel, category_probs, score)
end

function iif(model::PartialCreditModel{PointEstimate}, theta, i, y)
    checkresponsetype(response_type(model), y)
    category_prob = irf(model, theta, i, y)
    item_information = iif(model, theta, i)
    return category_prob / item_information
end

function _iif(::Type{PartialCreditModel}, probs, score)
    info = zero(Float64)
    for (category, prob) in enumerate(probs)
        info += (category - score)^2 * prob
    end
    return info
end

"""
    expected_score
"""
function expected_score(model::PartialCreditModel{SamplingEstimate}, theta, is)
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)
    for i in is
        category_probs = irf(model, theta, i)
        categories = 1:length(first(category_probs))
        category_scores = [probs .* categories for probs in category_probs]
        score .+= sum.(category_scores)
    end
    return score
end

function expected_score(model::PartialCreditModel, theta)
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items)
    return score
end

function expected_score(model::PartialCreditModel{PointEstimate}, theta, is)
    score = zero(Float64)
    for i in is
        category_probs = irf(model, theta, i)
        categories = 1:length(category_probs)
        score += sum(category_probs .* categories)
    end
    return score
end

"""
    information
"""
function information(model::PartialCreditModel{SamplingEstimate}, theta, is)
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)
    for i in is
        info += iif(model, theta, i)
    end
    return info
end

function information(model::PartialCreditModel{PointEstimate}, theta, is)
    info = zero(Float64)
    for i in is
        info += iif(model, theta, i)
    end
    return info
end

function information(model::PartialCreditModel, theta)
    items = 1:size(model.data, 2)
    info = information(model, theta, items)
    return info
end

# Turing implementation
function _turing_model(::Type{PartialCreditModel}; priors)
    @model function partialcredit(y, i, p, ::Type{T}=Float64; priors=priors) where {T}
        I = maximum(i)
        P = maximum(p)
        K = [maximum(y[i.==item]) - 1 for item in 1:I]

        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta

        beta = Vector{T}.(undef, K)

        for item in eachindex(beta)
            beta[item] ~ filldist(mu_beta + priors.beta_norm * sigma_beta, K[item])
        end

        eta = [theta[p[n]] .- beta[i[n]] for n in eachindex(y)]
        Turing.@addlogprob! sum(logpdf.(PartialCredit.(eta), y))
    end
end

function PartialCredit(eta::AbstractVector{<:Real}; check_args=false)
    extended = vcat(0.0, eta)
    probs = softmax(cumsum(extended))
    return Categorical(probs; check_args)
end
