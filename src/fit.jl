fit(T::Type{<:AbstractRaschModel}, data::AbstractMatrix, alg, args...; kwargs...) = _fit(T, data, alg, args...; kwargs...)

function _fit(T::Type{RaschModel}, data::AbstractMatrix, alg::Turing.InferenceAlgorithm, args...; kwargs...)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    Turing.setadbackend(:reversediff)
    Turing.setrdcache(true)

    model = rasch(y, i, p)
    chain = sample(model, alg, args...)
    return RaschModel{SamplingEstimate,typeof(data),typeof(chain)}(data, chain)
end

function _fit(T::Type{RaschModel}, data::AbstractMatrix, alg::Union{Turing.MLE,Turing.MAP}, args...; kwargs...)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    model = rasch(y, i, p)
    estimate = optimize(model, alg, args...)
    return RaschModel{PointEstimate,typeof(data),typeof(estimate)}(data, estimate)
end

function _fit(T::Type{RatingScaleModel}, data::AbstractMatrix, alg::Turing.InferenceAlgorithm, args...; kargs...)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    model = ratingscale(y, i, p)
    chain = sample(model, alg, args...)
    return RatingScaleModel{SamplingEstimate,typeof(data),typeof(chain)}(data, chain)
end

function _fit(T::Type{RatingScaleModel}, data::AbstractMatrix, alg::Union{Turing.MLE,Turing.MAP}, args...; kwargs...)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    model = ratingscale(y, i, p)
    estimate = optimize(model, alg, args...)
    return RatingScaleModel{PointEstimate,typeof(data),typeof(estimate)}(data, estimate)
end

@model function rasch(y, i, p; I=maximum(i), P=maximum(p))
    theta ~ filldist(Normal(), P)
    mu_beta ~ Normal()
    sigma_beta ~ InverseGamma(3, 2)
    beta ~ filldist(Normal(mu_beta, sigma_beta), I)
    Turing.@addlogprob! sum(logpdf.(BernoulliLogit.(theta[p] .- beta[i]), y))
end

@model function ratingscale(y, i, p)
    I = maximum(i)
    P = maximum(p)
    K = maximum(y) - 1

    theta ~ filldist(Normal(), P)
    mu_beta ~ Normal()
    sigma_beta ~ InverseGamma(3, 2)
    beta ~ filldist(Normal(mu_beta, sigma_beta), I)
    tau ~ filldist(Normal(), K)

    eta = [theta[p[n]] .- (beta[i[n]] .+ tau) for n in eachindex(y)]
    Turing.@addlogprob! sum(logpdf.(PartialCredit.(eta), y))
end

function PartialCredit(eta::AbstractVector{<:Real}; check_args=false)
    extended = vcat(0.0, eta)
    probs = softmax(cumsum(extended))
    return Categorical(probs; check_args)
end
