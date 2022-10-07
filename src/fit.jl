fit(T::Type{RaschModel}, data::AbstractMatrix, alg; kwargs...) = _fit(T, data, alg; kwargs...)

function _fit(T::Type{RaschModel}, data::AbstractMatrix, alg::Turing.InferenceAlgorithm; kwargs...)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    Turing.setadbackend(:reversediff)
    Turing.setrdcache(true)

    model = rasch(y, i, p)
    chain = sample(model, alg, 1000)
    return RaschModel{SamplingEstimate,typeof(data),typeof(chain)}(data, chain)
end

@model function rasch(y, i, p; I=maximum(i), P=maximum(p))
    theta ~ filldist(Normal(), P)
    mu_beta ~ Normal()
    sigma_beta ~ InverseGamma(3, 2)
    beta ~ filldist(Normal(mu_beta, sigma_beta), I)
    Turing.@addlogprob! sum(logpdf.(BernoulliLogit.(theta[p] .- beta[i]), y))
end
