fit(T::Type{<:AbstractRaschModel}, data::AbstractMatrix, alg, args...; kwargs...) = _fit(T, data, alg, args...; kwargs...)

function _fit(T::Type{<:AbstractRaschModel}, data::AbstractMatrix, alg::Turing.InferenceAlgorithm, args...; kwargs...)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    Turing.setadbackend(:reversediff)
    Turing.setrdcache(true)

    model = turing_model(T, y, i, p)
    chain = sample(model, alg, args...)
    return T{SamplingEstimate,typeof(data),typeof(chain)}(data, chain)
end

function _fit(T::Type{<:AbstractRaschModel}, data::AbstractMatrix, alg::Union{Turing.MLE,Turing.MAP}, args...; kwargs...)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)
    model = turing_model(T, y, i, p)
    estimate = optimize(model, alg, args...)
    return T{PointEstimate,typeof(data),typeof(estimate)}(data, estimate)
end

turing_model(::Type{RaschModel}, args...) = rasch(args...)
turing_model(::Type{RatingScaleModel}, args...) = ratingscale(args...)
turing_model(::Type{PartialCreditModel}, args...) = partialcredit(args...)

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
