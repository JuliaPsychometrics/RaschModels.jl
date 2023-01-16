"""
    fit()

Fit a Rasch model to response `data`.

"""
function fit(T::Type{<:AbstractRaschModel}, data::AbstractMatrix, alg, args...; kwargs...)
    _fit(T, data, alg, args...; kwargs...)
end

function _fit(T::Type{<:AbstractRaschModel}, data::AbstractMatrix, alg::Turing.InferenceAlgorithm, args...; priors::Prior=Prior())
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    Turing.setadbackend(:reversediff)
    Turing.setrdcache(true)

    model = turing_model(T; priors)
    chain = sample(model(y, i, p), alg, args...)
    return T{SamplingEstimate,typeof(data),typeof(chain)}(data, chain)
end

function _fit(T::Type{<:AbstractRaschModel}, data::AbstractMatrix, alg::Union{Turing.MLE,Turing.MAP}, args...; priors::Prior=Prior())
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(T), y)

    model = turing_model(T; priors)
    estimate = optimize(model(y, i, p), alg, args...)
    return T{PointEstimate,typeof(data),typeof(estimate)}(data, estimate)
end
