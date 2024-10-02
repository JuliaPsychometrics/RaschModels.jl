const Fittable = Union{Type{RaschModel},Type{PartialCreditModel},Type{RatingScaleModel}}

function fit(
    M::Fittable,
    data,
    alg::InferenceAlgorithm,
    args...;
    priors::Prior = Prior(),
    kwargs...,
)
    # map the abstract type to a concrete implementation
    T = map_model_type(M, :bayesian)

    y, i, p = matrix_to_long(data)
    model = turing_model(T; priors)
    conditioned_model = model(y, i, p)
    chain = sample(conditioned_model, alg, args...; kwargs...)

    return T(data, priors, chain)
end

function fit(
    M::Fittable,
    data,
    alg::EstimationAlgorithm,
    alg_pp::PersonParameterAlgorithm = WLE(),
    args...;
    kwargs...,
)
    T = map_model_type(M, :frequentist)
    estimate = optimize(T, data, alg)
    return T(data, estimate; alg_pp)
end

function map_model_type(M, type)
    if type == :bayesian
        T = if M == RaschModel
            BayesianRaschModel
        elseif M == PartialCreditModel
            BayesianPartialCreditModel
        elseif M == RatingScaleModel
            BayesianRatingScaleModel
        end
    elseif type == :frequentist
        T = if M == RaschModel
            FrequentistRaschModel
        elseif M == PartialCreditModel
            FrequentistPartialCreditModel
        elseif M == RatingScaleModel
            FrequentistRatingScaleModel
        end
    end

    return T
end
