function fit(
    M::Type{<:AbstractRaschModel{SamplingEstimate}},
    data,
    alg,
    args...;
    priors::Prior = Prior(),
    kwargs...,
)
    y, i, p = matrix_to_long(data)
    model = turing_model(M; priors)
    conditioned_model = model(y, i, p)
    chain = sample(conditioned_model, alg, args...; kwargs...)
    return M(data, priors, chain)
end

function fit(
    M::Type{<:AbstractRaschModel{PointEstimate}},
    data,
    alg::EstimationAlgorithm,
    alg_pp::PersonParameterAlgorithm = WLE(),
    args...;
    kwargs...,
)
    estimate = optimize(M, data, alg)
    return M(data, estimate; alg_pp)
end
