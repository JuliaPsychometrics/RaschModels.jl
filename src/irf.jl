function AbstractItemResponseModels.irf(model::RaschModel{PointEstimate}, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getindex(model.pars, i)
    return _irf(theta, beta, y)
end

function AbstractItemResponseModels.irf(model::RaschModel{SamplingEstimate}, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = vec(model.pars["beta[$i]"])
    return _irf(theta, beta, y)
end

function _irf(theta, beta, y)
    exp_linpred = exp.(theta .- beta)
    prob = @. exp_linpred / (1 + exp_linpred)
    return ifelse(y == 1, prob, 1 .- prob)
end

function AbstractItemResponseModels.expected_score(model::RaschModel{SamplingEstimate}, theta::Real, is)
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)
    for i in is
        score .+= irf(model, theta, i, 1)
    end
    return score
end

expected_score(model::RaschModel, theta::Real) = expected_score(model, theta, 1:size(model.data, 2))
