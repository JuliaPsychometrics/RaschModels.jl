function AbstractItemResponseModels.irf(model::RaschModel{PointEstimate}, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getindex(model.pars, i)
    return _irf(theta, beta, y)
end

function AbstractItemResponseModels.irf(model::RaschModel{SamplingEstimate}, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getindex(model.pars, i)
    return _irf(theta, beta, y)
end

function _irf(theta, beta, y)
    exp_linpred = exp.(theta .- beta)
    prob = @. exp_linpred / (1 + exp_linpred)
    return ifelse(y == 1, prob, 1 .- prob)
end
