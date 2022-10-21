function irf(model::RaschModel, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _irf(theta, beta, y)
end

function _irf(theta, beta, y)
    exp_linpred = exp.(theta .- beta)
    prob = @. exp_linpred / (1 + exp_linpred)
    return ifelse(y == 1, prob, 1 .- prob)
end

function irf(model::RatingScaleModel{ET,DT,PT}, theta::Real, i, y::Real) where {ET<:SamplingEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    beta, tau = getitempars(model, i)
    eta = theta .- (beta .+ tau)
    p = probs.(PartialCredit.(eachrow(eta)))
    return getindex.(p, Int(y))
end

function irf(model::RatingScaleModel{ET,DT,PT}, theta::Real, i, y::Real) where {ET<:PointEstimate,DT,PT}
    checkresponsetype(response_type(model), y)
    beta, tau = getitempars(model, i)
    eta = theta .- (beta .+ tau)
    p = probs(PartialCredit(eta))
    return getindex(p, Int(y))
end

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

function expected_score(model::RaschModel{SamplingEstimate}, theta::Real, is)
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)
    for i in is
        score .+= irf(model, theta, i, 1)
    end
    return score
end

function expected_score(model::RaschModel{PointEstimate}, theta::Real, is)
    score = zero(Float64)
    for i in is
        score += irf(model, theta, i, 1)
    end
    return score
end

expected_score(model::RaschModel, theta::Real) = expected_score(model, theta, 1:size(model.data, 2))
