function iif(model::RaschModel, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _iif(theta, beta, y)
end

_iif(theta, beta, y) = _irf(theta, beta, y) .* _irf(theta, beta, 1 - y)

function iif(model::RatingScaleModel, theta::Real, i, y::Real)
    category_prob = irf(model, theta, i, y)
    return category_prob .* (1 .- category_prob)
end

function information(model::RaschModel{SamplingEstimate}, theta::Real, is)
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)
    for i in is
        info .+= iif(model, theta, i, 1)
    end
    return info
end

function information(model::RaschModel{PointEstimate}, theta::Real, is)
    info = zero(Float64)
    for i in is
        info += iif(model, theta, i, 1)
    end
    return info
end

information(model::RaschModel, theta::Real) = information(model, theta, 1:size(model.data, 2))
