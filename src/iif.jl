function AbstractItemResponseModels.iif(model::RaschModel{PointEstimate}, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getindex(model.pars, i)
    return _iif(theta, beta, y)
end

function AbstractItemResponseModels.iif(model::RaschModel{SamplingEstimate}, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = vec(model.pars["beta[$i]"])
    return _iif(theta, beta, y)
end

_iif(theta, beta, y) = _irf(theta, beta, y) .* _irf(1 - theta, beta, y)

function AbstractItemResponseModels.information(model::RaschModel{SamplingEstimate}, theta::Real, is)
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)
    for i in is
        info .+= iif(model, theta, i, 1)
    end
    return info
end

information(model::RaschModel, theta::Real) = information(model, theta, 1:size(model.data, 2))
