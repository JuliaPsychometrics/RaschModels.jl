function AbstractItemResponseModels.iif(model::RaschModel{<:EstimationType}, theta::Real, i, y::Real)
    checkresponsetype(response_type(model), y)
    beta = getindex(model.pars, i)
    return _iif(theta, beta, y)
end

_iif(theta, beta, y) = _irf(theta, beta, y) .* _irf(1 - theta, beta, y)
