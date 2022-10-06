function fit(T::Type{RaschModel}, data::AbstractMatrix; type::Symbol=:mcmc)
    if type == :optim
        return _fit_optim(T, data)
    elseif type == :mcmc
        return _fit_mcmc(T, data)
    else
        error("Unknown type")
    end
end

function _fit_optim(::Type{RaschModel}, data)
    nbetas = size(data, 2)
    betas = randn(nbetas)  # dummy
    return RaschModel{PointEstimate,typeof(data),typeof(betas)}(data, betas)
end

function _fit_mcmc(::Type{RaschModel}, data)
    nbetas = size(data, 2)
    nsamples = 100
    betas = [randn(nsamples) for _ in 1:nbetas]
    return RaschModel{SamplingEstimate,typeof(data),typeof(betas)}(data, betas)
end
