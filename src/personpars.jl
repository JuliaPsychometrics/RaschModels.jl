abstract type PersonParameterAlgorithm end
"""
    PersonParameterWLE

Warm's weighted likelihood estimation for person parameters of Rasch models

# References

- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. Psychometrika, 54, 427-450. doi: 10.1007/BF02294627
"""
struct PersonParameterWLE <: PersonParameterAlgorithm end

"""
    PersonParameterMLE

Maximum likelihood estimation for person parameters of Rasch models
"""
struct PersonParameterMLE <: PersonParameterAlgorithm end

"""
    PersonParameterResult

A wrapper struct to store various results from a person parameter estimation.
"""
struct PersonParameterResult{PPA<:PersonParameterAlgorithm}
    "modeltype"
    modeltype::Type{<:AbstractRaschModel}
    "point estimates/coefs"
    values::AbstractVector
    "standard errors"
    se::AbstractVector
    "estimation algorithm"
    alg::PPA
end

function _fit_personpars(cmlresult::CMLResult, alg::PPA) where {PPA<:PersonParameterAlgorithm}
    (; modeltype, values) = cmlresult
    personpars, se = _fit_personpars_by_alg(alg, modeltype, values)
    return PersonParameterResult(modeltype, personpars, se, alg)
end

function _fit_personpars(modeltype::Type{RaschModel}, values::AbstractVector{<:AbstractFloat}, alg::PPA) where {PPA<:PersonParameterAlgorithm}
    personpars, se = _fit_personpars_by_alg(alg, modeltype, values)
    return PersonParameterResult(modeltype, personpars, se, alg)
end

function _fit_personpars_by_alg(
    ::PersonParameterWLE,
    modeltype::Type{RaschModel},
    betas::AbstractVector{T};
    I = length(betas),
) where {T<:AbstractFloat}
    personpars = zeros(T, I + 1)
    se = zeros(T, I + 1)
    init_x = zero(T)
    for r in eachindex(personpars)
        personpars[r] += find_zero(x -> _wle_ll(x, betas, r), init_x)
        se[r] += sqrt(1 / _information(modeltype, personpars[r], betas))
    end

    return personpars, se
end

function _wle_ll(
    theta::T,
    betas::AbstractVector{T},
    r::Int;
    irf::Function = (b) -> _irf(RaschModel, theta, b, 1),
    I = length(betas),
) where {T<:AbstractFloat}
    ll = zero(T)
    prop = zeros(T, I)
    info = ones(T, I)
    sum_prop = zero(T)
    sum_info = zero(T)
    ll_part = zero(T)

    for (i,v) in enumerate(betas)
        prop_i = irf(v)
        prop[i] += prop_i
        info[i] -= prop_i
        info[i] *= prop_i
        sum_prop += prop[i]
        sum_info += info[i]
        ll_part += info[i] * (1 - (2 * prop[i]))
    end

    ll += (r - 1)
    ll -= sum_prop - (ll_part / (2 * sum_info))

    return ll
end

function _fit_personpars_by_alg(
    ::PersonParameterMLE,
    modeltype::Type{RaschModel},
    betas::AbstractVector{T};
    I = length(betas),
) where {T<:AbstractFloat}
    personpars = Vector{Union{Nothing, T}}(nothing, I+1)
    se = Vector{Union{Nothing, T}}(nothing, I+1)

    personpars_est = zeros(T, I - 1)
    se_est = zeros(T, I - 1)
    init_x = zero(T)

    for r in eachindex(personpars_est)
        personpars_est[r] += find_zero(x -> _mle_ll(x, betas, r), init_x)
        se_est[r] += sqrt(1 / _information(modeltype, personpars_est[r], betas))
    end

    personpars[2:I] .= personpars_est
    se[2:I] .= se_est

    return personpars, se
end

function _mle_ll(
    theta::AbstractFloat,
    betas::AbstractVector{T},
    r::Int;
    irf::Function = (b) -> _irf(RaschModel, theta, b, 1) 
) where {T<:AbstractFloat}
    ll = zero(T)
    for b in betas 
        ll += irf(b)
    end
    ll = r - ll
    return ll
end

function _maptheta(rs::AbstractVector{Int}, thetas::AbstractVector{T}) where {T}
    personpars = Vector{T}(undef, length(rs))   
    for (i, v) in enumerate(rs)
        if isnothing(thetas[v+1])
            personpars[i] = nothing
        end
        personpars[i] = thetas[v+1] 
    end
    return personpars
end