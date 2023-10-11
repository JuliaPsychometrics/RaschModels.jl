abstract type PersonParameterAlgorithm end
"""
    PersonParameterWLE

Warm's weighted likelihood estimation for person parameters of Rasch models

# References

- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. Psychometrika, 54, 427-450. doi: 10.1007/BF02294627
"""
struct PersonParameterWLE <: PersonParameterAlgorithm end

rational_bounds(::PersonParameterWLE) = true

"""
    PersonParameterMLE

Maximum likelihood estimation for person parameters of Rasch models
"""
struct PersonParameterMLE <: PersonParameterAlgorithm end

rational_bounds(::PersonParameterMLE) = false

"""
    PersonParameterResult

A wrapper struct to store various results from a person parameter estimation.
"""
struct PersonParameterResult{PPA<:PersonParameterAlgorithm} <: StatisticalModel
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
    return _fit_personpars(modeltype, values, alg) 
end

function _fit_personpars(modeltype::Type{RaschModel}, betas::AbstractVector{T}, alg::PPA; I = length(betas)) where {T<:AbstractFloat, PPA<:PersonParameterAlgorithm}
    personpars = zeros(T, I + 1)
    se = zeros(T, I + 1)
    init_x = zero(T)

    for r in eachindex(personpars)
        # if estimation of rational bounds is not possible 
        if (r == 1 || r == I+1) && !rational_bounds(alg)
            personpars[r] += NaN
            se[r] += NaN
            continue
        end
        personpars[r] += find_zero(x -> optfun(alg, modeltype, x, betas, r), init_x)
        se[r] += √(var(alg, modeltype, personpars[r], betas))
    end

    return PersonParameterResult(modeltype, personpars, se, alg)
end

function optfun(
    ::PersonParameterWLE,
    modeltype::Type{RaschModel},
    theta::T,
    betas::AbstractVector{T},
    r::Int
) where {T<:AbstractFloat}
    sum_prop = zero(T)
    info = zero(T)
    sum_deriv = zero(T)

    for beta in betas
        prop_i = _irf(modeltype, theta, beta, 1)
        sum_prop += prop_i

        info_i = (1 - prop_i) * prop_i
        info += info_i

        sum_deriv += info_i * (1 - (2 * prop_i))
    end

    return (r - 1) - (sum_prop - (sum_deriv / (2 * info)))
end

function optfun(
    ::PersonParameterMLE,
    modeltype::Type{RaschModel},
    theta::T,
    betas::AbstractVector{T},
    r::Int
) where {T<:AbstractFloat}
    optvalue = zero(T)
    for beta in betas 
        optvalue += _irf(modeltype, theta, beta, 1)
    end
    return (r - 1) - optvalue
end

function var(
    ::PersonParameterWLE,
    modeltype::Type{RaschModel},
    theta::T,
    betas::AbstractVector{T},
) where {T<:AbstractFloat}
    # variance equal (asymptotically) to variance of MLE (Warm, 1989) 
    return var(PersonParameterMLE(), modeltype, theta, betas)
end

function var(
    ::PersonParameterMLE,
    modeltype::Type{RaschModel},
    theta::T,
    betas::AbstractVector{T},
) where {T<:AbstractFloat}
    info = zero(T)

    for beta in betas
        info += _iif(modeltype, theta, beta)
    end

    return 1 / info
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