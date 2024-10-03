"""
    $(TYPEDEF)

The `ESFAlgorithm` describes the algorithm of calculating elementary symmetric functions for
the model.
"""
abstract type ESFAlgorithm end

"""
    $(TYPEDEF)

Summation algorithm for calculating elementary symmetric functions and their derivatives
(Fischer, 1974).

# References

- Fischer, G. H. (1974). *Einfühung in die Theorie psychologischer Tests: Grundlagen und
  Anwendungen*. Huber.

"""
struct SummationAlgorithm <: ESFAlgorithm end

"""
    $(TYPEDEF)

A struct to store elementary symmetric functions up to their second derivative.
"""
mutable struct ESF{T<:AbstractFloat}
    γ0::Vector{T}
    γ1::Array{T,2}
    γ2::Array{T,3}
    I::Int
    R::Int
end

function ESF(I::Int; R::Int = I + 1)
    γ0 = zeros(Float64, R)
    γ1 = zeros(Float64, R, I)
    γ2 = zeros(Float64, R, I, I)
    return ESF(γ0, γ1, γ2, I, R)
end

function _esf!(
    ::SummationAlgorithm,
    γ0::AbstractVector{T},
    γtemp::AbstractVector{T},
    ϵ::AbstractVector{T},
    R::Int,
) where {T<:AbstractFloat}
    fill!(γ0, zero(T))
    γ0[1] += one(T)
    γ0[2] += ϵ[1]
    γtemp .= γ0

    for i in 3:R
        for j in 3:i
            γtemp[j] = (ϵ[i-1] * γ0[j-1]) + γ0[j]
        end
        γ0[3:i] = view(γtemp, 3:i)
        γ0[2] = ϵ[i-1] + γ0[2]
    end
    return nothing
end

function _esf0!(
    alg::SummationAlgorithm,
    esfstate::ESF,
    ϵ::AbstractVector{T},
) where {T<:AbstractFloat}
    (; γ0, R) = esfstate
    γtemp = copy(γ0)

    _esf!(alg, γ0, γtemp, ϵ, R)
    return nothing
end

function _esf1!(
    ::SummationAlgorithm,
    esfstate::ESF,
    ϵ::AbstractVector{T},
) where {T<:AbstractFloat}
    (; γ1, I) = esfstate

    γ0_i = zeros(T, I)
    γtemp_i = copy(γ0_i)

    fill!(γ1, zero(T))
    inv_ind = zeros(Int, I - 1)
    for i in 1:I
        if i == 1
            inv_ind .= 2:I
        elseif i == I
            inv_ind .= 1:(I-1)
        else
            inv_ind .= cat(1:(i-1), (i+1):I, dims = 1)
        end
        _esf!(SummationAlgorithm(), γ0_i, γtemp_i, ϵ[inv_ind], I)
        γ1[1:I, i] = γ0_i
    end
    return nothing
end

function _esf2!(
    ::SummationAlgorithm,
    esfstate::ESF,
    ϵ::AbstractVector{T},
) where {T<:AbstractFloat}
    (; γ1, γ2, I, R) = esfstate

    ϵ_times_ϵ = ϵ .* ϵ'
    γ0_ij = zeros(T, I - 1)
    γtemp_ij = copy(γ0_ij)

    fill!(γ2, zero(T))
    for i in 1:I-1
        for j in (i+1):I
            inv_ind = filter(n -> ((n != i) && (n != j)), 1:I)
            _esf!(SummationAlgorithm(), γ0_ij, γtemp_ij, ϵ[inv_ind], I - 1)
            γ2[2:I, j, i] = γ0_ij
            γ2[:, i, j] = γ2[:, j, i]
        end
    end

    for i in 1:R
        γ2[i, :, :] = ϵ_times_ϵ .* γ2[i, :, :]
    end

    for i in 1:I
        γ2[:, i, i] .= γ1[:, i]
    end
    return nothing
end

"""
    $(SIGNATURES)

Computation of elementary symmetric functions (ESFs) and their derivatives for dichotomous
responses up to a user-specified `order`.

Arguments:
- ϵ : vector of exp(-β)
- alg : algorithm for computing ESFs and their derivatives
- order : integer between 0 and 2; 0: ESFs only, 1: ESFs + first derivative, 2: ESFs + first
  and second derivative
"""
function esf(
    ϵ::AbstractVector{T},
    alg::ESFA;
    order::Int = 2,
    I = length(ϵ),
    R = I + 1,
) where {T<:AbstractFloat,ESFA<:ESFAlgorithm}
    # checks
    (order ≤ 2 || order ≥ 0) || throw(ArgumentError("Invalid order value $(order)"))
    I > 1 || throw(DomainError("item length must be over 1"))

    Γ = ESF(I; R = R)

    # esf
    order ≥ 0 && _esf0!(alg, Γ, ϵ)

    # 1st-order derivatives of esf
    order ≥ 1 && _esf1!(alg, Γ, ϵ)

    # 2nd-order derivatives of esf
    order == 2 && _esf2!(alg, Γ, ϵ)

    return Γ
end
