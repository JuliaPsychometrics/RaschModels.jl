"""
    ESF{T<:AbstractFloat}

Basic struct to store elementary symmetric functions up to their second derivative. 
"""
mutable struct ESF{T<:AbstractFloat}
    γ0::Vector{T}
    γ1::UN{Array{T, 2}}
    γ2::UN{Array{T, 3}}
end

function ESF(I::Int; R::Int = I+1)
    γ0 = zeros(Float64, R)
    γ1 = zeros(Float64, R, I)
    γ2 = zeros(Float64, R, I, I)
    return ESF(γ0, γ1, γ2)
end

"""
    ESFAlgorithm

The `ESFAlgorithm` describes the algorithm of calculating elementary symmetric functions for the model.
""" 
abstract type ESFAlgorithm end

"""
    SummationAlgorithm <: ESFAlgorithm

Summation algorithm for calculating elementary symmetric functions and their derivatives (Fischer, 1974).
        
# References
        
- Fischer, G. H. (1974). *Einfühung in die Theorie psychologischer Tests: Grundlagen und Anwendungen*. Huber.
        
"""
struct SummationAlgorithm <: ESFAlgorithm end

# elementary symmetric function of order 0
function _esf!(::SummationAlgorithm, ϵ::Vector{T}, γ0::Vector{T}, R::Int) where {T<:AbstractFloat}
    fill!(γ0, zero(T))
    γ0[1] += one(T)
    γ0[2] += ϵ[1]

    γ_temp = copy(γ0)
    @inbounds for i in 3:R
        for j in 3:i
            γ_temp[j] = (ϵ[i-1] * γ0[j-1]) + γ0[j]
        end
        γ0[3:i] = γ_temp[3:i]
        γ0[2] = ϵ[i-1] + γ0[2]
    end
    return nothing
end

# elementary symmetric function of order 1
function _esf!(::SummationAlgorithm, ϵ::Vector{T}, γ0::Vector{T}, γ1::Array{T,2}, R::Int) where {T<:AbstractFloat}
    I = R-1
    γ0_temp = zeros(T, I)

    fill!(γ1, zero(T))
    inv_ind = zeros(T, I-1)
    @inbounds for i in 1:I
        fill!(γ0_temp, zero(T))
        # looks ugly, but was the fastest way (less allocations)
        if i == 1
            inv_ind = 2:I
        elseif i == I
            inv_ind = 1:(I-1)
        else
            inv_ind = cat(1:(i-1), (i+1):I, dims=1)
        end
        _esf!(SummationAlgorithm(), ϵ[inv_ind], γ0_temp, I)
        γ1[1:I, i] = γ0_temp
    end
    return nothing
end

# elementary symmetric function of order 2
function _esf!(::SummationAlgorithm, ϵ::Vector{T}, γ0::Vector{T}, γ1::Array{T, 2}, γ2::Array{T, 3}, R::Int) where {T<:AbstractFloat}
    I = R-1
    ϵ_times_ϵ = ϵ .* ϵ'
    γ_temp = zeros(T, I-1)

    fill!(γ2, zero(T))
    @inbounds for i in 1:I-1
        for j in (i+1):I
            fill!(γ_temp, zero(T))
            inv_ind = filter(n -> ((n != i) && (n != j)), 1:I)
            _esf!(SummationAlgorithm(), ϵ[inv_ind], γ_temp, I-1)
            γ2[2:I, j, i] = γ_temp
            γ2[:, i, j] = γ2[:, j, i]
        end
    end

    @inbounds for i in 1:R
        γ2[i, :, :] = ϵ_times_ϵ .* γ2[i, :, :]
    end

	@inbounds for i in 1:I
        γ2[:, i, i] .= γ1[:, i]
    end
    return nothing
end

"""
    esf(ϵ::Vector{<:AbstractFloat}, alg::{<:ESFAlgorithm}; order::Int)

Computation of elementary symmetric functions (ESFs) and their derivatives for dichotomous responses up to a user-specified `order`.

Arguments:
- ϵ : vector of exp(-β) 
- alg : algorithm for computing ESFs and their derivatives
- order : integer between 0 and 2; 0: ESFs only, 1: ESFs + first derivative, 2: ESFs + first and second derivative
"""
function esf(ϵ::Vector{T}, alg::ESFA; order::Int = 2) where {T<:AbstractFloat, ESFA<:ESFAlgorithm}
    I = length(ϵ)
    R = I + 1 # number of raw scores

    # checks
    (order ≤ 2 || order ≥ 0) || throw(ArgumentError("Invalid order value $(order)"))
    I > 1 || throw(DomainError("item length must be over 1"))

    Γ = ESF(zeros(T, R), nothing, nothing)

    # esf
    if order ≥ 0
        _esf!(alg, ϵ, Γ.γ0, R)
    end

    # 1st-order derivatives of esf
    if order ≥ 1
        Γ.γ1 = zeros(T, R, I)
        _esf!(alg, ϵ, Γ.γ0, Γ.γ1, R)
    end

    # 2nd-order derivatives of esf
    if order == 2
        Γ.γ2 = zeros(T, R, I, I)
        _esf!(alg, ϵ, Γ.γ0, Γ.γ1, Γ.γ2, R)
    end

    return Γ
end