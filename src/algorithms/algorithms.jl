"""
    $(TYPEDEF)

An abstract type representing an estimation algorithm for Rasch models.
"""
abstract type EstimationAlgorithm end

"""
    estimate
"""
function estimate end

"""
    $(SIGNATURES)

normalize estimated values of a Rasch model from β[1]=0 to ∑β = 0.
"""
function sumzero!(
    values::AbstractVector,
    vcov::AbstractMatrix{<:Real};
    n_items = length(values),
)
    values[1] == 0 || throw(DomainError("first element of values must be zero."))
    values .-= (sum(values) / n_items)
    D = I(n_items) .- 1 / n_items
    vcov .= D * vcov * D'
    return nothing
end

include("ESF.jl")
include("CML.jl")
