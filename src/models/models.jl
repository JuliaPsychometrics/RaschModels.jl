"""
    $(TYPEDEF)

An abstract type representing a Rasch model.

All implementations `T <: AbstractRaschModel{<:EstimationType}` must implement the following
traits:

- [`model_type`](@ref): The model type for the calculation of [`irf`](@ref), [`iif`](@ref), [`expected_score`](@ref), and [`information`](@ref)
- [`response_type`](@ref): The allowed types of item responses
"""
abstract type AbstractRaschModel{T<:EstimationType} <: ItemResponseModel end

estimation_type(::Type{<:AbstractRaschModel{T}}) where {T} = T
person_dimensionality(::Type{<:AbstractRaschModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:AbstractRaschModel}) = AbstractItemResponseModels.Univariate

abstract type RaschModel{T} <: AbstractRaschModel{T} end
abstract type PartialCreditModel{T} <: AbstractRaschModel{T} end
abstract type RatingScaleModel{T} <: AbstractRaschModel{T} end

"""
    $(SIGNATURES)

The model type for a Rasch model. This function is used to determine the response and
information functions for the calculations in [`irf`](@ref), [`iif`](@ref), [`expected_score`](@ref)
and [`information`](@ref).
"""
function model_type end

function model_type(m::AbstractRaschModel)
    return model_type(typeof(m))
end

function expected_score(
    model::AbstractRaschModel,
    theta;
    scoring_function::F = identity,
) where {F}
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items; scoring_function)
    return score
end

function information(
    model::AbstractRaschModel,
    theta;
    scoring_function::F = identity,
) where {F}
    items = 1:size(model.data, 2)
    info = information(model, theta, items; scoring_function)
    return info
end

include("bayesian/bayesian.jl")
include("frequentist/frequentist.jl")
