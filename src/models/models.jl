abstract type AbstractRaschModel <: ItemResponseModel end

person_dimensionality(::Type{<:AbstractRaschModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:AbstractRaschModel}) = AbstractItemResponseModels.Univariate

include("RaschModel.jl")
include("PolytomousRaschModel.jl")
include("PartialCreditModel.jl")
include("RatingScaleModel.jl")
