abstract type AbstractRaschModel <: ItemResponseModel end

person_dimensionality(::Type{<:AbstractRaschModel}) = AbstractItemResponseModels.Univariate
item_dimensionality(::Type{<:AbstractRaschModel}) = AbstractItemResponseModels.Univariate
