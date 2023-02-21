# convenience types and methods for handling missing data
MatrixWithMissings{T} = AbstractMatrix{Union{T,Missing}}

"""
    ResponsePatterns

stores for a given dataset the following information:
    - `patterns`: dict of boolean response patterns (false = missing response)
    - `pattern_idx`: a vector mapping each person to one response pattern in `patterns`
    - `n`: number of patterns.
"""
struct ResponsePatterns
    patterns::Dict{Int,Vector{Bool}}
    pattern_idx::Vector{Int}
    n::Int
end

function ResponsePatterns(
    data::AbstractMatrix;
    response_ind::AbstractMatrix{T} = isresponse.(data),
    P = size(data, 1),
) where {T<:Integer}
    unique_patterns::Vector{Vector{Bool}} = unique(eachrow(response_ind))
    pattern_idx = zeros(Int64, P)

    for (index_i, value_i) in enumerate(eachrow(response_ind))
        for (index_j, value_j) in enumerate(unique_patterns)
            if value_i == value_j
                pattern_idx[index_i] += index_j
            end
        end
    end

    list_patterns = Dict(i => v for (i, v) in enumerate(unique_patterns))
    return ResponsePatterns(list_patterns, pattern_idx, length(unique_patterns))
end

"""
    checkpatterns(data::AbstractMatrix; response_ind{<:Integer} = isresponse(data))

relevant check for conditional maximum likelihood estimation if response matrix `data` contains
    - items with only missing responses
    - subjects with less than two responses
"""
function checkpatterns(
    data::AbstractMatrix;
    response_ind::AbstractMatrix{T} = isresponse.(data),
) where {T<:Integer}
    n_responses_col = sum(response_ind, dims = 1)
    n_responses_row = sum(response_ind, dims = 2)

    if any(n -> n == 0, n_responses_col)
        throw(DomainError("Items with only missing responses are not permitted."))
    end

    if any(n -> n < 2, n_responses_row)
        throw(DomainError("Only subjects with at least two non-missing responses allowed."))
    end

    return nothing
end
