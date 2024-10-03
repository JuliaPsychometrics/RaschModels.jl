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

function check_response_patterns(data::AbstractMatrix)
    if any_missing_patterns(data)
        err = "Items with only missing responses are not permitted"
        throw(DomainError(err))
    end

    if any_subject_responses_less_than(data, 2)
        err = "Only subjects with at least two nonmissing responses are permitted"
        throw(DomainError(err))
    end

    return nothing
end

"""
    $(SIGNATURES)

Check if any column of the matrix `data` has all missing values.
"""
function any_missing_patterns(data::AbstractMatrix)
    for col in eachcol(data)
        all(ismissing, col) && return true
    end

    return false
end

"""
    $(SIGNATURES)

Check if any row of the matrix `data` has less than `n` nonmissing values.
"""
function any_subject_responses_less_than(data::AbstractMatrix, n)
    for row in eachrow(data)
        sum(!ismissing, row) < n && return true
    end

    return false
end
