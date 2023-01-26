

"""
    matrix_to_long(m::AbstractMatrix; dropmissing=true)

Convert a matrix `m` to long format responses.
Returns 3 vectors of equal length corresponding to the responses, item ids and person ids.

By default missing values are treated as missing at random and dropped from the response vectors.
"""
function matrix_to_long(m::AbstractMatrix; dropmissing=true)
    N = dropmissing ? sum(!ismissing, m) : prod(size(m))

    response_vec = construct_response_vector(m, N; dropmissing)
    item_vec = Vector{Int}(undef, N)
    person_vec = Vector{Int}(undef, N)

    n = 0
    for (j, col) in enumerate(eachcol(m))
        for (i, y) in enumerate(col)
            dropmissing && ismissing(y) && continue
            n += 1

            response_vec[n] = y
            item_vec[n] = j
            person_vec[n] = i
        end
    end

    return response_vec, item_vec, person_vec
end

construct_response_vector(m::AbstractMatrix{T}, N; dropmissing) where {T} = Vector{T}(undef, N)
function construct_response_vector(m::AbstractMatrix{Union{Missing,T}}, N; dropmissing) where {T}
    type = dropmissing ? T : Union{Missing,T}
    return Vector{type}(undef, N)
end
