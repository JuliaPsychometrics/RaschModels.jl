isresponse(x) = !ismissing(x)

"""
    matrix_to_long(m::AbstractMatrix; dropmissing=true)

Convert a matrix `m` to long format responses.
Returns 3 vectors of equal length corresponding to the responses, item ids and person ids.

By default missing values are treated as missing at random and dropped from the response vectors.
"""
function matrix_to_long(m::AbstractMatrix; dropmissing = true)
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

function construct_response_vector(m::AbstractMatrix{T}, N; dropmissing) where {T}
    return Vector{T}(undef, N)
end
function construct_response_vector(
    m::AbstractMatrix{Union{Missing,T}},
    N;
    dropmissing,
) where {T}
    type = dropmissing ? T : Union{Missing,T}
    return Vector{type}(undef, N)
end

"""
    betanames(n)

Construct a vector of parameter names for item difficulties/locations.
"""
function betanames(n)
    return [Symbol("beta[", i, "]") for i in 1:n]
end

"""
    thetanames(n)

Construct a vector of parameter names for person parameters.
"""
function thetanames(n)
    return [Symbol("theta[", p, "]") for p in 1:n]
end

"""
    taunames(n; item=nothing)

Construct a vector of parameter names for item thresholds.
"""
function taunames(n; item = nothing)
    item_str = isnothing(item) ? "" : string("[", item, "]")
    return [Symbol("tau", item_str, "[", i, "]") for i in 1:n]
end

"""
    gettotals(s::AbstractVector{T}, minvalue::T, maxvalue::T) where {T<:Int}

Helper function to obtain totals for a marginal score vector `s` (row or column scores) from `minvalue` to `maxvalue`.
"""
function gettotals(s::AbstractVector{T}, minvalue::T, maxvalue::T) where {T<:Int}
    totals = zeros(T, maxvalue + 1)
    for i in s
        (i < minvalue || i > maxvalue) && continue
        totals[i+1] += 1
    end
    return totals[(minvalue+1):end]
end

"""
    normalize_sumzero!(values::AbstractVector, vcov::AbstractMatrix{T}; I::Int = length(values)) where {T<:AbstractFloat}

normalize estimated values of a Rasch model from β[1]=0 to ∑β = 0
"""
function normalize_sumzero!(
    values::AbstractVector,
    vcov::AbstractMatrix{T};
    I::Int = length(values),
) where {T<:AbstractFloat}
    values[1] == 0 || throw(DomainError("first element of values must be zero."))
    values .-= (sum(values) / I)
    D = LinearAlgebra.I(I) .- 1 / I
    vcov .= D * vcov * D'
    return nothing
end

function getrowsums(data; P = size(data, 1), I = size(data, 2))
    rs = zeros(Int, P)

    for p in eachindex(rs)
        for i in 1:I
            response = data[p, i]
            if isresponse(response)
                rs[p] += response
            end
        end
    end

    return rs
end

function getcolsums(data; P = size(data, 1), I = size(data, 2))
    cs = zeros(Int, I)

    for i in eachindex(cs)
        for p in 1:P
            response = data[p, i]
            if isresponse(response)
                cs[i] += response
            end
        end
    end

    return cs
end
