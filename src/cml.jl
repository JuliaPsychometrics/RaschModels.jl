"""
    CML

Conditional maximum likelihood estimation

Note: As of now, only estimation of a dichotomous Rasch model is supported.

Arguments:

- esf_alg::{<:ESFAlgorithm} : The algorithm for calculating elementary symmetric functions,
  default = SummationAlgorithm()
- start : Vector of starting values for β (item difficulties); default = nothing
- normalize::Bool : Sum-zero normalization of estimates and vcov, default = true
"""
Base.@kwdef struct CML{ESFA<:ESFAlgorithm}
    esf_alg::ESFA = SummationAlgorithm()
    start::UN{Vector{AbstractFloat}} = nothing
    normalize::Bool = true
end

"""
    CMLResult{
        V<:NamedArrays.NamedArray,
        O<:Optim.MultivariateOptimizationResults,
    }

A wrapper struct to store various results from a CML estimation.
"""
struct CMLResult{V<:NamedArrays.NamedArray,O<:Optim.MultivariateOptimizationResults} <:
       StatisticalModel
    "modeltype"
    modeltype::Type{<:AbstractRaschModel}
    "point estimates/coefs"
    values::V
    "Optim.jl results"
    optim_result::O
    "conditional log likelihood (maximized lp)"
    lp::Float64
    "degrees of freedom"
    df::Int
    "elementary symmetric functions"
    esf::Union{ESF,Dict{Int,<:ESF}}
    "variance-covariance matrix"
    vcov::AbstractMatrix{Float64}
    "cml settings"
    settings::CML
end

#####################
# StatsAPI methods  #
#####################
function coeftable(m::CMLResult)
    # Get columns for coeftable.
    terms = String.(coefnames(m))
    estimates = m.values.array
    stderrors = stderror(m)
    tstats = estimates ./ stderrors

    return CoefTable(
        [estimates, stderrors, tstats],
        ["estimate", "stderror", "tstat"],
        terms,
    )
end

coef(m::CMLResult) = m.values
coefnames(m::CMLResult) = names(m.values)[1]
params(m::CMLResult) = coefnames(m)
informationmatrix(m::CMLResult) = m.vcov
vcov(m::CMLResult) = informationmatrix(m)
loglikelihood(m::CMLResult) = m.lp


function _fit_by_cml(
    modeltype::Type{RaschModel},
    data::AbstractMatrix{T},
    alg::CML,
    args...;
    kwargs...,
) where {T<:Integer}
    P, I = size(data)
    checkcondition(data; P = P, I = I)
    cs = getcolsums(data)

    Γ = ESF(I)
    H = zeros(Float64, I - 1, I - 1)

    # optimization
    β0 = _set_startvalues(modeltype, cs, I, P, alg)
    neglogLC, g!, h! = optfuns(modeltype, data, Γ, alg.esf_alg)
    estimate =
        Optim.optimize(β -> neglogLC(β, cs[2:I]), (G, β) -> g!(G, β), β0, BFGS(), args...)

    # estimate vcov
    h!(H)
    vcov = cat(0, inv(H), dims = (1, 2))

    # handle varibles for output
    values =
        NamedArrays.NamedArray(estimate.minimizer, betanames(I))
    ## sum-zero normalization (∑β = 0), else β[1] = 0
    alg.normalize && normalize_sumzero!(values, vcov; I = I)

    return CMLResult(modeltype, values, estimate, -estimate.minimum, I - 1, Γ, vcov, alg)
end

function _fit_by_cml(
    modeltype::Type{RaschModel},
    data::MatrixWithMissings{T},
    alg::CML,
    args...;
    kwargs...,
) where {T<:Integer}
    response_ind = isresponse.(data)

    checkcondition(data; B = response_ind)
    checkpatterns(data; response_ind)

    I = size(data, 2)
    rp = ResponsePatterns(data; response_ind)

    rs = getrowsums(data)
    cs = getcolsums(data)

    I_split = Dict(i => sum(v) for (i, v) in rp.patterns)
    esf_split = Dict(i => ESF(v) for (i, v) in I_split)

    # inits for optimization
    β0 = _set_startvalues(modeltype, cs, I, vec(sum(response_ind, dims = 1)), alg)
    H = zeros(Float64, I - 1, I - 1)
    vcov = copy(H)

    # optimization
    neglogLC, g!, h! = optfuns(modeltype, data, I_split, cs, rs, rp, esf_split, alg.esf_alg)
    estimate = Optim.optimize(β -> neglogLC(β), (G, β) -> g!(G, β), β0, BFGS(), args...)

    # estimate vcov
    h!(H)
    vcov = cat(0, inv(H), dims = (1, 2))

    # handle variables for output
    values =
        NamedArrays.NamedArray(estimate.minimizer, betanames(I))
    ## sum-zero normalization (∑β = 0), else β[1] = 0
    alg.normalize && normalize_sumzero!(values, vcov; I = I)

    return CMLResult(
        modeltype,
        values,
        estimate,
        -estimate.minimum,
        I - 1,
        esf_split,
        vcov,
        alg,
    )
end

# optimization functions
function optfuns(
    ::Type{RaschModel},
    data::AbstractMatrix{<:Integer},
    esfstate::ESF,
    esf_alg::ESFA;
    I::Int = size(data, 2),
    R::Int = I + 1,
) where {ESFA<:ESFAlgorithm}
    (; γ0, γ1, γ2) = esfstate

    rs = getrowsums(data)
    rf = gettotals(rs, 0, I)[2:end]

    last_β = fill(NaN, I)
    ϵ = ones(Float64, I)

    calculate_common! = function (β, last_β)
        if β != last_β
            copyto!(last_β, β)
            @. ϵ = exp(-β)
            _esf0!(esf_alg, esfstate, ϵ)
        end
        return nothing
    end

    neglogLC = function (β, cs)
        calculate_common!(β, last_β)
        @views cll = -cs'log.(ϵ[2:I]) + rf'log.(γ0[2:R])
        if !isfinite(cll)
            cll = -prevfloat(Inf)
        end
        return cll
    end

    g! = function (G, β)
        calculate_common!(β, last_β)
        _esf1!(esf_alg, esfstate, ϵ)
        @. γ1 = exp(log(γ1)' - β)'
        rs_ind = rs .!= 0
        @views G_temp =
            (.-data[rs_ind, :].+γ1[rs[rs_ind], :]./γ0[rs[rs_ind].+1])[:, 2:I]
        G[1] = 0
        G[2:I] = .-sum(G_temp, dims = 1)
        return nothing
    end

    h! = function (H)
        _esf2!(esf_alg, esfstate, ϵ)
        @views g0 = γ0[2:R]
        @views g1 = γ1[1:I, 2:I]
        @views g2 = γ2[1:I, 2:I, 2:I]
        g1divg0 = g1 ./ g0

        for i in 1:(I-1)
            @views H[i, :] = rf' * ((g2[:, i, :] ./ g0) .- (g1[:, i] ./ g0) .* g1divg0)
        end
        return nothing
    end

    return neglogLC, g!, h!
end

function optfuns(
    ::Type{RaschModel},
    data::MatrixWithMissings{<:Integer},
    I_split::Dict{Int,Int},
    cs::Vector{Int},
    rs::Vector{Int},
    rp::ResponsePatterns,
    esfstate_split::Dict{Int,<:ESF},
    esf_alg::ESFA;
    I::Int = size(data, 2),
) where {ESFA<:ESFAlgorithm}
    (; patterns, pattern_idx) = rp

    # split by response patterns
    cs_split =
        Dict(i => getcolsums(data[pattern_idx.==i, v]) for (i, v) in patterns)
    rs_split = Dict(i => rs[pattern_idx.==i] for i in keys(patterns))
    rf_split = Dict(i => gettotals(v, 0, I_split[i])[2:end] for (i, v) in rs_split)
    ϵ_split = Dict(i => ones(Float64, v) for (i, v) in I_split)

    last_β = fill(NaN, I)
    ϵ = ones(Float64, I)

    G_temp = zeros(Float64, I)
    H_temp = zeros(Float64, I - 1, I - 1)

    calculate_common! = function (β, last_β)
        if β != last_β
            copyto!(last_β, β)
            @. ϵ = exp(-β)

            for (i, v) in patterns
                ϵ_split[i] .= ϵ[v]
                _esf0!(esf_alg, esfstate_split[i], ϵ_split[i])
            end
        end
        return nothing
    end

    neglogLC = function (β)
        calculate_common!(β, last_β)
        cll = 0.0

        for (i, v) in I_split
            @views cll +=
                -cs_split[i]'log.(ϵ_split[i]) +
                rf_split[i]'log.(esfstate_split[i].γ0[2:(v+1)])
        end

        if !isfinite(cll)
            cll = -prevfloat(Inf)
        end
        return cll
    end

    g! = function (G, β)
        calculate_common!(β, last_β)
        fill!(G_temp, zero(Float64))

        for (i, v) in patterns
            idx_i = pattern_idx .== i
            esfstate_i = esfstate_split[i]
            rs_i = rs_split[i]
            rs_ind = rs_i .!= 0

            _esf1!(esf_alg, esfstate_i, ϵ_split[i])
            @. esfstate_i.γ1 = exp(log(esfstate_i.γ1)' - β[v])'

            @views G_i =
                .-data[idx_i, v][rs_ind, :] .+
                esfstate_i.γ1[rs_i[rs_ind], :] ./ esfstate_i.γ0[rs_i[rs_ind].+1]
            G_temp[v] .-= vec(sum(G_i, dims = 1))
        end

        G[1] = 0.0
        G[2:I] = G_temp[2:I]
        return nothing
    end

    h! = function (H)
        H_temp = fill!(H_temp, zero(Float64))

        for (i, v) in patterns
            esfstate_i = esfstate_split[i]

            I_i = esfstate_i.I
            R_i = esfstate_i.R
            rf_i = rf_split[i]

            v_est = v[2:end]
            sum_v_est = sum(v_est)
            iter_est = sum_v_est == I_i ? (1:I_i) : (2:I_i)

            @views H_temp_i = H_temp[v_est, v_est]

            _esf2!(esf_alg, esfstate_i, ϵ_split[i])

            @views g0 = esfstate_i.γ0[2:R_i]
            @views g1 = esfstate_i.γ1[1:I_i, iter_est]
            @views g2 = esfstate_i.γ2[1:I_i, iter_est, iter_est]
            g1divg0 = g1 ./ g0

            for i in 1:sum_v_est
                @views H_temp_i[i, :] .+= vec(
                    sum(
                        rf_i' * ((g2[:, i, :] ./ g0) .- (g1[:, i] ./ g0) .* g1divg0);
                        dims = 1,
                    ),
                )
            end
        end

        H .= H_temp
        return nothing
    end

    return neglogLC, g!, h!
end

# functions for setting starting values for β
function _set_startvalues(
    modeltype::Type{RaschModel},
    cs::Vector{Int},
    I::Int,
    P::Union{Int,Vector{Int}},
    alg::CML,
)
    start = zeros(Float64, I)

    if isnothing(alg.start)
        start = _get_startvalues(modeltype, cs, I, P)
    else
        length(alg.start) == I || throw(
            DimensionMismatch("vector of starting values must be the same length of items"),
        )
    end

    if start[1] != 0
        start = _set_constraint(modeltype, start, I)
    end

    return start
end

function _get_startvalues(::Type{RaschModel}, cs::Vector{Int}, I, P::Int)
    start = zeros(Float64, I)
    for (index, value) in enumerate(cs)
        start[index] = log(P - value) - log(value)
    end
    return start
end

function _get_startvalues(::Type{RaschModel}, cs::Vector{Int}, I, P::Vector{Int})
    start = zeros(Float64, I)
    for (index, value) in enumerate(cs)
        start[index] = log(P[index] - value) - log(value)
    end
    return start
end

function _set_constraint(::Type{RaschModel}, start::Vector{T}, I) where {T<:AbstractFloat}
    # constraint: β[1] = 0
    for i in 2:I
        start[i] -= start[1]
    end
    start[1] = 0.0
    return start
end

# checks specific to CML estimation
"""
    checkcondition(A::AbstractMatrix{<:Integer})
    checkcondition(A::MatrixWithMissings{<:Integer})

Check if a response matrix A is well-conditioned according to Fischer (1981). If no missing
values are present in response matrix A, the check is based on sufficient marginal sums
(Equation 12/Lemma 4 in Fischer, 1981).
On the other hand, if missing values exists, the check is using a graph-theoretic approach
(Lemma 6 and 7 in Fischer, 1981).

# References
- Fischer, G.H. (1981). On the existence and uniqueness of maximum-likelihood estimates in
the Rasch model. *Psychometrika, 46*, 59--76.

"""
function checkcondition(
    A::AbstractMatrix{T};
    P::Int = size(A, 1),
    I::Int = size(A, 2),
) where {T<:Integer}
    rs = zeros(Int, P)
    cs_ordered = zeros(Int, I)

    cs_ordered .+= getcolsums(A)
    sort!(cs_ordered)

    rs .+= sum(A; dims = 2)
    filter!(x -> x != 0 && x != I, rs)
    nr = gettotals(rs, 1, I - 1)

    return _checkcondition(nr, cs_ordered, I)
end

function _checkcondition(nr::AbstractVector{Int}, cs::AbstractVector{Int}, I::Int)
    sum_cs = zero(Int)

    # Equation 12 in Fischer (1981)
    for i in 1:(I-1)
        sum_cs += cs[i]
        sum_lside = zero(Int)
        sum_rside = zero(Int)

        for j in (I-i):(I-1)
            sum_lside += nr[j] * j
            sum_rside += nr[j]
        end

        sum_rside *= I - i
        sum_rside += sum_cs

        (sum_lside == sum_rside) &&
            throw(DomainError("response matrix is not well-conditioned"))

    end
    return nothing
end

function checkcondition(
    A::MatrixWithMissings{T};
    B::AbstractMatrix{<:Integer} = isresponse.(A),
) where {T<:Integer}
    size(A) == size(B) || throw(DimensionMismatch("input matrices must match in size"))
    P, I = size(A)
    C = zeros(Int, I, I)
    tmp = zero(Int)

    for i in 1:I, j in 1:I
        tmp = 0
        for p in 1:P
            mul_B = B[p, i] * B[p, j]
            mul_B == 0 && continue
            tmp += A[p, i] * (1 - A[p, j]) * mul_B
        end
        C[i, j] += tmp > 0
    end

    C += LinearAlgebra.I

    any(i -> i ≤ 0, C^(I - 1)) &&
        throw(DomainError("response matrix is not well-conditioned"))
    return nothing
end
