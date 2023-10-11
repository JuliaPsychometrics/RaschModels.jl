"""
    RaschModel

## Fields
- `data`: The data matrix used to fit the model
- `pars`: The structure holding parameter values
- `parnames_beta`: A vector of parameter names for item parameters
"""
mutable struct RaschModel{ET<:EstimationType,DT<:AbstractMatrix,PT} <: AbstractRaschModel
    data::DT
    pars::PT
    parnames_beta::Vector{Symbol}
end

response_type(::Type{<:RaschModel}) = AbstractItemResponseModels.Dichotomous
estimation_type(::Type{<:RaschModel{ET,DT,PT}}) where {ET,DT,PT} = ET

"""
    getitempars(model::RaschModel, i)

Fetch the item parameters of `model` for item `i`.
"""
function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:Chains}
    parname = model.parnames_beta[i]
    betas = vec(view(model.pars.value, var = parname))
    return betas
end

function getitempars(model::RaschModel{ET,DT,PT}, i) where {ET,DT,PT<:StatisticalModel}
    parname = model.parnames_beta[i]
    betas = coef(model.pars)
    return getindex(betas, parname)
end

function getitempars(
    model::RaschModel{ET,DT,PT},
    i,
) where {ET,DT,PT<:CombinedStatisticalModel}
    parname = model.parnames_beta[i]
    betas = coef(model.pars.itemresult)
    return getindex(betas, parname)
end

"""
    getpersonpars(model::RaschModel, p)

Fetch the person parameters of `model` for person `p`.
"""
function getpersonpars(model::RaschModel{ET,DT,PT}, p) where {ET,DT,PT<:Chains}
    parname = Symbol("theta[", p, "]")
    thetas = vec(view(model.pars.value, var = parname))
    return thetas
end

function getpersonpars(model::RaschModel{ET,DT,PT}, p) where {ET,DT,PT<:StatisticalModel}
    parname = Symbol("theta[", p, "]")
    thetas = coef(model.pars)
    return getindex(thetas, parname)
end

@doc raw"""
    irf(model::RaschModel, theta, i, y)
    irf(model::RaschModel, theta, i)

Evaluate the item response function for a dichotomous Rasch model for item `i` at the ability
value `theta`.

If the response value `y` is omitted, the item response probability for a correct response
`y = 1` is returned.

## Examples
### Point estimation
```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> irf(rasch, 0.0, 1)
0.3989070983504997
```

### Bayesian estimation
```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MH(), 100; progress = false);

julia> irf(rasch, 0.0, 1);
```
"""
function irf(model::RaschModel{SamplingEstimate}, theta, i, y = 1)
    n_iter = length(getitempars(model, i))
    probs = zeros(Float64, n_iter)
    add_irf!(model, probs, theta, i, y, scoring_function = one)
    return probs
end

function irf(model::RaschModel{PointEstimate}, theta, i, y = 1)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _irf(RaschModel, theta, beta, y)
end

function add_irf!(
    model::RaschModel{SamplingEstimate},
    probs,
    theta,
    i,
    y;
    scoring_function::F = identity,
) where {F}
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)

    for j in eachindex(beta)
        probs[j] += _irf(RaschModel, theta, beta[j], y) * scoring_function(y)
    end

    return nothing
end

function _irf(::Type{RaschModel}, theta, beta, y)
    prob = logistic(theta - beta)
    return ifelse(y == 1, prob, 1 - prob)
end

@doc raw"""
    iif(model::RaschModel, theta, i, y)
    iif(model::RaschModel, theta, i)

Evaluate the item information function for a dichotomous Rasch model for item `i` at the
ability value `theta`.

If the response value `y` is omitted, the item information for a correct response `y = 1` is
returned.

## Examples
### Point estimation
```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> iif(rasch, 0.0, 1)
0.07287100351275909
```

### Bayesian estimation
```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MH(), 100; progress = false);

julia> iif(rasch, 0.0, 1);
```
"""
function iif(model::RaschModel{SamplingEstimate}, theta, i, y = 1)
    n_iter = length(getitempars(model, i))
    info = zeros(Float64, n_iter)
    add_iif!(model, info, theta, i, y)
    return info
end

function iif(model::RaschModel{PointEstimate}, theta, i, y = 1)
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)
    return _iif(RaschModel, theta, beta)
end

function add_iif!(
    model::RaschModel{SamplingEstimate},
    info,
    theta,
    i,
    y;
    scoring_function::F = identity,
) where {F}
    checkresponsetype(response_type(model), y)
    beta = getitempars(model, i)

    for j in eachindex(beta)
        info[j] += _iif(RaschModel, theta, beta[j]; scoring_function)
    end

    return nothing
end

function _iif(::Type{RaschModel}, theta, beta; scoring_function::F = identity) where {F}
    expected = _irf(RaschModel, theta, beta, 1) * scoring_function(1)
    info = zero(Float64)

    for y in 0:1
        prob = _irf(RaschModel, theta, beta, y)
        info += (scoring_function(y) - expected)^2 * prob
    end

    return info
end

@doc raw"""
    expected_score(model::RaschModel, theta, is; scoring_function)
    expected_score(model::RaschModel, theta; scoring_function)

Calculate the expected score for a dichotomous Rasch model at ability value `theta` for a
set of items `is`.

`is` can either be a single item index, an array of item indices, or a range of values.
If `is` is omitted, the expected score for the whole test is calculated.

`scoring_function` can be used to add weights to the resulting expected scores (see Examples
for details).

## Examples
### Point estimation
```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> expected_score(rasch, 0.0)  # all 3 items
0.4435592483649984

julia> expected_score(rasch, 0.0, 1:2)  # items 1 and 2
0.3054954277378461

julia> expected_score(rasch, 0.0, [1, 3])  # items 1 and 3
0.21719686244768088
```

### Bayesian estimation
```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MH(), 100; progress = false);

julia> expected_score(rasch, 0.0);  # all 3 items

julia> expected_score(rasch, 0.0, 1:2);  # items 1 and 2

julia> expected_score(rasch, 0.0, [1, 3]);  # items 1 and 3
```

### Using the scoring function
Using the `scoring_function` keyword argument allows to weigh response probabilities by
a value depending on the response `y`. It is of the form `f(y) = x`, assigning a scalar
value to every possible reponse value `y`.

For the Rasch Model the valid responses are 0 and 1. If we want to calculate expected scores
doubling the weight for `y = 1`, the weighted responses are 0 and 2.
The corresponding `scoring_function` is `y -> 2y`,

```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> expected_score(rasch, 0.0; scoring_function = y -> 2y)
4.592642952772852
```
"""
function expected_score(
    model::RaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    niter = size(model.pars, 1)
    score = zeros(Float64, niter)

    for i in is
        for y in 0:1
            add_irf!(model, score, theta, i, y; scoring_function)
        end
    end

    return score
end

function expected_score(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    score = zero(Float64)
    for i in is
        for y in 0:1
            score += irf(model, theta, i, y) * scoring_function(y)
        end
    end
    return score
end

function expected_score(model::RaschModel, theta; scoring_function::F = identity) where {F}
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items; scoring_function)
    return score
end

@doc raw"""
    information(model::RaschModel, theta, is; scoring_function)
    information(model::RaschModel, theta; scoring_function)

Calculate the information for a dichotomous Rasch model at the ability value `theta` for a
set of items `is`.

`is` can either be a single item index, an array of item indices, or a range of values.
If `is` is omitted, the information for the whole test is calculated.

`scoring_function` can be used to add weights to the resulting information (see Examples
for details).

## Examples
### Point estimation
```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> information(rasch, 0.0)  # all 3 items
0.519893299555712

julia> information(rasch, 0.0, 1:2)  # items 1 and 2
0.4121629113381827

julia> information(rasch, 0.0, [1, 3])  # items 1 and 3
0.313811843802304
```

### Bayesian estimation
```jldoctest
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MH(), 100; progress = false);

julia> information(rasch, 0.0);  # all 3 items

julia> information(rasch, 0.0, 1:2);  # items 1 and 2

julia> information(rasch, 0.0, [1, 3]);  # items 1 and 3
```

### Using the scoring function
Using the `scoring_function` keyword argument allows to weigh response probabilities by
a value depending on the response `y`. It is of the form `f(y) = x`, assigning a scalar
value to every possible reponse value `y`.

For the Rasch Model the valid responses are 0 and 1. If we want to calculate the information
doubling the weight for `y = 1`, the weighted responses are 0 and 2.
The corresponding `scoring_function` is `y -> 2y`,

```jldoctest; filter = r"[0-9\.]+"
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> information(rasch, 0.0; scoring_function = y -> 2y)
2.079573198222848
```
"""
function information(
    model::RaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    niter = size(model.pars, 1)
    info = zeros(Float64, niter)

    for i in is
        add_iif!(model, info, theta, i, 1; scoring_function)
    end

    return info
end

function information(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    info = zero(Float64)
    for i in is
        beta = getitempars(model, i)
        info += _iif(RaschModel, theta, beta; scoring_function)
    end
    return info
end

function information(model::RaschModel, theta; scoring_function::F = identity) where {F}
    items = 1:size(model.data, 2)
    info = information(model, theta, items; scoring_function)
    return info
end

# Turing implementation
function _turing_model(::Type{RaschModel}; priors)
    @model function rasch_model(y, i, p; I = maximum(i), P = maximum(p), priors = priors)
        theta ~ filldist(priors.theta, P)
        mu_beta ~ priors.mu_beta
        sigma_beta ~ priors.sigma_beta
        beta ~ filldist(mu_beta + priors.beta_norm * sigma_beta, I)

        eta = theta[p] .- beta[i]
        Turing.@addlogprob! sum(logpdf.(Rasch.(eta), y))
    end
end

function Rasch(eta::Real)
    return BernoulliLogit(eta)
end
