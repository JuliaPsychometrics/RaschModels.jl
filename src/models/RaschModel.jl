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
    getitemlocations(model::RaschModel, i)

Fetch the item difficulty parameter of `model` for item `i`.
"""
function getitemlocations(
    model::RaschModel{PointEstimate,DT,PT},
    i,
    y = nothing,
) where {DT,PT<:StatisticalModel}
    parname = model.parnames_beta[i]
    betas = coef(model.pars)
    return getindex(betas, parname)
end

function getitemlocations(
    model::RaschModel{PointEstimate,DT,PT},
    i,
    y = nothing,
) where {DT,PT<:CombinedStatisticalModel}
    parname = model.parnames_beta[i]
    betas = coef(model.pars.itemresult)
    return getindex(betas, parname)
end

function getitemlocations(model::RaschModel{SamplingEstimate}, i, y = nothing)
    parname = model.parnames_beta[i]
    betas = reshape(view(model.pars.value, var = parname), :, length(i))
    return betas
end

"""
    getpersonlocations(model::RaschModel, p)

Fetch the person ability parameter of `model` for person `p`.
"""
function getpersonlocations(
    model::RaschModel{PointEstimate,DT,PT},
    p,
) where {DT,PT<:StatisticalModel}
    parname = Symbol("theta[", p, "]")
    thetas = coef(model.pars)
    return getindex(thetas, parname)
end

function getpersonlocations(model::RaschModel{SamplingEstimate}, p)
    parname = Symbol("theta[", p, "]")
    thetas = vec(view(model.pars.value, var = parname))
    return thetas
end

"""
    irf(model::RaschModel, theta, i, y)
    irf(model::RaschModel, theta, i)

Evaluate the item response function for a dichotomous Rasch model for item `i` at the ability
value `theta`.

If the response value `y` is omitted, the item response probability for a correct response
`y = 1` is returned.

## Examples
### Point estimation
```jldoctest
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> irf(rasch, 0.0, 1)
0.3989070983504997
```

### Bayesian estimation
```jldoctest
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MH(), 100; progress = false);

julia> irf(rasch, 0.0, 1)
```
"""
function irf(model::RaschModel{PointEstimate}, theta, i, y = 1)
    checkresponsetype(response_type(model), y)
    beta = getitemlocations(model, i)
    return irf(ItemResponseFunctions.OnePL, theta, (; b = beta), y)
end

function irf(model::RaschModel{SamplingEstimate}, theta, i, y = 1)
    betas = getitemlocations(model, i)

    probs = map(betas) do beta
        return irf(ItemResponseFunctions.OnePL, theta, (; b = beta), y)
    end

    return probs
end

"""
    iif(model::RaschModel, theta, i, y)
    iif(model::RaschModel, theta, i)

Evaluate the item information function for a dichotomous Rasch model for item `i` at the
ability value `theta`.

If the response value `y` is omitted, the item information for a correct response `y = 1` is
returned.

## Examples
### Point estimation
```jldoctest
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> iif(rasch, 0.0, 1)
0.07287100351275909
```

### Bayesian estimation
```jldoctest
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MH(), 100; progress = false);

julia> iif(rasch, 0.0, 1);
```
"""
function iif(model::RaschModel{PointEstimate}, theta, i, y = 1)
    checkresponsetype(response_type(model), y)
    beta = getitemlocations(model, i)
    return iif(ItemResponseFunctions.OnePL, theta, (; b = beta), y)
end

function iif(model::RaschModel{SamplingEstimate}, theta, i, y = 1)
    betas = getitemlocations(model, i)

    info = map(betas) do beta
        return iif(ItemResponseFunctions.OnePL, theta, (; b = beta), y)
    end

    return info
end

"""
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
```jldoctest
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
```jldoctest
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

```jldoctest
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> expected_score(rasch, 0.0; scoring_function = y -> 2y)
4.592642952772852
```
"""
function expected_score(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    betas = getitemlocations(model, is)
    score = expected_score(ItemResponseFunctions.OnePL, theta, betas; scoring_function)
    return score
end

function expected_score(
    model::RaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function::F = identity,
) where {F}
    betas = getitemlocations(model, is)

    scores = map(eachrow(betas)) do beta
        return expected_score(ItemResponseFunctions.OnePL, theta, beta; scoring_function)
    end

    return scores
end

function expected_score(model::RaschModel, theta; scoring_function::F = identity) where {F}
    items = 1:size(model.data, 2)
    score = expected_score(model, theta, items; scoring_function)
    return score
end

"""
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
```jldoctest
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

```jldoctest
julia> data = rand(0:1, 50, 3);

julia> rasch = fit(RaschModel, data, MLE());

julia> information(rasch, 0.0; scoring_function = y -> 2y)
2.079573198222848
```
"""
function information(
    model::RaschModel{PointEstimate},
    theta,
    is;
    scoring_function::F = one,
) where {F}
    betas = getitemlocations(model, is)
    score = information(ItemResponseFunctions.OnePL, theta, betas; scoring_function)
    return score
end

function information(
    model::RaschModel{SamplingEstimate},
    theta,
    is;
    scoring_function::F = one,
) where {F}
    betas = getitemlocations(model, is)

    scores = map(eachrow(betas)) do beta
        return information(ItemResponseFunctions.OnePL, theta, beta; scoring_function)
    end

    return scores
end

function information(model::RaschModel, theta; scoring_function::F = identity) where {F}
    items = 1:size(model.data, 2)
    score = information(model, theta, items; scoring_function)
    return score
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
