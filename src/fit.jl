"""
    fit(modeltype::Type{<: AbstractRaschModel}, data, alg, args...; kwargs...)

Fit a Rasch model to response `data`.

## Arguments
### `modeltype`
An `AbstractRaschModel` type.
Currently RaschModels.jl implements the following model types:

- `PartialCreditModel`
- `RaschModel`
- `RatingScaleModel`

### `data`
The observed data matrix.
It is assumed that each person corresponds to a row and each item to a column in the matrix.

For dichotomous models correct responses are coded as `1` and incorrect responses are coded
as `0`.

For polytomous models, categories are coded as an integer sequence starting from `1`.
If your model features e.g. three categories, they are coded as `1`, `2` and `3` respectively.

### `alg`
The estimation algorithm.

For (bayesian) sampling based estimation all `Turing.InferenceAlgorithm` types are supported.
The following algorithms are reexported from this package:

- `NUTS`: No-U-Turn-Sampler
- `HMC`: Hamiltonian Monte Carlo
- `MH`: Metropolis-Hastings

Bayesian point estimation is supported by

- `MAP`: Maximum a posteriori (Posterior mode)
- `MLE`: Maximum Likelihood (Posterior mean)

## Examples
### Bayesian estimation
Fitting a simple Rasch model with the No-U-Turn-Sampler.
Note that estimation with `Turing.InterenceAlgorithm` uses the
[AbstractMCMC.sample](https://turing.ml/library/AbstractMCMC/dev/api/#Sampling-a-single-chain)
interface and thus requires an additional argument to specify the number of iterations.

```julia
X = rand(0:1, 100, 10)
rasch_fit = fit(RaschModel, X, NUTS(), 1_000)
```

You can also sample multiple chains in parallel.

```julia
X = rand(0:1, 100, 10)
rasch_fit = fit(RaschModel, X, NUTS(), MCMCThreads(), 1_000, 4)
```

For bayesian point estimation you can just swap the inference algorithm to one of the supported
point estimation algorithms.

```julia
X = rand(0:1, 100, 10)
rasch_fit = fit(RaschModel, X, MAP())
```
"""
function fit(
    modeltype::Type{<:AbstractRaschModel},
    data::AbstractMatrix,
    alg,
    args...;
    kwargs...
)
    _fit(modeltype, data, alg, args...; kwargs...)
end

function _fit(
    modeltype::Type{<:AbstractRaschModel},
    data::AbstractMatrix,
    alg::Turing.InferenceAlgorithm,
    args...;
    priors::Prior=Prior()
)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(modeltype), y)

    Turing.setadbackend(:reversediff)
    Turing.setrdcache(true)

    model = turing_model(modeltype; priors)
    chain = sample(model(y, i, p), alg, args...)
    parnames_beta = [Symbol("beta[", i, "]") for i in 1:size(data, 2)]
    return modeltype{SamplingEstimate,typeof(data),typeof(chain)}(data, chain, parnames_beta)
end

function _fit(
    modeltype::Type{<:AbstractRaschModel},
    data::AbstractMatrix,
    alg::Union{Turing.MLE,Turing.MAP},
    args...;
    priors::Prior=Prior()
)
    y, i, p = matrix_to_long(data)
    checkresponsetype(response_type(modeltype), y)

    model = turing_model(modeltype; priors)
    estimate = optimize(model(y, i, p), alg, args...)

    # test for new raschmodel
    parnames_beta = [Symbol("beta[", i, "]") for i in 1:size(data, 2)]
    return modeltype{PointEstimate,typeof(data),typeof(estimate)}(data, estimate, parnames_beta)
end
