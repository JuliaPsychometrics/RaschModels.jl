"""
    fit(modeltype::Type{<: AbstractRaschModel}, data, alg, args...; kwargs...)

Fit a Rasch model to response `data`.

## Arguments
### `modeltype`
An `AbstractRaschModel` type.
Currently the package implements the following model types:

- `PartialCreditModel`
- `RaschModel`
- `RatingScaleModel`

### `data`
The observed data matrix.
It is assumed that each person corresponds to a row and each item to a column in the matrix.

### `alg`
The estimation algorithm.
For (bayesian) sampling based estimation all `Turing.InferenceAlgorithm` are supported.
The following algorithms are reexported from this package:

- `NUTS`: No-U-Turn-Sampler
- `HMC`: hamiltonian monte carlo
- `MH`: Metropolis-Hastings

Bayesian point estimation is supported by using either

- `MAP`: Maximum a posteriori (Posterior mode)
- `MLE`: Maximum Likelihood (Posterior mean)

## Examples
### Bayesian estimation
Fitting a simple Rasch model with the No-U-Turn-Sampler.
Note that estimation with `Turing.InterenceAlgorithm` uses the [AbstractMCMC.sample](https://turing.ml/library/AbstractMCMC/dev/api/#Sampling-a-single-chain) interface
and thus requires an additional argument to specify the number of iterations.

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
    return modeltype{SamplingEstimate,typeof(data),typeof(chain)}(data, chain)
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
    return modeltype{PointEstimate,typeof(data),typeof(estimate)}(data, estimate)
end
