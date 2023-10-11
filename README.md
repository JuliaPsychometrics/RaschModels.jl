# RaschModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliapsychometrics.github.io/RaschModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliapsychometrics.github.io/RaschModels.jl/dev/)
[![Build Status](https://github.com/JuliaPsychometrics/RaschModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaPsychometrics/RaschModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaPsychometrics/RaschModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaPsychometrics/RaschModels.jl)

RaschModels.jl is a Julia package for fitting and evaluating Rasch Models. It implements
the basic Rasch Model, Partial Credit Model, and Rating Scale Model, as well as their 
linear extensions. 

**Note:** Currently only a subset of models is available. Please see [Roadmap](#roadmap)
for details.

## Installation
To install this package you can use Julias package management system.

```julia
] add RaschModels
```

## Getting started
Fitting a model using RaschModels.jl is easy. First, get some response data as a `Matrix`. 
In this example we just use some random data for 100 persons and 5 items.

```julia
data = rand(0:1, 100, 5)
```

Using `data` as our response data we can fit a Rasch Model. 

```julia
rasch = fit(RaschModel, data, CML())
```

This function call fits the model using conditional Maximum Likelihood estimation. 
To fit the Rasch Model using Bayesian estimation just change the algorithm and provide the 
required additional arguments.

```julia
rasch_bayes = fit(RaschModel, data, NUTS(), 1_000)
```

Additional plotting capabilities are provided by [ItemResponsePlots.jl](https://github.com/JuliaPsychometrics/ItemResponsePlots.jl).

## Roadmap 
RaschModels.jl is still under active development. Therefore, not all functionality is 
available yet. This roadmap provides a quick overview of the current state of the package.

### Existing features
- Fitting Rasch Models (CML estimation, Bayesian estimation)
- Fitting Rating Scale Models (Bayesian estimation)
- Fitting Partial Credit Models (Bayesian estimation)
- Item response functions (all model types)
- Item information functions (all model types)
- Test response functions/Expected score functions (all model types)
- Test information functions (all model types)

### Features in development
- Fitting Rating Scale Models via CML
- Fitting Partial Credit Models via CML
- Linear model extensions (Linear Logistic Test Model, Linear Rating Scale Model, Linear Partial Credit Model)
- Variational inference for Bayesian models

### Planned features
- Model evaluation 
- Model comparison