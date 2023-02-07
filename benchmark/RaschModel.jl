using BenchmarkTools
using RaschModels

n_items = 5
n_persons = 100
n_iter = 1000

data = rand(0:1, n_persons, n_items)

# PointEstimate
mle = fit(RaschModel, data, MLE())

@benchmark irf($mle, $randn(), $(rand(1:n_items)))
@benchmark iif($mle, $randn(), $(rand(1:n_items)))

@benchmark expected_score($mle, $randn(), $(rand(1:n_items)))
@benchmark expected_score($mle, $randn(), $(rand(1:n_items, 3)))
@benchmark expected_score($mle, $randn())

@benchmark information($mle, $randn(), $(rand(1:n_items)))
@benchmark information($mle, $randn(), $(rand(1:n_items, 3)))
@benchmark information($mle, $randn())

# SamplingEstimate
mcmc = fit(RaschModel, data, NUTS(), n_iter)

@benchmark irf($mcmc, $randn(), $(rand(1:n_items)))
@benchmark iif($mcmc, $randn(), $(rand(1:n_items)))

@benchmark expected_score($mcmc, $randn(), $(rand(1:n_items)))
@benchmark expected_score($mcmc, $randn(), $(rand(1:n_items, 3)))
@benchmark expected_score($mcmc, $randn())

@benchmark information($mcmc, $randn(), $(rand(1:n_items)))
@benchmark information($mcmc, $randn(), $(rand(1:n_items, 3)))
@benchmark information($mcmc, $randn())
