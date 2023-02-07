using BenchmarkTools
using RaschModels

n_items = 5
n_categories = 4
n_persons = 100
n_iter = 100

data = rand(1:n_categories, n_persons, n_items)

# PointEstimate
mle = fit(RatingScaleModel, data, MLE())

@benchmark irf($mle, $randn(), $(rand(1:n_items)), $(rand(1:n_categories)))
@benchmark iif($mle, $randn(), $(rand(1:n_items)), $(rand(1:n_categories)))

@benchmark expected_score($mle, $randn(), $(rand(1:n_items)))
@benchmark expected_score($mle, $randn(), $(rand(1:n_items, 3)))
@benchmark expected_score($mle, $randn())

@benchmark information($mle, $randn(), $(rand(1:n_items)))
@benchmark information($mle, $randn(), $(rand(1:n_items, 3)))
@benchmark information($mle, $randn())

# SamplingEstimate
mcmc = fit(RatingScaleModel, data, NUTS(), n_iter)

@benchmark irf($mcmc, $randn(), $(rand(1:n_items)), $(rand(1:n_categories)))
@benchmark iif($mcmc, $randn(), $(rand(1:n_items)), $(rand(1:n_categories)))

@benchmark expected_score($mcmc, $randn(), $(rand(1:n_items)))
@benchmark expected_score($mcmc, $randn(), $(rand(1:n_items, 3)))
@benchmark expected_score($mcmc, $randn())

@benchmark information($mcmc, $randn(), $(rand(1:n_items)))
@benchmark information($mcmc, $randn(), $(rand(1:n_items, 3)))
@benchmark information($mcmc, $randn())
