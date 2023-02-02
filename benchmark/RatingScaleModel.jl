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


iif(mle, 0.0, 1, 1)
@profview_allocs iif(mle, 0.0, 1, 1) sample_rate = 1

# SamplingEstimate
mcmc = fit(RatingScaleModel, data, NUTS(), n_iter)

irf(mcmc, 0.0, 1)

@benchmark irf($mcmc, $randn(), $(rand(1:n_items)))
@benchmark irf($mcmc, $randn(), $(rand(1:n_items)))

