using BenchmarkTools
using RaschModels

n_items = 5
n_categories = 4
n_persons = 100
n_iter = 1000

data = rand(1:n_categories, n_persons, n_items)

# PointEstimate
mle = fit(RatingScaleModel, data, MLE())


irf(mle, 0.0, 1)
@code_warntype irf(mle, 0.0, 1)
@benchmark irf($mle, $randn(), $(rand(1:n_items)))
@profview_allocs irf(mle, 0.0, 1) sample_rate = 1

@benchmark irf($mle, $randn(), $(rand(1:n_items)), $(rand(1:n_categories)))
