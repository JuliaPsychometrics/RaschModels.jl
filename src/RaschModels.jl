module RaschModels

# using AbstractItemResponseModels
# using LogExpFunctions
# using Optim
# using Roots
# using Reexport
# using ReverseDiff
# using Turing

# using NamedArrays
# using LinearAlgebra

# import ItemResponseFunctions
# using ItemResponseFunctions: irf, iif, information, expected_score, ItemParameters

# import LinearAlgebra: I
# import StatsAPI:
#     StatisticalModel,
#     coef,
#     coeftable,
#     coefnames,
#     params,
#     informationmatrix,
#     vcov,
#     stderror,
#     loglikelihood
# import StatsBase: CoefTable
# import AbstractItemResponseModels:
#     response_type, person_dimensionality, item_dimensionality, estimation_type

# @reexport begin
#     import AbstractItemResponseModels:
#         irf, iif, expected_score, information, fit, getitemlocations, getpersonlocations
# end

# @reexport begin
#     using Turing: MH, HMC, NUTS, MLE, MAP
#     using Turing: MCMCSerial, MCMCThreads, MCMCDistributed
# end

# export PartialCreditModel
# export RatingScaleModel
# export RaschModel

# export CML, SummationAlgorithm
# export PersonParameterWLE, PersonParameterMLE

# # test
# using DimensionalData
# using DocStringExtensions
# using PersonParameters

# export BayesianRaschModel, BayesianRatingScaleModel, BayesianPartialCreditModel
# export FrequentistRaschModel

# # include("types.jl")
# # include("priors.jl")
# # include("missings.jl")
# # include("models/models.jl")
# # include("algorithms/algorithms.jl")


# # include("turing_model.jl")

# # # include("esf.jl")
# # # include("cml.jl")
# # # include("personpars.jl")

# # include("fit.jl")

# # include("utils.jl")

using AbstractItemResponseModels
using DimensionalData: AbstractDimArray, DimArray
using Distributions: ContinuousUnivariateDistribution, Normal, InverseGamma, Categorical
using DocStringExtensions: TYPEDEF, FIELDS, SIGNATURES
using ItemResponseFunctions: ItemParameters, OnePL, PCM, RSM
using LinearAlgebra: I
using MCMCChains: Chains, namesingroup
using Optim: MultivariateOptimizationResults, optimize, BFGS
using PersonParameters: PersonParameterAlgorithm, person_parameters
using Reexport: @reexport
using Turing: @model, @addlogprob!, condition, sample, filldist, logpdf, BernoulliLogit
import ReverseDiff

import Optim: optimize

import StatsAPI:
    StatisticalModel,
    coef,
    coeftable,
    coefnames,
    params,
    informationmatrix,
    vcov,
    stderror,
    loglikelihood

@reexport import AbstractItemResponseModels:
    estimation_type,
    expected_score,
    fit,
    getitemlocations,
    getpersonlocations,
    iif,
    information,
    irf,
    item_dimensionality,
    person_dimensionality

@reexport using PersonParameters: WLE, MLE, MAP, EAP

@reexport using Turing:
    MH,
    HMC,
    NUTS,
    MCMCSerial,
    MCMCThreads,
    MCMCDistributed,
    AutoForwardDiff,
    AutoReverseDiff

export Prior,
    AbstractRaschModel,
    BayesianRaschModel,
    BayesianRatingScaleModel,
    BayesianPartialCreditModel,
    FrequentistRaschModel,
    CML,
    SummationAlgorithm

include("priors.jl")
include("missings.jl")
include("models/models.jl")
include("algorithms/algorithms.jl")
include("fit.jl")
include("utils.jl")

end
