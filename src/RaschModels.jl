module RaschModels

using AbstractItemResponseModels
using DimensionalData: AbstractDimArray, DimArray
using Distributions: ContinuousUnivariateDistribution, Normal, InverseGamma, Categorical
using DocStringExtensions: TYPEDEF, FIELDS, SIGNATURES
using ItemResponseFunctions: ItemParameters, OnePL, PCM, RSM
using LinearAlgebra: I
using MCMCChains: Chains, namesingroup
using Optim: MultivariateOptimizationResults, optimize, BFGS
using PersonParameters: PersonParameterAlgorithm, person_parameters, value
using Reexport: @reexport
using Turing: @model, @addlogprob!, condition, sample, filldist, logpdf, BernoulliLogit
using Turing.Inference: InferenceAlgorithm

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
    person_dimensionality,
    response_type

@reexport using PersonParameters: WLE, MLE, MAP, EAP
@reexport using Turing: NUTS, MCMCThreads, MCMCDistributed, AutoReverseDiff

export Prior,
    AbstractRaschModel,
    RaschModel,
    PartialCreditModel,
    RatingScaleModel,
    CML,
    SummationAlgorithm

include("priors.jl")
include("missings.jl")
include("models/models.jl")
include("algorithms/algorithms.jl")
include("fit.jl")
include("utils.jl")

end
