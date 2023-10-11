module RaschModels

using AbstractItemResponseModels
using LogExpFunctions
using Optim
using Roots
using Reexport
using ReverseDiff
using Turing

using NamedArrays
using LinearAlgebra

import LinearAlgebra: I
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
import StatsBase: CoefTable
import AbstractItemResponseModels:
    response_type, person_dimensionality, item_dimensionality, estimation_type

@reexport begin
    import AbstractItemResponseModels:
        irf, iif, expected_score, information, fit, getitemlocations, getpersonlocations
end

@reexport begin
    using Turing: MH, HMC, NUTS, MLE, MAP
    using Turing: MCMCSerial, MCMCThreads, MCMCDistributed
end

export PartialCreditModel
export RatingScaleModel
export RaschModel

export CML, SummationAlgorithm
export PersonParameterWLE, PersonParameterMLE

include("utils.jl")
include("types.jl")
include("priors.jl")
include("missings.jl")

include("models/RaschModel.jl")
include("models/PolytomousRaschModel.jl")
include("models/PartialCreditModel.jl")
include("models/RatingScaleModel.jl")

include("turing_model.jl")

include("esf.jl")
include("cml.jl")
include("personpars.jl")

include("fit.jl")

end
