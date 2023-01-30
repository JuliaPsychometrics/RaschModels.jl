module RaschModels

using AbstractItemResponseModels
using LogExpFunctions
using Optim
using Reexport
using ReverseDiff
using Turing
using StaticArrays

import StatsAPI: StatisticalModel, coef, params
import AbstractItemResponseModels: response_type, person_dimensionality, item_dimensionality,
    estimation_type, fit, irf, iif, information, expected_score

@reexport begin
    using Turing: MH, HMC, NUTS, MLE, MAP
    using Turing: MCMCSerial, MCMCThreads, MCMCDistributed
end

export PartialCreditModel
export RatingScaleModel
export RaschModel
export fit
export irf, iif, expected_score, information

include("utils.jl")
include("types.jl")
include("priors.jl")

include("models/RaschModel.jl")
include("models/PolytomousRaschModel.jl")
include("models/PartialCreditModel.jl")
include("models/RatingScaleModel.jl")

include("turing_model.jl")

include("fit.jl")

end
