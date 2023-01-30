module RaschModels

using AbstractItemResponseModels
using LogExpFunctions
using Optim
using Reexport
using ReverseDiff
using Turing

import StatsAPI: StatisticalModel, coef, params
import AbstractItemResponseModels: response_type, person_dimensionality, item_dimensionality,
    estimation_type

@reexport begin
    import AbstractItemResponseModels: irf, iif, expected_score, information, fit
end

@reexport begin
    using Turing: MH, HMC, NUTS, MLE, MAP
    using Turing: MCMCSerial, MCMCThreads, MCMCDistributed
end

export PartialCreditModel
export RatingScaleModel
export RaschModel

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
