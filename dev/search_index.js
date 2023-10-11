var documenterSearchIndex = {"docs":
[{"location":"api/types/","page":"Types","title":"Types","text":"CurrentModule = RaschModels","category":"page"},{"location":"api/types/#Types","page":"Types","title":"Types","text":"","category":"section"},{"location":"api/types/#Index","page":"Types","title":"Index","text":"","category":"section"},{"location":"api/types/","page":"Types","title":"Types","text":"Pages = [\"types.md\"]","category":"page"},{"location":"api/types/#Type-definitions","page":"Types","title":"Type definitions","text":"","category":"section"},{"location":"api/types/","page":"Types","title":"Types","text":"PartialCreditModel\nRaschModel\nRatingScaleModel","category":"page"},{"location":"api/types/#RaschModels.PartialCreditModel","page":"Types","title":"RaschModels.PartialCreditModel","text":"PartialCreditModel <: PolytomousRaschModel\n\nA type representing a Partial Credit Model.\n\n\n\n\n\n","category":"type"},{"location":"api/types/#RaschModels.RaschModel","page":"Types","title":"RaschModels.RaschModel","text":"RaschModel\n\nFields\n\ndata: The data matrix used to fit the model\npars: The structure holding parameter values\nparnames_beta: A vector of parameter names for item parameters\n\n\n\n\n\n","category":"type"},{"location":"api/types/#RaschModels.RatingScaleModel","page":"Types","title":"RaschModels.RatingScaleModel","text":"RatingScaleModel <: PolytomousRaschModel\n\nA type representing a Rating Scale Model\n\n\n\n\n\n","category":"type"},{"location":"tutorials/bayesian_modeling/#Bayesian-Estimation-of-a-Rasch-Model","page":"Bayesian modeling","title":"Bayesian Estimation of a Rasch Model","text":"","category":"section"},{"location":"tutorials/bayesian_modeling/","page":"Bayesian modeling","title":"Bayesian modeling","text":"In this tutorial we will fit a Rasch Model using Bayesian estimation.  Bayesian estimation in RaschModels.jl leverages the Turing.jl ecosystem and thus is composable with all algorithms provided by Turing.jl.","category":"page"},{"location":"#RaschModels.jl","page":"Home","title":"RaschModels.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) (Image: Coverage)","category":"page"},{"location":"","page":"Home","title":"Home","text":"RaschModels.jl is a Julia package for fitting and evaluating Rasch Models. It implements the basic Rasch Model, Partial Credit Model, and Rating Scale Model, as well as their  linear extensions. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note: Currently only a subset of models is available. Please see Roadmap for details.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install this package you can use Julias package management system.","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add RaschModels","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Fitting a model using RaschModels.jl is easy. First, get some response data as a Matrix.  In this example we just use some random data for 100 persons and 5 items.","category":"page"},{"location":"","page":"Home","title":"Home","text":"data = rand(0:1, 100, 5)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Using data as our response data we can fit a Rasch Model. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"rasch = fit(RaschModel, data, CML())","category":"page"},{"location":"","page":"Home","title":"Home","text":"This function call fits the model using conditional Maximum Likelihood estimation.  To fit the Rasch Model using Bayesian estimation just change the algorithm and provide the  required additional arguments.","category":"page"},{"location":"","page":"Home","title":"Home","text":"rasch_bayes = fit(RaschModel, data, NUTS(), 1_000)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Additional plotting capabilities are provided by ItemResponsePlots.jl.","category":"page"},{"location":"#Roadmap","page":"Home","title":"Roadmap","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"RaschModels.jl is still under active development. Therefore, not all functionality is  available yet. This roadmap provides a quick overview of the current state of the package.","category":"page"},{"location":"#Existing-features","page":"Home","title":"Existing features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Fitting Rasch Models (CML estimation, Bayesian estimation)\nFitting Rating Scale Models (Bayesian estimation)\nFitting Partial Credit Models (Bayesian estimation)\nItem response functions (all model types)\nItem information functions (all model types)\nTest response functions/Expected score functions (all model types)\nTest information functions (all model types)","category":"page"},{"location":"#Features-in-development","page":"Home","title":"Features in development","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Fitting Rating Scale Models via CML\nFitting Partial Credit Models via CML\nLinear model extensions (Linear Logistic Test Model, Linear Rating Scale Model, Linear Partial Credit Model)\nVariational inference for Bayesian models","category":"page"},{"location":"#Planned-features","page":"Home","title":"Planned features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Model evaluation \nModel comparison","category":"page"},{"location":"api/functions/","page":"Functions","title":"Functions","text":"CurrentModule = RaschModels","category":"page"},{"location":"api/functions/#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"api/functions/#Index","page":"Functions","title":"Index","text":"","category":"section"},{"location":"api/functions/","page":"Functions","title":"Functions","text":"Pages = [\"functions.md\"]","category":"page"},{"location":"api/functions/#Model-fitting","page":"Functions","title":"Model fitting","text":"","category":"section"},{"location":"api/functions/","page":"Functions","title":"Functions","text":"fit","category":"page"},{"location":"api/functions/#StatsAPI.fit","page":"Functions","title":"StatsAPI.fit","text":"fit(modeltype::Type{<: AbstractRaschModel}, data, alg, args...; kwargs...)\n\nFit a Rasch Model to response data.\n\nArguments\n\nmodeltype\n\nAn AbstractRaschModel type. Currently RaschModels.jl implements the following model types:\n\nPartialCreditModel\nRaschModel\nRatingScaleModel\n\ndata\n\nThe observed data matrix. It is assumed that each person corresponds to a row and each item to a column in the matrix.\n\nFor dichotomous models correct responses are coded as 1 and incorrect responses are coded as 0.\n\nFor polytomous models, categories are coded as an integer sequence starting from 1. If your model features e.g. three categories, they are coded as 1, 2 and 3 respectively.\n\nalg\n\nThe estimation algorithm.\n\nThe following algorithm is still in development phase, by now only the RaschModel is supported.\n\nCML: Conditional maximum likelihood\n\nFor (bayesian) sampling based estimation all Turing.InferenceAlgorithm types are supported. The following algorithms are reexported from this package:\n\nNUTS: No-U-Turn-Sampler\nHMC: Hamiltonian Monte Carlo\nMH: Metropolis-Hastings\n\nBayesian point estimation is supported by\n\nMAP: Maximum a posteriori (Posterior mode)\nMLE: Maximum Likelihood (Posterior mean)\n\nExamples\n\nConditional maximum likelihood estimation\n\ndata = rand(0:1, 100, 10)\nrasch_fit = fit(RaschModel, data, CML())\n\nBayesian estimation\n\nFitting a simple Rasch model with the No-U-Turn-Sampler. Note that estimation with Turing.InterenceAlgorithm uses the AbstractMCMC.sample interface and thus requires an additional argument to specify the number of iterations.\n\ndata = rand(0:1, 100, 10)\nrasch_fit = fit(RaschModel, data, NUTS(), 1_000)\n\nYou can also sample multiple chains in parallel.\n\ndata = rand(0:1, 100, 10)\nrasch_fit = fit(RaschModel, data, NUTS(), MCMCThreads(), 1_000, 4)\n\nFor bayesian point estimation you can just swap the inference algorithm to one of the supported point estimation algorithms.\n\ndata = rand(0:1, 100, 10)\nrasch_fit = fit(RaschModel, data, MAP())\n\n\n\n\n\n","category":"function"},{"location":"api/functions/#Model-evaluation","page":"Functions","title":"Model evaluation","text":"","category":"section"},{"location":"api/functions/","page":"Functions","title":"Functions","text":"irf\niif\nexpected_score\ninformation","category":"page"},{"location":"api/functions/#AbstractItemResponseModels.irf","page":"Functions","title":"AbstractItemResponseModels.irf","text":"irf(model::RaschModel, theta, i, y)\nirf(model::RaschModel, theta, i)\n\nEvaluate the item response function for a dichotomous Rasch model for item i at the ability value theta.\n\nIf the response value y is omitted, the item response probability for a correct response y = 1 is returned.\n\nExamples\n\nPoint estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MLE());\n\njulia> irf(rasch, 0.0, 1)\n0.3989070983504997\n\nBayesian estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MH(), 100; progress = false);\n\njulia> irf(rasch, 0.0, 1);\n\n\n\n\n\nirf(model::PolytomousRaschModel, theta, i, y)\nirf(model::PolytomousRaschModel, theta, i)\n\nEvaluate the item response function for a polytomous Rasch model (Partial Credit Model or Rating Scale Model) for item i at the ability value theta.\n\nIf the response value y is omitted, the item response probabilities for each category are returned. To calculate expected scores for an item, see expected_score.\n\n\n\n\n\n","category":"function"},{"location":"api/functions/#AbstractItemResponseModels.iif","page":"Functions","title":"AbstractItemResponseModels.iif","text":"iif(model::RaschModel, theta, i, y)\niif(model::RaschModel, theta, i)\n\nEvaluate the item information function for a dichotomous Rasch model for item i at the ability value theta.\n\nIf the response value y is omitted, the item information for a correct response y = 1 is returned.\n\nExamples\n\nPoint estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MLE());\n\njulia> iif(rasch, 0.0, 1)\n0.07287100351275909\n\nBayesian estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MH(), 100; progress = false);\n\njulia> iif(rasch, 0.0, 1);\n\n\n\n\n\niif(model::PolytomousRaschModel, theta, i, y)\niif(model::PolytomousRaschModel, theta, i)\n\nEvaluate the item (category) information function for a polytomous Rasch model (Partial Credit Model or Rating Scale Model) for item i at the ability value theta.\n\nIf the response value y is omitted, the item information for each category is returned. To calculate the total information of an item, see @ref.\n\n\n\n\n\n","category":"function"},{"location":"api/functions/#AbstractItemResponseModels.expected_score","page":"Functions","title":"AbstractItemResponseModels.expected_score","text":"expected_score(model::RaschModel, theta, is; scoring_function)\nexpected_score(model::RaschModel, theta; scoring_function)\n\nCalculate the expected score for a dichotomous Rasch model at ability value theta for a set of items is.\n\nis can either be a single item index, an array of item indices, or a range of values. If is is omitted, the expected score for the whole test is calculated.\n\nscoring_function can be used to add weights to the resulting expected scores (see Examples for details).\n\nExamples\n\nPoint estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MLE());\n\njulia> expected_score(rasch, 0.0)  # all 3 items\n0.4435592483649984\n\njulia> expected_score(rasch, 0.0, 1:2)  # items 1 and 2\n0.3054954277378461\n\njulia> expected_score(rasch, 0.0, [1, 3])  # items 1 and 3\n0.21719686244768088\n\nBayesian estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MH(), 100; progress = false);\n\njulia> expected_score(rasch, 0.0);  # all 3 items\n\njulia> expected_score(rasch, 0.0, 1:2);  # items 1 and 2\n\njulia> expected_score(rasch, 0.0, [1, 3]);  # items 1 and 3\n\nUsing the scoring function\n\nUsing the scoring_function keyword argument allows to weigh response probabilities by a value depending on the response y. It is of the form f(y) = x, assigning a scalar value to every possible reponse value y.\n\nFor the Rasch Model the valid responses are 0 and 1. If we want to calculate expected scores doubling the weight for y = 1, the weighted responses are 0 and 2. The corresponding scoring_function is y -> 2y,\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MLE());\n\njulia> expected_score(rasch, 0.0; scoring_function = y -> 2y)\n4.592642952772852\n\n\n\n\n\nexpected_score(model::PolytomousRaschModel, theta, is)\nexpected_score(model::PolytomousRaschModel, theta)\n\nCalculate the expected score for a polytomous Rasch model at theta for a set of items is.\n\nis can either be a single item index, an array of item indices, or a range of values. If is is omitted, the expected score for the whole test is calculated.\n\n\n\n\n\n","category":"function"},{"location":"api/functions/#AbstractItemResponseModels.information","page":"Functions","title":"AbstractItemResponseModels.information","text":"information(model::RaschModel, theta, is; scoring_function)\ninformation(model::RaschModel, theta; scoring_function)\n\nCalculate the information for a dichotomous Rasch model at the ability value theta for a set of items is.\n\nis can either be a single item index, an array of item indices, or a range of values. If is is omitted, the information for the whole test is calculated.\n\nscoring_function can be used to add weights to the resulting information (see Examples for details).\n\nExamples\n\nPoint estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MLE());\n\njulia> information(rasch, 0.0)  # all 3 items\n0.519893299555712\n\njulia> information(rasch, 0.0, 1:2)  # items 1 and 2\n0.4121629113381827\n\njulia> information(rasch, 0.0, [1, 3])  # items 1 and 3\n0.313811843802304\n\nBayesian estimation\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MH(), 100; progress = false);\n\njulia> information(rasch, 0.0);  # all 3 items\n\njulia> information(rasch, 0.0, 1:2);  # items 1 and 2\n\njulia> information(rasch, 0.0, [1, 3]);  # items 1 and 3\n\nUsing the scoring function\n\nUsing the scoring_function keyword argument allows to weigh response probabilities by a value depending on the response y. It is of the form f(y) = x, assigning a scalar value to every possible reponse value y.\n\nFor the Rasch Model the valid responses are 0 and 1. If we want to calculate the information doubling the weight for y = 1, the weighted responses are 0 and 2. The corresponding scoring_function is y -> 2y,\n\njulia> data = rand(0:1, 50, 3);\n\njulia> rasch = fit(RaschModel, data, MLE());\n\njulia> information(rasch, 0.0; scoring_function = y -> 2y)\n2.079573198222848\n\n\n\n\n\ninformation(model::PolytomousRaschModel, theta, is)\ninformation(model::PolytomousRaschModel, theta)\n\nCalculate the information for a polytomous Rasch model at theta for a set of items is.\n\nis can either be a single item index, an array of item indices, or a range of values. If is is omitted, the information for the whole test is calculated.\n\n\n\n\n\n","category":"function"}]
}
