using RaschModels
using Documenter

DocMeta.setdocmeta!(RaschModels, :DocTestSetup, :(using RaschModels); recursive = true)

makedocs(;
    modules = [RaschModels],
    authors = "Philipp Gewessler",
    repo = "https://github.com/JuliaPsychometrics/RaschModels.jl/blob/{commit}{path}#{line}",
    sitename = "RaschModels.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliapsychometrics.github.io/RaschModels.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["Bayesian modeling" => "tutorials/bayesian_modeling.md"],
        "API" => ["Types" => "api/types.md", "Functions" => "api/functions.md"],
    ],
)

deploydocs(; repo = "github.com/JuliaPsychometrics/RaschModels.jl", devbranch = "main")
