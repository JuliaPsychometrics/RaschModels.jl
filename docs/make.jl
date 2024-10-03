using RaschModels
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(RaschModels, :DocTestSetup, :(using RaschModels); recursive = true)

makedocs(;
    sitename = "RaschModels.jl",
    authors = ["Philipp Gewessler", "Tobias Alfers"],
    modules = [RaschModels],
    warnonly = true,
    checkdocs = :all,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaPsychometrics/RaschModels.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    clean = true,
    draft = false,
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["Bayesian modeling" => "tutorials/bayesian_modeling.md"],
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaPsychometrics/RaschModels.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
