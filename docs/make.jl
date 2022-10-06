using RaschModels
using Documenter

DocMeta.setdocmeta!(RaschModels, :DocTestSetup, :(using RaschModels); recursive=true)

makedocs(;
    modules=[RaschModels],
    authors="Philipp Gewessler",
    repo="https://github.com/p-gw/RaschModels.jl/blob/{commit}{path}#{line}",
    sitename="RaschModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://p-gw.github.io/RaschModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/p-gw/RaschModels.jl",
    devbranch="main",
)
