using AllometricModels
using Documenter

DocMeta.setdocmeta!(AllometricModels, :DocTestSetup, :(using AllometricModels); recursive=true)

makedocs(;
    modules=[AllometricModels],
    authors="Marcos Daniel da Silva <marcosdasilva@5a.tec.br> and contributors",
    sitename="AllometricModels.jl",
    format=Documenter.HTML(;
        canonical="https://marcosdanieldasilva.github.io/AllometricModels.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/marcosdanieldasilva/AllometricModels.jl",
    devbranch="master",
)
