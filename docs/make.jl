using Documenter
using DocumenterCitations
using CairoMakie
using PolynomialModelReductionDataset

ENV["JULIA_DEBUG"] = "Documenter"
DocMeta.setdocmeta!(PolynomialModelReductionDataset, :DocTestSetup, :(using PolynomialModelReductionDataset); recursive=true)

PAGES = [
    "Home" => "index.md",
    "1D Models" => [
        "1D Heat" => "1D/heat1d.md",
        "Viscous Burgers'" => "1D/burgers.md",
        "Kuramoto-Sivashinsky" => "1D/kse.md",
        "FisherkPP" => "1D/fisherkpp.md",
        "Allen-Cahn" => "1D/allencahn.md",
        "Modified Korteweg-de Vries" => "1D/mKdV.md",
        "Modified Korteweg-de Vries-Burgers" => "1D/mKdVB.md",
        "Gardner" => "1D/gardner.md",
        "Damped Gardner-Burgers" => "1D/dgb.md",
        "FitzHugh-Nagumo" => "1D/fhn.md",
    ],
    "2D Models" => [
        "2D Heat" => "2D/heat2d.md",
    ],
    "API Reference" => "api.md",
    "Paper Reference" => "paper.md",
]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "PoMoReDa",
    clean = true, doctest = false, linkcheck = false,
    authors = "Tomoki Koike <tkoike45@gmail.com>",
    repo = Remotes.GitHub("smallpondtom", "PolynomialModelReductionDataset.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "https://github.com/smallpondtom/PoMoReDa",
        assets=String[
            "assets/citations.css",
            "assets/favicon.ico",
        ],
        # analytics = "G-B2FEJZ9J99",
    ),
    modules = [PolynomialModelReductionDataset,],
    pages = PAGES,
    plugins=[bib],
)

deploydocs(
    repo = "github.com/smallpondtom/PolynomialModelReductionDataset.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
    # Add other deployment options as needed
)