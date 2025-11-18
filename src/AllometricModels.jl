module AllometricModels

using GLM, Logging, MixedModels, Reexport, StatsModels, Tables

@reexport using GLM
@reexport using MixedModels
@reexport using StatsModels

const nullschema = StatsModels.Schema()
const β0 = InterceptTerm{true}()
const S = Union{Symbol,String}
const TermTuple = Tuple{Vararg{AbstractTerm}}

include("regression.jl")

export regression

end