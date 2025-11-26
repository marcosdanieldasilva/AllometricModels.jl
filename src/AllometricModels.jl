module AllometricModels

using StatsModels, Tables, LinearAlgebra, Combinatorics, Base.Threads

const S = Union{Symbol,String}
const TermTuple = Tuple{Vararg{Tuple{AbstractTerm,Vector{Float64}}}}
const β0 = InterceptTerm{true}()

abstract type AllometricModel <: RegressionModel end

function combinationsfit(model, args...)
  error("Backend for $model not loaded or implemented.")
end

include("regression.jl")

export regression, AllometricModel

end