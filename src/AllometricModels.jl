module AllometricModels

using StatsAPI, StatsModels, Revise, Tables, LinearAlgebra, Combinatorics, Base.Threads
# Distributions, HypothesisTests not used by now

function combinationsfit(model, args...)
  error("Backend for $model not loaded or implemented.")
end

include("structs.jl")
include("parameters.jl")
include("regression.jl")
include("io.jl")

export regression,
  AllometricModel,
  predict,
  residuals,
  deviance,
  nulldeviance,
  nobs,
  dof_residual,
  r2,
  adjr2,
  formula

end