module AllometricModels

using StatsBase, Statistics, StatsModels, Revise, Tables, LinearAlgebra, Combinatorics, Base.Threads
# Distributions, HypothesisTests not used by now
import StatsBase: coef, coeftable, coefnames, confint, deviance, nulldeviance, dof, dof_residual,
  loglikelihood, nullloglikelihood, nobs, stderror, vcov,
  residuals, predict, predict!,
  fitted, fit, model_response, response, modelmatrix, r2, r², adjr2, adjr², PValue
import StatsModels: missing_omit, formula

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