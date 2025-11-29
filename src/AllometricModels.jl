module AllometricModels

using StatsBase, Distributions, HypothesisTests, StatsModels, Revise, Tables, LinearAlgebra, Combinatorics, Base.Threads

import StatsAPI: coef, coeftable, coefnames, confint, deviance, nulldeviance, dof, dof_residual, loglikelihood,
  nullloglikelihood, nobs, stderror, vcov, residuals, predict, predict!, fitted, fit, response,
  modelmatrix, r2, r², adjr2, adjr², pvalue

import StatsModels: missing_omit, formula, modelmatrix

function combinationsfit(model, args...)
  error("Backend for $model not loaded or implemented.")
end

include("structs.jl")
include("parameters.jl")
include("regression.jl")
include("io.jl")

# Export Main Interface
export regression, AllometricModel,
  # Export StatsModels Types (Necessary for 'hints' argument)
  ContinuousTerm, CategoricalTerm,
  # Export StatsAPI Methods (So user can call them directly)
  formula,
  coef,
  coefnames,
  coeftable,
  confint,
  vcov,
  stderror,
  loglikelihood,
  nullloglikelihood,
  deviance,
  nulldeviance,
  nobs,
  dof,
  dof_residual,
  r2, r²,           # Unicode alias included
  adjr2, adjr²,     # Unicode alias included
  predict, fitted,
  residuals,
  response,
  pvalue

end