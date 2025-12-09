module AllometricModels

using StatsBase, Distributions, HypothesisTests, StatsModels, Printf, Tables, LinearAlgebra, Combinatorics, Base.Threads

import StatsAPI: coef, coeftable, coefnames, confint, deviance, nulldeviance, dof, dof_residual, loglikelihood,
  nullloglikelihood, nobs, stderror, vcov, residuals, predict, predict!, fitted, fit, response,
  modelmatrix, r2, r², adjr2, adjr², pvalue

import StatsModels: missing_omit, formula, modelmatrix, @formula

function combinationsfit(model, args...)
  error("Backend for $model not loaded or implemented.")
end

include("structs.jl")
include("parameters.jl")
include("regression.jl")
include("selectioncriteria.jl")
include("io.jl")

# Export Main Interface
export regression, fit, AllometricModel,
  # Export StatsModels Types (Necessary for 'hints' argument)
  ContinuousTerm, CategoricalTerm,
  # Export StatsAPI Methods (So user can call them directly)
  @formula,
  adjr2,
  adjr²,
  bestmodel,
  coef,
  coefnames,
  coeftable,
  confint,
  cooksdistance,
  criteriatable,
  deviance,
  dispersion,
  dof_residual,
  dof,
  fitted,
  formula,
  gvif,
  isnormality,
  loglikelihood,
  metrics,
  modelmatrix,
  naslund,
  nobs,
  nulldeviance,
  nullloglikelihood,
  petterson,
  predict,
  prodan,
  pvalue,
  r2,
  r²,
  residuals,
  response,
  stderror,
  termnames,
  vcov,
  vif

end