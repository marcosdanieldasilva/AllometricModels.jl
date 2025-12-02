module GLMExt

using AllometricModels, GLM, StatsModels, Base.Threads

import AllometricModels: combinationsfit, TermTuple, β₀

function AllometricModels.combinationsfit(::Type{LinearModel}, cols::NamedTuple, ylist::Vector{Tuple{AbstractTerm,Vector{Float64}}}, combinations::Vector{TermTuple}, qterms::Vector{<:AbstractTerm}, positive::Bool)
  # pre-calculate intercept column
  X0 = modelcols(β₀, cols)

  # pre-calculate categorical column(s)
  if !isempty(qterms)
    Qsum = sum(qterms)
    Qmatrix = modelmatrix(Qsum, cols)
  end

  nx = length(combinations)
  ny = length(ylist)

  # pre-allocate output matrix
  fittedmodels = Matrix{Union{StatsModels.TableRegressionModel,Missing}}(undef, ny, nx)

  Threads.@threads for ix in 1:nx
    c = combinations[ix]

    if isempty(qterms)
      X = hcat(X0, map(last, c)...)
      rhs = MatrixTerm(mapfoldl(first, +, c; init=β₀))
    else
      X = hcat(X0, map(last, c)..., Qmatrix)
      rhs = MatrixTerm(mapfoldl(first, +, c; init=β₀) + Qsum)
    end

    for iy in 1:ny
      (yt, Y) = ylist[iy]

      try
        # fit using GLM
        fittedmodel = fit(LinearModel, X, Y)

        # reconstruct wrappers for TableRegressionModel
        ft = FormulaTerm(yt, MatrixTerm(rhs))
        mf = ModelFrame(ft, cols)
        mm = ModelMatrix(X, StatsModels.asgn(rhs))

        fittedmodels[iy, ix] = StatsModels.TableRegressionModel(fittedmodel, mf, mm)
      catch
        fittedmodels[iy, ix] = missing
      end
    end
  end

  # flatten and remove missing fits
  fittedmodels = collect(skipmissing(vec(fittedmodels)))

  if isempty(fittedmodels)
    error("failed to fit all linear regression models via GLM: LinearModel")
  end

  return fittedmodels
end

end