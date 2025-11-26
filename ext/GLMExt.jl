module GLMExt

using AllometricModels, GLM, StatsModels, Base.Threads

import AllometricModels: combinationsfit, TermTuple, β0

function AllometricModels.combinationsfit(::Type{LinearModel}, cols::NamedTuple, ylist::Vector{Tuple{AbstractTerm,Vector{Float64}}}, combinations::Vector{TermTuple}, qterms::Vector{<:AbstractTerm})
  # pre-calculate intercept column (X0)
  X0 = modelcols(β0, cols)

  nx = length(combinations)
  ny = length(ylist)

  # pre-allocate output matrix
  fittedmodels = Matrix{Union{StatsModels.TableRegressionModel,Missing}}(undef, ny, nx)

  # split path for optimization
  if isempty(qterms)
    # PATH A: Continuous Variables Only
    Threads.@threads for ix in 1:nx
      c = combinations[ix]

      # construct X: Intercept + Continuous Columns
      X = hcat(X0, map(last, c)...)

      # prepare formula rhs part
      xrhs = mapfoldl(first, +, c; init=β0)

      for iy in 1:ny
        (yt, Y) = ylist[iy]

        try
          # fit using GLM
          fittedmodel = fit(LinearModel, X, Y)

          # reconstruct wrappers for TableRegressionModel
          ft = FormulaTerm(yt, MatrixTerm(xrhs))
          mf = ModelFrame(ft, cols)
          mm = ModelMatrix(X, StatsModels.asgn(xrhs))

          fittedmodels[iy, ix] = StatsModels.TableRegressionModel(fittedmodel, mf, mm)
        catch
          fittedmodels[iy, ix] = missing
        end
      end
    end

  else
    # PATH B: Continuous + Categorical Variables
    # pre-calculate Q matrix and sum term once
    qsum = sum(qterms)
    qmatrix = modelmatrix(qsum, cols)

    Threads.@threads for ix in 1:nx
      c = combinations[ix]

      # construct X: Intercept + Continuous Columns + Q Matrix
      X = hcat(X0, map(last, c)..., qmatrix)

      # prepare formula rhs part
      xrhs = mapfoldl(first, +, c; init=β0) + qsum

      for iy in 1:ny
        (yt, Y) = ylist[iy]

        try
          # fit using GLM
          fittedmodel = fit(GLM.LinearModel, X, Y)

          # reconstruct wrappers
          ft = FormulaTerm(yt, MatrixTerm(xrhs))
          mf = ModelFrame(ft, cols)
          mm = ModelMatrix(X, StatsModels.asgn(xrhs))

          fittedmodels[iy, ix] = StatsModels.TableRegressionModel(fittedmodel, mf, mm)
        catch
          fittedmodels[iy, ix] = missing
        end
      end
    end
  end

  # flatten and remove missing fits
  results = collect(skipmissing(vec(fittedmodels)))

  if isempty(results)
    error("failed to fit all linear regression models via GLM: LinearModel")
  end

  return results
end

end