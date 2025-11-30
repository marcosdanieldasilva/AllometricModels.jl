expandxterms(xterm::AbstractTerm) = AbstractTerm[
  xterm,
  # polynomials (shape)
  FunctionTerm(x -> x^1.5, [xterm], :($(xterm)^1.5)),
  FunctionTerm(x -> x^2, [xterm], :($(xterm)^2)),
  FunctionTerm(x -> x^3, [xterm], :($(xterm)^3)),
  # fractional/root (stabilization)
  FunctionTerm(sqrt, [xterm], :(√$(xterm))),
  FunctionTerm(cbrt, [xterm], :(∛$(xterm))),
  # logarithmic (growth behavior)
  FunctionTerm(log, [xterm], :(log($(xterm)))),
  FunctionTerm(x -> log(x)^2, [xterm], :(log($(xterm))^2)),
  FunctionTerm(x -> log(x)^3, [xterm], :(log($(xterm))^3)),
  # inverse (asymptotic)
  FunctionTerm(x -> x^-0.5, [xterm], :($(xterm)^-0.5)),
  FunctionTerm(inv, [xterm], :($(xterm)^-1)),
  FunctionTerm(x -> x^-2, [xterm], :($(xterm)^-2)),
  FunctionTerm(x -> x^-3, [xterm], :($(xterm)^-3))
]

inversesqrt(y::Real) = 1 / √y
xoversqrty(y::Real, x::Real) = x / √y
xsquaredovery(y::Real, x::Real) = x^2 / y

function transformyterms(yterm::AbstractTerm, xterms::Vector{<:AbstractTerm})
  ylist = AbstractTerm[
    yterm
    FunctionTerm(log, [yterm], :(log($yterm)))
    FunctionTerm(inv, [yterm], :($yterm^-1))
    FunctionTerm(inversesqrt, [yterm], :(√($yterm)^-1))
  ]
  # add combined transformations with each x
  for xt in xterms
    push!(ylist,
      FunctionTerm(xoversqrty, [yterm, xt], :($xt / √($yterm)))
    )
    push!(ylist,
      FunctionTerm(xsquaredovery, [yterm, xt], :(($xt)^2 / $yterm))
    )
  end

  return ylist
end

function combinationsfit(::Type{AllometricModel}, cols::NamedTuple, ylist::Vector{Tuple{AbstractTerm,Vector{Float64}}}, combinations::Vector{TermTuple}, qterms::Vector{<:AbstractTerm})
  # pre-calculate intercept column
  X₀ = modelcols(β₀, cols)
  # the real dependent variable
  yᵣ = cols[1]
  # total sum of squares
  ȳ = mean(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)
  # pre-calculate categorical column(s)
  if !isempty(qterms)
    Qsum = sum(qterms)
    Qmatrix = modelmatrix(Qsum, cols)
  end
  # pre-allocate output matrix
  nx = length(combinations)
  ny = length(ylist)
  fittedmodels = Matrix{Union{Missing,AllometricModel}}(undef, ny, nx)

  Threads.@threads for ix in 1:nx
    c = combinations[ix]

    if isempty(qterms)
      X = hcat(X₀, map(last, c)...)
      rhs = MatrixTerm(mapfoldl(first, +, c; init=β₀))
    else
      X = hcat(X₀, map(last, c)..., Qmatrix)
      rhs = MatrixTerm(mapfoldl(first, +, c; init=β₀) + Qsum)
    end

    try
      chol = cholesky!(Symmetric(X'X))

      for iy in 1:ny
        (yt, y) = ylist[iy]
        # β = (X'X)⁻¹ * (X'y)
        β = X'y
        # Solve for regression coefficients β in-place using the Cholesky factor
        ldiv!(chol, β)
        # Allocate space for predicted values (ŷ) with the same structure as y
        ŷ = similar(y)
        # Compute predicted values in-place (ŷ = X * β)
        mul!(ŷ, X, β)
        # compute residuals
        ε = y - ŷ
        # Determine the number of observations (n) and the number of predictors (p)
        (n, p) = size(X)
        # Calculate the degrees of freedom for residuals
        ν = n - p
        # sum of squared errors
        sse = ε ⋅ ε
        # residual variance
        σ² = sse / ν
        # compute dispersion matrix
        Σ = rmul!(inv(chol), σ²)
        # normality check of residuals
        normality = isnormality(ε)
        # Correct the predicted values and residuals for models with a function on the left-hand side of the formula
        if isa(yt, FunctionTerm)
          # Apply the function-specific prediction logic
          # ŷ = predictBiasCorrected(cols, yt, ŷ, σ²)
          predictbiascorrected!(ŷ, cols, yt, σ²)
          ε = yᵣ - ŷ
          sse = ε ⋅ ε
        end
        # Compute the coefficient of determination (R²), a measure of model fit
        r² = 1 - sse / sst
        # Compute the adjusted R², penalized for the number of predictors
        adjr² = 1 - (1 - r²) * (n - 1) / ν
        # Explained Variance (EV)
        ev = 1.0 - (var(ε) / var(yᵣ))
        # Calculate the Mean Absolute Error (MAE) as the average absolute residual value
        mae = mean(abs, ε)
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = mean(abs.(ε) ./ abs.(yᵣ)) * 100
        # Calculate the variance of the residuals (Mean Squared Error - Unbiased)
        mse = sse / ν
        # Standard Error of the Estimate (RMSE - Absolute)
        rmse = √(mse)
        # Standard Error of the Estimate % (Syx% - Relative)
        cv = (rmse / ȳ) * 100
        # store fitted model
        fittedmodels[iy, ix] = AllometricModel(
          FormulaTerm(yt, rhs), cols, β, Σ, σ², n, ν, p, sse, sst, r², adjr², ev, mae, mape, mse, rmse, cv, normality)
      end
    catch
      for iy in 1:ny
        fittedmodels[iy, ix] = missing
      end
    end

  end

  # # flatten and remove missing fits
  fittedmodels = collect(skipmissing(vec(fittedmodels)))
  isempty(fittedmodels) && error("failed to fit all Allometric Linear Regression Models")

  return fittedmodels
end

function regression(data, yname::S, xnames::S...; hints=Dict{Symbol,Any}(), model=AllometricModel, nmin::Int=1, nmax::Int=3)
  # Input Validation
  if isempty(xnames)
    throw(ArgumentError("no independent variables provided"))
  end

  if nmin < 1
    throw(ArgumentError("nmin must be >= 1"))
  end

  if nmax < nmin
    throw(ArgumentError("nmax must be >= nmin"))
  end

  # --- CORREÇÃO: Limite Rígido de Segurança ---
  if nmax > 5
    throw(ArgumentError("nmax ($nmax) exceeds the practical limit for allometric models (max 5). "
                        * "Models with more than 5 terms (e.g., degree > 5) cause severe overfitting, "
                        * "numerical instability (singular matrices), and lack biological meaning."))
  end

  yname = Symbol(yname)
  xnames = Symbol.(xnames)

  if !Tables.istable(data)
    throw(ArgumentError("data must be a valid table"))
  else
    cols = columntable(data)[(yname, xnames...)]
  end

  # Term Preparation
  yterm = concrete_term(term(yname), cols, ContinuousTerm)

  # apply schema to x terms
  xschema = apply_schema(term.(xnames), schema(cols, hints))
  xterms = xschema isa AbstractTerm ? AbstractTerm[xschema] : collect(AbstractTerm, xschema)

  # separate categorical (q) and continuous (x) terms
  qterms = filter(t -> t isa CategoricalTerm, xterms)
  filter!(t -> !(t isa CategoricalTerm), xterms)

  if isempty(xterms)
    throw(ArgumentError("no continuous variables found"))
  end

  ft = isempty(qterms) ? FormulaTerm(yterm, MatrixTerm(β₀ + sum(xterms))) : FormulaTerm(yterm, MatrixTerm(β₀ + sum(xterms) + sum(qterms)))

  cols, _ = missing_omit(cols, ft)

  # Build Transformation Groups (Continuous)
  termgroups = Vector{Vector{Tuple{AbstractTerm,Vector{Float64}}}}()
  sizehint!(termgroups, length(xterms))

  for var in xterms
    subgroup = Tuple{AbstractTerm,Vector{Float64}}[]

    for t in expandxterms(var)
      try
        col = modelcols(t, cols)
        if all(isfinite, col)
          push!(subgroup, (t, col))
        end
      catch
        # ignore invalid transforms
      end
    end

    if !isempty(subgroup)
      push!(termgroups, subgroup)
    end
  end

  if isempty(termgroups)
    throw(ErrorException("failed to generate valid transformations"))
  end

  # generate combinations (continuous part only)
  combinations = if length(termgroups) == 1
    TermTuple[Tuple(c) for c in powerset(termgroups[1], nmin, nmax)]
  else
    TermTuple[Tuple(g) for g in Iterators.product(termgroups...)] |> vec
  end

  # Build y List
  ylist = Vector{Tuple{AbstractTerm,Vector{Float64}}}()

  for t in transformyterms(yterm, xterms)
    try
      v = modelcols(t, cols)
      if all(isfinite, v)
        push!(ylist, (t, v))
      end
    catch
      # ignore invalid transforms
    end
  end

  combinationsfit(model, cols, ylist, combinations, qterms)
end

