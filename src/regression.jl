expandxterms(xterm::AbstractTerm) = AbstractTerm[
  xterm,
  # polynomials (shape)
  FunctionTerm(x -> x^2, [xterm], :($(xterm)^2)),
  FunctionTerm(x -> x^3, [xterm], :($(xterm)^3)),
  # fractional/root (stabilization)
  FunctionTerm(sqrt, [xterm], :(√$(xterm))),
  FunctionTerm(cbrt, [xterm], :(∛$(xterm))),
  # logarithmic (growth behavior)
  FunctionTerm(log, [xterm], :(log($(xterm)))),
  FunctionTerm(x -> log(x)^2, [xterm], :(log($(xterm))^2)),
  FunctionTerm(x -> log(x)^3, [xterm], :(log($(xterm))^3)),
  FunctionTerm(log1p, [xterm], :(log1p($(xterm)))),
  # inverse (asymptotic)
  FunctionTerm(inv, [xterm], :($(xterm)^-1)),
  FunctionTerm(x -> inv(x^2), [xterm], :($(xterm)^-2)),
  FunctionTerm(x -> inv(x^3), [xterm], :($(xterm)^-3))
]

"""
    petterson(y) = 1 / √y

Transformation used in the linearized Petterson hypsometric model.
Form: 1/√H = β0 + β1 * (1/D)
"""
petterson(y::Real) = inv(√y)

"""
    naslund(y, x) = x / √y

Transformation used in the linearized Naslund hypsometric model.
Form: D/√H = β0 + β1 * D
"""
naslund(y::Real, x::Real) = x / √y

"""
    prodan(y, x) = x^2 / y

Transformation used in the linearized Prodan hypsometric model.
Form: D²/H = β0 + β1 * D + β2 * D²
"""
prodan(y::Real, x::Real) = x^2 / y

function transformyterms(yterm::AbstractTerm, xterms::Vector{<:AbstractTerm})
  ylist = AbstractTerm[
    yterm,
    FunctionTerm(log, [yterm], :(log($yterm))),
    FunctionTerm(log1p, [yterm], :(log1p($yterm))),
    FunctionTerm(inv, [yterm], :($yterm^-1)),
    FunctionTerm(sqrt, [yterm], :(√$yterm)),
    FunctionTerm(cbrt, [yterm], :(∛$yterm)),
    FunctionTerm(petterson, [yterm], :(√($yterm)^-1))
  ]
  # # add combined transformations with each x
  for xt in xterms
    push!(ylist,
      FunctionTerm(naslund, [yterm, xt], :($xt / √($yterm)))
    )
    push!(ylist,
      FunctionTerm(prodan, [yterm, xt], :(($xt)^2 / $yterm))
    )
  end

  return ylist
end

function combinationsfit(::Type{AllometricModel}, cols::NamedTuple, ylist::Vector{Tuple{AbstractTerm,Vector{Float64}}}, combinations::Vector{TermTuple}, qterms::Vector{<:AbstractTerm}, nonnegative::Bool)
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

    # Determine the number of observations (n) and the number of predictors (p)
    (n, p) = size(X)
    # Calculate the degrees of freedom for residuals
    ν = n - p
    # Allocate space for regression coefficients β
    β = Vector{Float64}(undef, p)
    # Allocate space for predicted values (ŷ) 
    ŷ = Vector{Float64}(undef, n)
    # Allocate space for residuals (ε)
    ε = Vector{Float64}(undef, n)

    try
      chol = cholesky!(Symmetric(X'X))
      invchol = inv(chol)

      for iy in 1:ny
        (yt, y) = ylist[iy]
        # β = (X'X)⁻¹ * (X'y)
        mul!(β, X', y)
        # Solve for regression coefficients β in-place using the Cholesky factor
        ldiv!(chol, β)
        # Compute predicted values in-place (ŷ = X * β)
        mul!(ŷ, X, β)
        # compute residuals
        @. ε = y - ŷ
        # sum of squared errors
        sse = ε ⋅ ε
        # residual variance
        σ² = sse / ν
        # compute dispersion matrix
        Σ = copy(invchol)
        rmul!(Σ, σ²)
        # normality check of residuals
        normality = isnormality(ε)
        # Correct the predicted values and residuals for models with a function on the left-hand side of the formula
        if isa(yt, FunctionTerm)
          # xtract interaction vector 'x' if the transformation requires it (e.g., x/sqrt(y))
          x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
          # Apply the function-specific prediction logic
          predictbiascorrected!(ŷ, x, yt, σ²)
          @. ε = yᵣ - ŷ
          sse = ε ⋅ ε
        end

        # check for nonnegative fitted values if required
        if nonnegative && any(v -> v < 0, ŷ)
          fittedmodels[iy, ix] = missing
          continue
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
          FormulaTerm(yt, rhs), cols, copy(β), Σ, σ², n, ν, p, sse, sst, r², adjr², ev, mae, mape, mse, rmse, cv, normality)
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

function regression(data, yname::S, xnames::S...; contrasts=Dict{Symbol,Any}(), model=AllometricModel, nmin::Int=1, nmax::Int=2, nonnegative::Bool=true)
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

  if nmax > 5
    throw(ArgumentError("nmax ($nmax) exceeds the practical limit for allometric models (max 5). "
                        * "Models with more than 5 terms (e.g., degree > 5) cause severe overfitting, "
                        * "numerical instability (singular matrices), and lack biological meaning."))
  end

  yname = Symbol(yname)
  xnames = Symbol.(xnames)

  # Construct Model Frame
  mf = ModelFrame(
    term(yname) ~ sum(term.(xnames)),
    data; model=model, contrasts=contrasts
  )
  # Extract components
  cols = mf.data
  yterm = mf.f.lhs

  if yterm isa CategoricalTerm
    throw(ArgumentError("dependent variable must be continuous"))
  end

  xterms = filter(t -> t isa ContinuousTerm, mf.f.rhs.terms) |> collect

  isempty(xterms) && throw(ArgumentError("no continuous independent variables provided"))

  qterms = filter(t -> t isa CategoricalTerm, mf.f.rhs.terms) |> collect

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

  combinationsfit(model, cols, ylist, combinations, qterms, nonnegative)
end

function fit(::Type{AllometricModel}, formula::FormulaTerm, data; contrasts=Dict{Symbol,Any}(), nonnegative::Bool=true)
  # Construct Model Frame (Handle missing data, categorical contrasts, etc.)
  mf = ModelFrame(formula, data; model=AllometricModel, contrasts=contrasts)
  # Extract Components from the processed ModelFrame
  formula = mf.f
  # Dependent Variable Term
  yt = formula.lhs
  # Validate supported transformations on dependent variable
  if isa(yt, FunctionTerm)
    fname = nameof(yt.f)
    if fname ∉ (:log, :log1p, :inv, :sqrt, :cbrt, :petterson, :naslund, :prodan)
      throw(ArgumentError("""
        Transformation ':$fname' on the dependent variable is not supported by AllometricModel.
        We only support transformations with defined bias correction methods.
        
        Allowed: (:log, :log1p, :inv, :sqrt, :cbrt, :petterson, :naslund, :prodan)
      """))
    end
  end
  # Data Columns
  cols = mf.data
  # the real dependent variable
  yᵣ = cols[1]
  # total sum of squares
  ȳ = mean(yᵣ)
  sst = sum(abs2, yᵣ .- ȳ)
  # Build Model Matrix (X) and Transformed Y Columns (y)
  X = modelmatrix(mf)
  y = modelcols(yt, cols)
  # Determine the number of observations (n) and the number of predictors (p)
  (n, p) = size(X)
  # Calculate the degrees of freedom for residuals
  ν = n - p
  # Allocate space for regression coefficients β
  β = Vector{Float64}(undef, p)
  # Allocate space for predicted values (ŷ) 
  ŷ = Vector{Float64}(undef, n)
  # Allocate space for residuals (ε)
  ε = Vector{Float64}(undef, n)

  try
    chol = cholesky!(Symmetric(X'X))
    invchol = inv(chol)
    # β = (X'X)⁻¹ * (X'y)
    mul!(β, X', y)
    # Solve for regression coefficients β in-place using the Cholesky factor
    ldiv!(chol, β)
    # Compute predicted values in-place (ŷ = X * β)
    mul!(ŷ, X, β)
    # compute residuals
    @. ε = y - ŷ
    # sum of squared errors
    sse = ε ⋅ ε
    # residual variance
    σ² = sse / ν
    # compute dispersion matrix
    Σ = copy(invchol)
    rmul!(Σ, σ²)
    # normality check of residuals
    normality = isnormality(ε)
    # Correct the predicted values and residuals for models with a function on the left-hand side of the formula
    if isa(yt, FunctionTerm)
      # xtract interaction vector 'x' if the transformation requires it (e.g., x/sqrt(y))
      x = length(yt.args) > 1 ? cols[yt.args[2].sym] : nothing
      # Apply the function-specific prediction logic
      predictbiascorrected!(ŷ, x, yt, σ²)
      @. ε = yᵣ - ŷ
      sse = ε ⋅ ε
    end

    # check for nonnegative fitted values if required
    if nonnegative && any(v -> v < 0, ŷ)
      return throw(ErrorException("fitted values contain negative predictions"))
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
    return AllometricModel(
      formula, cols, β, Σ, σ², n, ν, p, sse, sst, r², adjr², ev, mae, mape, mse, rmse, cv, normality)
  catch err
    throw(ErrorException("model fit failed: $(err)"))
  end
end
