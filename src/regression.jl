transformXTerm(xterm::AbstractTerm) = AbstractTerm[
  xterm
  FunctionTerm(x -> x^2, [xterm], :($(xterm)^2))
  FunctionTerm(x -> x^3, [xterm], :($(xterm)^3))
  FunctionTerm(log, [xterm], :(log($(xterm))))
  FunctionTerm(x -> log(x)^2, [xterm], :(log($(xterm))^2))
  FunctionTerm(x -> log(x)^3, [xterm], :(log($(xterm))^3))
  FunctionTerm(inv, [xterm], :($(xterm)^-1))
  FunctionTerm(x -> 1 / x^2, [xterm], :($(xterm)^-2))
  FunctionTerm(x -> 1 / x^3, [xterm], :($(xterm)^-3))
]

inversesqrt(y::Real) = 1 / √y
xoversqrty(x::Real, y::Real) = x / √y
xsquaredovery(x::Real, y::Real) = x^2 / y

function transformYTerm(yterm::AbstractTerm, xterms::Vector{<:AbstractTerm})
  ylist = AbstractTerm[
    yterm
    FunctionTerm(log, [yterm], :(log($yterm)))
    FunctionTerm(inv, [yterm], :($yterm^-1))
    FunctionTerm(inversesqrt, [yterm], :(√($yterm)^-1))
  ]
  # add combined transformations with each x
  for xt in xterms
    push!(ylist,
      FunctionTerm(xoversqrty, [xt, yterm], :($xt / √($yterm)))
    )
    push!(ylist,
      FunctionTerm(xsquaredovery, [xt, yterm], :(($xt)^2 / $yterm))
    )
  end

  return ylist
end

function combinationsfit(::Type{AllometricModel}, y::Vector{<:Float64}, cols::NamedTuple, ylist::Vector{Tuple{AbstractTerm,Vector{Float64}}}, combinations::Vector{TermTuple}, qterms::Vector{<:AbstractTerm})
  # pre-calculate intercept column (X0)
  X0 = modelcols(β0, cols)

  nx = length(combinations)
  ny = length(ylist)
  fittedmodels = Matrix{Union{Missing,AllometricModel}}(undef, ny, nx)

  # Regression Loop (Split Path for Optimization)
  if isempty(qterms)
    # PATH A: Continuous Variables Only
    Threads.@threads for ix in 1:nx
      c = combinations[ix]

      # construct X: Intercept + Continuous Columns
      X = hcat(X0, map(last, c)...)

      # try
      XtX = X' * X
      chol = cholesky!(Symmetric(XtX))

      for iy in 1:ny
        (yt, Y) = ylist[iy]
        Xty = X' * Y
        β = ldiv!(chol, Xty)

        ŷ = similar(Y)
        mul!(ŷ, X, β)
        # compute residuals and sum of squared residuals
        ε = Y - ŷ
        SSR = ε ⋅ ε

        # nobs ands degrees of freedom
        n, p = size(X)
        ν = n - p

        # residual variance
        σ² = SSR / ν

        # Correct the predicted values and residuals for models with a function on the left-hand side of the formula
        if isa(yt, FunctionTerm)
          # Apply the function-specific prediction logic
          ŷ .= predictBiasCorrected(cols, yt, ŷ, σ²)
          ε .= y - ŷ
        end

        # formula: y ~ β0 + continuous_terms
        rhs = MatrixTerm(mapfoldl(first, +, c; init=β0))
        ft = FormulaTerm(yt, rhs)

        fittedmodels[iy, ix] = AllometricModel(ft, cols, β, ŷ, ε, σ², n, ν)

      end
      # catch
      #   for iy in 1:ny
      #     fittedmodels[iy, ix] = missing
      #   end
      # end
    end

  else
    # Continuous + Categorical Variables
    # Pre-calculate Q matrix and sum term once
    Qsum = sum(qterms)
    Qmatrix = modelmatrix(Qsum, cols)

    Threads.@threads for ix in 1:nx
      c = combinations[ix]

      # construct X: Intercept + Continuous Columns + Q Matrix
      X = hcat(X0, map(last, c)..., Qmatrix)

      try
        XtX = X' * X
        chol = cholesky!(Symmetric(XtX))

        for iy in 1:ny
          (yt, Y) = ylist[iy]
          Xty = X' * Y
          β = ldiv!(chol, Xty)

          ŷ = similar(Y)
          mul!(ŷ, X, β)

          # compute residuals and sum of squared residuals
          ε = Y - ŷ
          SSR = ε ⋅ ε

          # nobs ands degrees of freedom
          n, p = size(X)
          ν = n - p

          # residual variance
          σ² = SSR / ν

          # Correct the predicted values and residuals for models with a function on the left-hand side of the formula
          if isa(yt, FunctionTerm)
            # Apply the function-specific prediction logic
            ŷ .= predictBiasCorrected(cols, yt, ŷ, σ²)
            ε .= y - ŷ
          end

          # formula: y ~ β0 + continuous_terms + q_terms
          rhs = MatrixTerm(mapfoldl(first, +, c; init=β0) + Qsum)
          ft = FormulaTerm(yt, rhs)

          fittedmodels[iy, ix] = AllometricModel(ft, cols, β, ŷ, ε, σ², n, ν)
        end
      catch
        for iy in 1:ny
          fittedmodels[iy, ix] = missing
        end
      end
    end
  end

  # # flatten and remove missing fits
  fittedmodels = collect(skipmissing(vec(fittedmodels)))
  isempty(fittedmodels) && error("failed to fit all Linear Regression Models")

  return fittedmodels

end

function regression(data, yname::S, xnames::S...; hints=Dict{Symbol,Any}(), model=AllometricModel)
  # Input Validation
  if isempty(xnames)
    throw(ArgumentError("no independent variables provided"))
  end

  if !Tables.istable(data)
    throw(ArgumentError("data must be a valid table"))
  else
    cols = columntable(data)
  end

  # Term Preparation
  yterm = concrete_term(term(yname), cols, ContinuousTerm)

  # apply schema to x terms
  xschema = apply_schema(term.(xnames), StatsModels.schema(cols, hints))
  xterms = xschema isa AbstractTerm ? AbstractTerm[xschema] : collect(AbstractTerm, xschema)

  # separate categorical (q) and continuous (x) terms
  qterms = filter(t -> t isa CategoricalTerm, xterms)
  filter!(t -> !(t isa CategoricalTerm), xterms)

  if isempty(xterms)
    throw(ArgumentError("no continuous variables found"))
  end

  ft = isempty(qterms) ? FormulaTerm(yterm, MatrixTerm(β0 + sum(xterms))) : FormulaTerm(yterm, MatrixTerm(β0 + sum(xterms) + sum(qterms)))

  cols, _ = StatsModels.missing_omit(cols, ft)

  # Build Transformation Groups (Continuous)
  termgroups = Vector{Vector{Tuple{AbstractTerm,Vector{Float64}}}}()
  sizehint!(termgroups, length(xterms))

  for var in xterms
    subgroup = Tuple{AbstractTerm,Vector{Float64}}[]

    for t in transformXTerm(var)
      try
        col = modelcols(t, cols)
        push!(subgroup, (t, col))
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
    TermTuple[Tuple(c) for c in powerset(termgroups[1], 1, 3)]
  else
    TermTuple[Tuple(g) for g in Iterators.product(termgroups...)] |> vec
  end

  # Build y List
  ylist = Vector{Tuple{AbstractTerm,Vector{Float64}}}()

  for t in transformYTerm(yterm, xterms)
    try
      v = modelcols(t, cols)
      push!(ylist, (t, v))
    catch
      # ignore invalid transforms
    end
  end

  combinationsfit(model, collect(cols[yterm.sym]), cols, ylist, combinations, qterms)
end
