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
xoversqrty(y::Real, x::Real) = x / √y
xsquaredovery(y::Real, x::Real) = x^2 / y

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
  X0 = modelcols(β0, cols)

  # pre-calculate categorical column(s)
  if !isempty(qterms)
    Qsum = sum(qterms)
    Qmatrix = modelmatrix(Qsum, cols)
  end

  nx = length(combinations)
  ny = length(ylist)
  # pre-allocate output matrix
  fittedmodels = Matrix{Union{Missing,AllometricModel}}(undef, ny, nx)

  Threads.@threads for ix in 1:nx
    c = combinations[ix]

    if isempty(qterms)
      X = hcat(X0, map(last, c)...)
      rhs = MatrixTerm(mapfoldl(first, +, c; init=β0))
      ncontinuous = count(t -> !isa(t, InterceptTerm), rhs.terms)
    else
      X = hcat(X0, map(last, c)..., Qmatrix)
      rhs = MatrixTerm(mapfoldl(first, +, c; init=β0) + Qsum)
      ncontinuous = count(t -> !isa(t, InterceptTerm) && !isa(t, CategoricalTerm), rhs.terms)
    end

    # define range to check
    checkrange = 2:(1+ncontinuous)

    try
      chol = cholesky!(X'X)

      for iy in 1:ny
        (yt, y) = ylist[iy]
        # β = (X'X)⁻¹ * (X'y)
        β = X'y
        # Solve for regression coefficients β in-place using the Cholesky factor
        ldiv!(chol, β)
        # Allocate space for predicted values (ẑ) with the same structure as y
        ẑ = similar(y)
        # Compute predicted values in-place (ẑ = X * β)
        mul!(ẑ, X, β)
        # compute residuals
        ε = y - ẑ
        # Determine the number of observations (n) and the number of predictors (p)
        (n, p) = size(X)
        # Calculate the degrees of freedom for residuals
        ν = n - p
        # residual variance
        σ² = ε ⋅ ε / ν
        # compute dispersion matrix
        Σ = rmul!(inv(chol), σ²)
        # Correct the predicted values and residuals for models with a function on the left-hand side of the formula
        if isa(yt, FunctionTerm)
          # Apply the function-specific prediction logic
          # ẑ = predictBiasCorrected(cols, yt, ẑ, σ²)
          ŷ = predictbiascorrected(ẑ, cols, yt, σ²)
          εᵣ = cols[1] - ŷ
        else
          ŷ = ẑ
          εᵣ = ε
        end
        # store fitted model
        fittedmodels[iy, ix] = AllometricModel(FormulaTerm(yt, rhs), cols, β, ẑ, ε, ŷ, εᵣ, σ², Σ, n, ν)

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

  yname = Symbol(yname)
  xnames = Symbol.(xnames)

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

  ft = isempty(qterms) ? FormulaTerm(yterm, MatrixTerm(β0 + sum(xterms))) : FormulaTerm(yterm, MatrixTerm(β0 + sum(xterms) + sum(qterms)))

  cols, _ = missing_omit(cols, ft)

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

  combinationsfit(model, cols, ylist, combinations, qterms)
end
