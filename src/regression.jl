_independentvariable(xterm::AbstractTerm) = [
  xterm
  FunctionTerm(x -> x^2, [xterm], :($(xterm)^2))
  FunctionTerm(x -> x^3, [xterm], :($(xterm)^3))
  FunctionTerm(log, [xterm], :(log($(xterm))))
  FunctionTerm(x -> log(x)^2, [xterm], :(log($(xterm))^2))
  FunctionTerm(x -> log(x)^3, [xterm], :(log($(xterm))^3))
  FunctionTerm(x -> 1 / x, [xterm], :($(xterm)^-1))
  FunctionTerm(x -> 1 / x^2, [xterm], :($(xterm)^-2))
  FunctionTerm(x -> 1 / x^3, [xterm], :($(xterm)^-3))
]

function _expandterms(xterm::AbstractTerm...)
  # expand independent variables list
  xterms = _independentvariable.(xterm)

  # generate cartesian product forcing the type
  combinedterms = TermTuple[t for t in Iterators.product(xterms...)] |> vec

  return vcat(xterms...), combinedterms
end

function _expandterms(xterm::AbstractTerm)
  # Generate base xterms
  xterms = _independentvariable(xterm)
  n = length(xterms)

  # Initialize empty vector and reserve memory
  combinedterms = Vector{TermTuple}()

  # Reserves memory once, avoiding re-allocations
  sizehint!(combinedterms, n + (n * (n - 1)) ÷ 2 + (n * (n - 1) * (n - 2)) ÷ 6)

  # Generate combinations using push!
  # Singles
  for i in 1:n
    push!(combinedterms, (xterms[i],))
  end

  # Pairs
  for i in 1:n
    for j in (i+1):n
      push!(combinedterms, (xterms[i], xterms[j]))
    end
  end

  # Triples
  for i in 1:n
    for j in (i+1):n
      for k in (j+1):n
        push!(combinedterms, (xterms[i], xterms[j], xterms[k]))
      end
    end
  end

  return xterms, combinedterms
end

function _matrixterms(cols::NamedTuple, expandedterms::Tuple{Vector{AbstractTerm},Vector{TermTuple}};
  qterms=AbstractTerm[])

  xterms, combinedterms = expandedterms

  basecols = [β0; xterms]

  xcols = Dict{AbstractTerm,Union{Vector{Float64},Nothing}}()

  for xt in basecols
    try
      X = modelcols(xt, cols)
      xcols[xt] = X
    catch
      nothing
    end
  end
  # Calculate additional terms if qterms are provided
  if isempty(qterms)
    qsumterm = nothing
  else
    qsumterm = sum(qterms)
    qmatrix = modelmatrix(qsumterm, cols)
  end
  # Build modelmatrix using dictionary comprehension
  matrixterms = Dict{MatrixTerm,Matrix{Float64}}(
    qsumterm === nothing
    ? MatrixTerm(β0 + x) => hcat(xcols[β0], [xcols[t] for t in x]...)
    : MatrixTerm(β0 + x + qsumterm) => hcat(xcols[β0], [xcols[t] for t in x]..., qmatrix)
    for x in combinedterms
    if haskey(xcols, β0) && all(t -> haskey(xcols, t), x)
  )

  return matrixterms
end

_inverse(y::Real) = 1 / y
_inversesqrt(y::Real) = 1 / √y

function _dependentvariable(cols::NamedTuple, yterm::AbstractTerm)
  yterms = [
    yterm
    FunctionTerm(log, [yterm], :(log($(yterm))))
    FunctionTerm(_inverse, [yterm], :(($(yterm)^-1)))
    FunctionTerm(_inversesqrt, [yterm], :((√($(yterm)))^-1))
  ]

  dependentvariable = Dict{AbstractTerm,Union{Vector{Float64},Nothing}}()

  for yt in yterms
    try
      Y = modelcols(yt, cols)
      dependentvariable[yt] = Y
    catch
      nothing
    end
  end
  return dependentvariable
end

function _buildterms(cols::NamedTuple, y::S, x::Vector{S}, modfiers::Vector)
  # Dependent term
  yterm = concrete_term(term(Symbol(y)), cols, ContinuousTerm)

  # Continuous predictors
  xterms = [concrete_term(term(Symbol(xi)), cols, ContinuousTerm) for xi in x]

  # Parse categorical and random effects
  qterms = AbstractTerm[]
  rterms = AbstractTerm[]

  for mf in modfiers
    if mf isa S
      push!(qterms, concrete_term(term(Symbol(mf)), cols, CategoricalTerm))

    elseif mf isa Tuple
      if length(mf) == 1
        s = mf[1]
        t = concrete_term(term(Symbol(s)), cols, Grouping())
        push!(rterms, RandomEffectsTerm(β0, t))

      elseif length(mf) == 2
        s1 = mf[1]
        s2 = mf[2]

        iterm = InteractionTerm((
          concrete_term(term(Symbol(s1)), cols, StatsModels.FullDummyCoding()),
          concrete_term(term(Symbol(s2)), cols, Grouping()),
        ))

        push!(rterms, RandomEffectsTerm(β0, iterm))

      else
        error("Random effects Tuples must have 1 or 2 elements. Found: $arg")
      end
    end
  end

  return (yterm, xterms, qterms, rterms, cols)
end

function regression(data, y::S, x::S...; modfiers=[])
  # validate modifiers
  for mf in modfiers
    if !(mf isa S || mf isa Tuple{S} || mf isa Tuple{S,S})
      error("invalid argument in modfiers: $mf")
    end
  end
  if !Tables.istable(data)
    error("data must be a table")
  else
    cols = columntable(data)
  end

  # generate all model terms
  yterm, xterm, qterm, rterm, cols = _buildterms(cols, y, S[x...], modfiers)

  # build y and x structures
  yterms = _dependentvariable(cols, yterm)
  xterms = _matrixterms(cols, _expandterms(xterm...), qterms=qterm)

  ny = length(yterms)
  nx = length(xterms)

  ylist = collect(yterms)
  xlist = collect(xterms)

  # GLM path (no random effects)
  if isempty(rterm)
    fittedmodels = Matrix{Union{StatsModels.TableRegressionModel,Missing}}(undef, ny, nx)

    Threads.@threads for iy in 1:ny
      (yt, Y) = ylist[iy]
      if Y === nothing
        continue
      end
      for ix in 1:nx
        (xt, X) = xlist[ix]
        try
          fittedmodel = fit(LinearModel, X, Y)
          if !all(isfinite, stderror(fittedmodel))
            # fittedmodel has non-finite standard errors (indicating rank-deficiency or instability)
            fittedmodels[iy, ix] = missing
          else
            f = FormulaTerm(yt, xt)
            mf = ModelFrame(f, nullschema, cols, LinearModel)
            mm = ModelMatrix(X, StatsModels.asgn(xt))
            fittedmodels[iy, ix] = StatsModels.TableRegressionModel(fittedmodel, mf, mm)
          end
        catch
          fittedmodels[iy, ix] = missing
        end
      end
    end

    fittedmodels = collect(skipmissing(vec(fittedmodels)))
    isempty(fittedmodels) && error("failed to fit all Linear Regression Models")

    return fittedmodels
  end

  # LMM path (has random effects)
  remats = [modelmatrix(rt, cols) for rt in rterm]
  rsumterm = sum(rterm)

  fittedmodels = Matrix{Union{LinearMixedModel,Missing}}(undef, ny, nx)

  Threads.@threads for iy in 1:ny
    (yt, Y) = ylist[iy]
    if Y === nothing
      continue
    end
    for ix in 1:nx
      (xt, X) = xlist[ix]
      try
        Logging.with_logger(Logging.NullLogger()) do
          f = FormulaTerm(yt, xt + rsumterm)
          fittedmodel = LinearMixedModel(Y, (X, remats...), f, [], nothing, true)
          fit!(fittedmodel; REML=true, progress=false)
          if !all(isfinite, stderror(fittedmodel))
            # fittedmodel has non-finite standard errors (indicating rank-deficiency or instability)
            fittedmodels[iy, ix] = missing
          else
            fittedmodels[iy, ix] = fittedmodel
          end
        end
      catch
        fittedmodels[iy, ix] = missing
      end
    end
  end

  fittedmodels = collect(skipmissing(vec(fittedmodels)))
  isempty(fittedmodels) && error("failed to fit all Mixed Linear Regression Models")

  return fittedmodels
end

