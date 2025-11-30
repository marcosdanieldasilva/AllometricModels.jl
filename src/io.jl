import Base: show, summary

_coefnames(model::AllometricModel) = vcat("β0", string.(StatsModels.coefnames(model.formula.rhs.terms[2:end])))

# Display the equation of the fitted linear model.
function show(io::IO, model::AllometricModel)
  β = model.β
  n = length(β)
  output = string(StatsModels.coefnames(model.formula.lhs)) * " = $(round(β[1], digits = 4))"

  for i in 2:n
    term = _coefnames(model)[i]
    product = string(round(abs(β[i]), sigdigits=4)) * " * " * term
    output *= signbit(β[i]) ? " - $(product)" : " + $(product)"
  end

  print(io, output)
end

"""
    summary(io, model)

Prints a detailed statistical report including the equation, coefficients table, 
goodness-of-fit metrics, and diagnostic checks.
"""
function summary(io::IO, model::AllometricModel)
  println(io, "Allometric Regression Model")
  println(io, "------------------------------------------------")
  # Equation (Reuses your show function)
  print(io, "Equation: ")
  show(io, model)
  println(io, "\n") # Force double newline for spacing
  # Coefficients Table
  println(io, "Coefficients:")
  # Using 'show' on the CoefTable object ensures proper formatting (alignment/lines)
  show(io, coeftable(model))
  println(io, "\n")
  # Goodness-of-Fit Metrics
  println(io, "Goodness-of-Fit:")
  @printf(io, "  R² (Generalized):   %.4f\n", model.r2)
  @printf(io, "  Adj R²:             %.4f\n", model.adjr2)
  @printf(io, "  Explained Var (EV): %.4f\n", model.ev)
  @printf(io, "  RMSE (Syx):         %.4f\n", model.rmse)
  @printf(io, "  CV %%:               %.2f%%\n", model.cv)
  @printf(io, "  MSE:                %.4f\n", model.mse)
  @printf(io, "  MAE:                %.4f\n", model.mae)
  @printf(io, "  MAPE:               %.2f%%\n", model.mape)
  println(io, "")
  # Diagnostics
  println(io, "Diagnostics:")
  normstatus = model.normality ? "Pass (Normal)" : "Fail (Non-Normal)"
  println(io, "  Normality (Shapiro): ", normstatus)
  println(io, "------------------------------------------------")
end

# Convenience method to print to stdout if IO is not provided
summary(model::AllometricModel) = summary(stdout, model)

"""
    metrics(model)

Returns a Column Table (NamedTuple of Vectors) containing the model equation and goodness-of-fit statistics.
"""
function metrics(model::AllometricModel)
  return (
    equation=[sprint(show, model)],
    r2=[model.r2],
    adjr2=[model.adjr2],
    ev=[model.ev],
    rmse=[model.rmse],
    cv=[model.cv],
    mse=[model.mse],
    mae=[model.mae],
    mape=[model.mape],
    sse=[model.sse],
    sst=[model.sst],
    n=[model.n],
    p=[model.p],
    dof=[model.ν],
    normality=[model.normality]
  )
end

"""
    metrics(models::Vector{AllometricModel})

Returns a Column Table containing statistics for multiple models.
Efficiently constructs vectors for all metrics.
"""
function metrics(models::Vector{<:AllometricModel})
  n = length(models)
  # Pre-allocate output vectors
  eq = Vector{String}(undef, n)
  r2_vec = Vector{Float64}(undef, n)
  adjr2_vec = Vector{Float64}(undef, n)
  ev_vec = Vector{Float64}(undef, n)
  rmse_vec = Vector{Float64}(undef, n)
  cv_vec = Vector{Float64}(undef, n)
  mse_vec = Vector{Float64}(undef, n)
  mae_vec = Vector{Float64}(undef, n)
  mape_vec = Vector{Float64}(undef, n)
  sse_vec = Vector{Float64}(undef, n)
  sst_vec = Vector{Float64}(undef, n)
  n_vec = Vector{Int}(undef, n)
  p_vec = Vector{Int}(undef, n)
  dof_vec = Vector{Int}(undef, n)
  norm_vec = Vector{Bool}(undef, n)

  @inbounds for i in 1:n
    m = models[i]
    eq[i] = sprint(show, m)
    r2_vec[i] = m.r2
    adjr2_vec[i] = m.adjr2
    ev_vec[i] = m.ev
    rmse_vec[i] = m.rmse
    cv_vec[i] = m.cv
    mse_vec[i] = m.mse
    mae_vec[i] = m.mae
    mape_vec[i] = m.mape
    sse_vec[i] = m.sse
    sst_vec[i] = m.sst
    n_vec[i] = m.n
    p_vec[i] = m.p
    dof_vec[i] = m.ν
    norm_vec[i] = m.normality
  end

  return (
    equation=eq,
    r2=r2_vec,
    adjr2=adjr2_vec,
    ev=ev_vec,
    rmse=rmse_vec,
    cv=cv_vec,
    mse=mse_vec,
    mae=mae_vec,
    mape=mape_vec,
    sse=sse_vec,
    sst=sst_vec,
    n=n_vec,
    p=p_vec,
    dof=dof_vec,
    normality=norm_vec
  )
end