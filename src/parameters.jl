"""
    isnormality(x)

Tests if the input vector `x` (residuals) comes from a Normal distribution.
Returns `true` if the null hypothesis (normality) cannot be rejected (p > 0.05).

Implementation details:
- N < 3: Insufficient data (defaults to true).
- 3 ≤ N ≤ 5000: Uses **Shapiro-Wilk** (Gold standard for this range).
- N > 5000: Uses **Jarque-Bera** (Asymptotic test suitable for large samples).
"""
function isnormality(x::AbstractVector{<:Real})
  n = length(x)
  if n < 3
    return true
  elseif n <= 5000
    # Shapiro-Wilk is the most powerful test for this range
    # It tests: H0 = Data is Normal vs H1 = Data is NOT Normal
    return pvalue(ShapiroWilkTest(x)) > 0.05
  else
    # Shapiro-Wilk is not valid for N > 5000 in this package
    # Jarque-Bera uses Skewness and Kurtosis to test normality asymptotically
    return pvalue(JarqueBeraTest(x)) > 0.05
  end
end

isnormality(model::AllometricModel) = model.normality

# --- Basic Model Properties ---

nobs(model::AllometricModel) = model.n

dof_residual(model::AllometricModel) = model.ν

dof(model::AllometricModel) = length(model.β) + 1

formula(model::AllometricModel) = model.formula

# --- Coefficients & Inference ---

coef(model::AllometricModel) = model.β

coefnames(model::AllometricModel) = StatsModels.coefnames(model.formula.rhs)

vcov(model::AllometricModel) = model.Σ

stderror(model::AllometricModel) = sqrt.(diag(model.Σ))

function confint(model::AllometricModel; level::Real=0.95)
  h = 1 - (1 - level) / 2
  tdist = TDist(dof_residual(model))
  crit = quantile(tdist, h)
  se = stderror(model)
  return hcat(model.β .- crit .* se, model.β .+ crit .* se)
end

function pvalue(model::AllometricModel)
  β = coef(model)
  se = stderror(model)
  tval = β ./ se
  return 2 .* ccdf.(TDist(dof_residual(model)), abs.(tval))
end

function coeftable(model::AllometricModel; level::Real=0.95)
  cc = coef(model)
  se = stderror(model)
  tt = cc ./ se
  # use t-distribution based on residual degrees of freedom
  dist = TDist(dof_residual(model))
  # calculate p-values (two-tailed)
  pp = 2 .* ccdf.(dist, abs.(tt))
  # calculate confidence intervals
  α = 1 - level
  crit = quantile(dist, 1 - α / 2)
  lower = cc .- crit .* se
  upper = cc .+ crit .* se
  # format level string (e.g., "95")
  levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
  # coefficient table
  StatsBase.CoefTable(
    hcat(cc, se, tt, pp, lower, upper),
    ["Coef.", "Std. Error", "t", "Pr(>|t|)", "Lower $levstr%", "Upper $levstr%"],
    coefnames(model),
    4, # p-value column index
    3  # t-statistic column index
  )
end

# --- Data, Predictions & Response ---

modelmatrix(model::AllometricModel) = modelcols(model.formula.rhs, model.cols)

response(model::AllometricModel) = model.cols[1]

"""
    predictbiascorrected!(ŷ, cols, ft, σ²)

In-place version. Overwrites the input vector `ŷ` with the bias-corrected predictions on the original scale.
"""
function predictbiascorrected!(ŷ::Vector{<:Real}, xcol::Union{AbstractVector{<:Real},Nothing}, ft::FunctionTerm, σ²::Real)
  fname = nameof(ft.f)
  n = length(ŷ)

  # If no transformation, do nothing (ŷ remains unchanged)
  if fname ∉ (:log, :inv, :inversesqrt, :xoversqrty, :xsquaredovery)
    return
  end

  @inbounds @simd for i in 1:n
    z = ŷ[i]

    if fname == :log
      # log-normal (exact)
      ŷ[i] = exp(z + 0.5σ²)

    elseif fname == :inv
      # inverse (1/y)
      # g(z) = 1/z,  ∂²g = 2/z^3
      g = 1 / z
      ∂²g = 2 / (z^3)
      ŷ[i] = g + 0.5σ² * ∂²g

    elseif fname == :inversesqrt
      # inverse sqrt (1/√y)
      # g(z) = 1/z^2,  ∂²g = 6/z^4
      z² = z^2
      g = 1 / z²
      ∂²g = 6 / (z²^2)
      ŷ[i] = g + 0.5σ² * ∂²g

    elseif fname == :xoversqrty
      # x/√y
      # g(z) = x^2/z^2,  ∂²g = 6x^2/z^4
      x = xcol[i]
      x² = x^2
      z² = z^2

      g = x² / z²
      ∂²g = (6 * x²) / (z²^2)
      ŷ[i] = g + 0.5σ² * ∂²g

    elseif fname == :xsquaredovery
      # x^2/y
      # g(z) = x^2/z,  ∂²g = 2x^2/z^3
      x = xcol[i]
      x² = x^2

      g = x² / z
      ∂²g = (2 * x²) / (z^3)
      ŷ[i] = g + 0.5σ² * ∂²g
    end
  end
end

"""
    predict(model::AllometricModel)

Generates predictions from a regression model on the original scale of the dependent variable.

This function automatically handles the back-transformation of the dependent variable (`y`) if it was transformed during model fitting (e.g., `log(y)`, `1/y`). Crucially, it applies statistical bias correction to account for the non-linear transformation of the error term (Jensen's Inequality), ensuring unbiased estimates on the original scale.

# Correction Strategies

The function detects the transformation used and applies the appropriate correction:

**1. Log-Normal Correction (Exact):**
For logarithmic transformations (`log(y)`), it applies the analytical solution for the mean of a log-normal distribution (often called the Meyer or Baskerville correction):
``E[y] = \\exp(\\hat{z} + \\sigma^2/2)``

**2. Delta Method Correction (Approximate):**
For other non-linear transformations (e.g., `1/y`, `1/√y`), it employs the Second-Order Delta Method. This uses a truncated Taylor Series expansion to adjust for the curvature of the inverse function:
``E[y] \\approx g(\\hat{z}) + \\frac{1}{2}\\sigma^2 g''(\\hat{z})``

Where:
* ``\\hat{z}`` is the prediction on the transformed scale (linear predictor).
* ``g(\\cdot)`` is the inverse transformation function.
* ``\\sigma^2`` is the residual mean squared error (MSE) of the model.

# Parameters
- `model`: An `AllometricModel` object containing the fitted regression parameters, formula, and residual variance.

# Returns
- `Vector{Float64}`: The predicted values on the original scale of `y`, adjusted for transformations and corrected for bias.

# References
- Baskerville, G. L. (1972). Use of logarithmic regression in the estimation of plant biomass. *Canadian Journal of Forest Research*.
- Carroll, R. J., & Ruppert, D. (1988). *Transformation and Weighting in Regression*. Chapman and Hall.
- Miller, D. M. (1984). Reducing transformation bias in curve fitting. *The American Statistician*.

# Examples
- **Single Model Prediction:**
  ```julia
  ypred = predict(model)
  ```
"""
predict(model::AllometricModel) = predict(model, model.cols)

fitted(model::AllometricModel) = predict(model)

function predict(model::AllometricModel, data)
  if !Tables.istable(data)
    throw(ArgumentError("data must be a valid table"))
  else
    cols = columntable(data)
  end
  # Remove missing values and prepare the input data for the model
  cols, nonmissings = missing_omit(cols, formula(model).rhs)
  # Generate the model matrix (design matrix) from the input data
  X = modelmatrix(formula(model).rhs, cols)
  # Compute the predicted values: ŷ = X * β
  ŷ = X * coef(model)
  # Handle special cases where the left-hand side (lhs) is a function term
  ft = formula(model).lhs
  if isa(ft, FunctionTerm)
    x = length(ft.args) > 1 ? cols[ft.args[2].sym] : nothing
    # Apply the function-specific prediction logic
    predictbiascorrected!(ŷ, x, ft, model.σ²)
  end
  # Return predictions in the original data order, reinserting missing values
  StatsModels._return_predictions(
    Tables.materializer(data),
    ŷ,
    nonmissings,
    length(nonmissings)
  )
end

function predict!(dest::AbstractVector{<:Real}, model::AllometricModel)
  tmp = predict(model)
  resize!(dest, length(tmp))
  copyto!(dest, tmp)
  return dest
end

# --- Residuals & Deviance ---

residuals(model::AllometricModel) = response(model) - predict(model)

deviance(model::AllometricModel) = model.sse

nulldeviance(model::AllometricModel) = model.sst

# ---  Goodness of Fit (Likelihood & R²) ---

function loglikelihood(model::AllometricModel)
  n = nobs(model)
  # uses transformed residuals because Gaussian assumptions hold there
  rss = deviance(model)
  return -n / 2 * (log(2 * π) + 1 + log(rss / n))
end

function nullloglikelihood(model::AllometricModel)
  n = nobs(model)
  # uses transformed y for consistency with loglikelihood
  tss = nulldeviance(model)
  return -n / 2 * (log(2 * π) + 1 + log(tss / n))
end

r2(model::AllometricModel) = model.r2

adjr2(model::AllometricModel) = model.adjr2

dispersion(model::AllometricModel, sqr::Bool=false) = sqr ? model.mse : model.rmse

function StatsBase.cooksdistance(model::AllometricModel)
  u = residuals(model)
  mse = dispersion(model, true)
  k = dof(model) - 1
  X = modelmatrix(model)
  XtX = crossmodelmatrix(model)
  k == size(X, 2) || throw(ArgumentError("Models with collinear terms are not currently supported."))
  hii = diag(X * inv(XtX) * X')
  D = @. u^2 * (hii / (1 - hii)^2) / (k * mse)
  return D
end