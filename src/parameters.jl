"""
    predictbiascorrected!(ẑ, cols, ft, σ²)

In-place version. Overwrites `ẑ` with bias-corrected predictions.
"""
function predictBiasCorrected!(ẑ::AbstractVector{<:Real}, cols::NamedTuple, ft::FunctionTerm, σ²::Real)
  fname = nameof(ft.f)

  xcol = length(ft.args) > 1 ? cols[ft.args[2].sym] : nothing

  @inbounds for i in eachindex(ẑ)
    z = ẑ[i]

    if fname == :log
      # log-normal (exact)
      ẑ[i] = exp(z + σ² / 2)

    elseif fname == :inv
      # inverse (1/y)
      g = 1 / z
      ∂²g = 2 / (z^3)
      ẑ[i] = g + σ² / 2 * ∂²g

    elseif fname == :inversesqrt
      # inverse sqrt (1/√y)
      z² = z^2
      g = 1 / z²
      ∂²g = 6 / (z²^2)
      ẑ[i] = g + σ² / 2 * ∂²g

    elseif fname == :xoversqrty
      # x/√y
      x = xcol[i]
      x² = x^2
      z² = z^2

      g = x² / z²
      ∂²g = (6 * x²) / (z²^2)
      ẑ[i] = g + σ² / 2 * ∂²g

    elseif fname == :xsquaredovery
      # x^2/y
      x = xcol[i]
      x² = x^2

      g = x² / z
      ∂²g = (2 * x²) / (z^3)
      ẑ[i] = g + σ² / 2 * ∂²g
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
predict(model::AllometricModel) = model.ŷ

function predict(model::AllometricModel, data)
  if !Tables.istable(data)
    throw(ArgumentError("data must be a valid table"))
  else
    cols = columntable(data)
  end
  # Extract response vector (Y) and predictor matrix (X) from the model
  y, X = modelcols(model.formula, cols)
  # Number of observations
  n = length(y)
  # Allocate memory for predicted values
  ŷ = similar(Vector{Float64}, n)
  # Compute predicted values: ŷ = X * β
  # In-place multiplication for efficiency
  mul!(ŷ, X, model.β)
  # Handle special cases where the left-hand side (lhs) is a function term
  ft = model.formula.lhs
  if isa(ft, FunctionTerm)
    # Apply the function-specific prediction logic
    predictBiasCorrected!(ŷ, cols, ft, model.σ²)
  end
  # Return the vector of predicted values
  return ŷ
end

residuals(model::AllometricModel) = model.ε

deviance(model::AllometricModel) = residuals(model) ⋅ residuals(model)

function nulldeviance(model::AllometricModel)
  ft = model.formula.lhs
  if ft isa FunctionTerm
    y = model.cols[ft.args[1].sym]
  else
    y = model.cols[ft.sym]
  end
  ȳ = mean(y)
  sum(abs2.(y .- ȳ))
end

nobs(model::AllometricModel) = model.n

dof_residual(model::AllometricModel) = model.ν

r2(model::AllometricModel) = 1 - deviance(model) / nulldeviance(model)

adjr2(model::AllometricModel) = 1 - (1 - r2(model)) * (nobs(model) - 1) / dof_residual(model)

formula(model::AllometricModel) = model.formula
