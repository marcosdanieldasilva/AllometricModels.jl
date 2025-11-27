"""
    predictBiasCorrected(cols, ft, ẑ, σ²)

Computes bias-corrected predictions on the original scale, accounting for the non-linear transformation of the error term (Jensen's Inequality).

### Correction Strategies
**Log-Normal (Exact):**
For logarithmic transformations, applies the analytical solution for the mean of a log-normal distribution:
``E[y] = \\exp(\\hat{z} + \\sigma^2/2)``

**Delta Method (Approximate):**
For other non-linear transformations (e.g., inverse, inverse square root), employs the Second-Order Delta Method. This uses a truncated Taylor Series expansion to adjust for the curvature of the inverse function:
``E[y] \\approx g(\\hat{z}) + \\frac{1}{2}\\sigma^2 g''(\\hat{z})``

# Arguments
- `cols`: NamedTuple containing original data (required for interaction terms).
- `ft`: The `FunctionTerm` from the model formula.
- `ẑ`: Vector of predictions on the transformed scale (linear predictor).
- `σ²`: The residual mean squared error (MSE) of the model.

# References
- Baskerville, G. L. (1972). Use of logarithmic regression in the estimation of plant biomass. *Canadian Journal of Forest Research*.
- Carroll, R. J., & Ruppert, D. (1988). *Transformation and Weighting in Regression*. Chapman and Hall.
"""
function predictBiasCorrected(cols::NamedTuple, ft::FunctionTerm, ẑ::Vector{<:Real}, σ²::Real)
  fname = nameof(ft.f)
  n = length(ẑ)
  ŷ = similar(ẑ)

  # Pre-fetch interaction column if required by the transformation
  hasX = length(ft.args) > 1
  xcol = hasX ? cols[ft.args[1].sym] : nothing

  @inbounds for i in 1:n
    z = ẑ[i]

    # Exact analytical solution for Log-Normal distribution
    if fname == :log
      ŷ[i] = exp(z + 0.5σ²)
      continue
    end

    g = 0.0
    ∂²g = 0.0

    # Compute inverse function g(z) and curvature ∂²g(z)
    if fname == :inv
      g = 1 / z
      ∂²g = 2 / (z^3)

    elseif fname == :inversesqrt
      g = 1 / (z^2)
      ∂²g = 6 / (z^4)

    elseif fname == :xoversqrty
      x = xcol[i]
      x² = abs2(x)
      g = x² / (z^2)
      ∂²g = 6x² / (z^4)

    elseif fname == :xsquaredovery
      x = xcol[i]
      x² = abs2(x)
      g = x² / z
      ∂²g = 2x² / (z^3)

    else
      ŷ[i] = z
      continue
    end

    # Second-order Delta Method approximation
    ŷ[i] = g + 0.5σ² * ∂²g
  end

  return ŷ
end

"""
    predict(model::AllometricModel)
  
The `predict` function family provides a versatile way to generate predictions from regression models, 
  supporting both individual and grouped models. It handles predictions on the original scale even if the
   dependent variable (`y`) has been transformed (e.g., `log(y)`), ensuring that any transformations 
   applied during model fitting are correctly reversed, including the application of Meyer correction 
   factors for logarithmic transformations.

# Parameters:
- `model`: 
    The regression model(s) to be evaluated and compared. This parameter can accept:
    - `AllometricModel`: A single linear regression model.

# Returns:
- `Vector{<:Real}` or `Vector{Union{Missing, <:Real}}`: The predicted values on the original scale of `y`, adjusted for any transformations and corrected using the Meyer factor for logarithmic transformations.

# Key Features:
- **Handles Transformed Dependent Variables:** If the dependent variable was transformed (e.g., using log transformations), the function correctly inverts the transformation to return predictions on the original scale.
- **Applies Meyer Correction Factor:** For models using logarithmic transformations, the Meyer correction factor is applied to the predictions to correct for the bias introduced by the log transformation.

# Examples:
- **Single Model Prediction:**
  ```julia
  y_pred = predict(model)
  ```
"""
predict(model::AllometricModel) = model.ŷ

function predict(model::AllometricModel, data)
  cols = columntable(data)
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
    ŷ .= predictBiasCorrected(cols, ft, ŷ, model.σ²)
  end
  # Return the vector of predicted values
  return ŷ
end

residuals(model::AllometricModel) = model.ε

deviance(model::AllometricModel) = residuals(model) ⋅ residuals(model)

function nulldeviance(model::AllometricModel)
  ft = model.formula.lhs
  if ft isa FunctionTerm
    hasX = length(ft.args) > 1
    y = hasX ? model.cols[ft.args[2].sym] : model.cols[ft.args[1].sym]
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
