const S = Union{Symbol,String}

const TermTuple = Tuple{Vararg{Tuple{AbstractTerm,Vector{Float64}}}}

"""
    const β₀ = InterceptTerm{true}()

Represents an intercept term for linear models.
"""
const β₀ = InterceptTerm{true}()

"""
    struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,T<:Float64,B<:Bool}
      
Represents a fitted linear model.

# Fields
- `formula::F`: The formula used to specify the relationship between dependent and independent variables.
- `data::N`: The data set (e.g., NamedTuple or DataFrame) containing the variables used in the model.
- `β::Array{T,1}`: The estimated regression coefficients (a vector).
- `residuals::Array{T,1}`: The residuals, representing the difference between observed and predicted values.
- `σ²::T`: The variance of residuals, indicating the variability of residuals around the fitted values.
- `r²::T`: The coefficient of determination (R²), measuring the proportion of variance explained by the model.
- `adjr²::T`: The adjusted R², adjusted for the number of predictors.
- `d::T`: The Willmott’s index of agreement, indicating how closely the predicted values match the observed values.
- `mse::T`: The mean squared error, representing the average squared residual.
- `rmse::T`: The root mean squared error, a measure of the prediction error.
- `syx::T`: The standard error of the estimate (Syx), expressed as a percentage of the mean response.
- `aic::T`: The Akaike Information Criterion, used for model comparison.
- `bic::T`: The Bayesian Information Criterion, penalizing model complexity more heavily than AIC.
- `normality::B`: Boolean flag indicating whether residuals follow a normal distribution (`true` or `false`).
- `significance::B`: Boolean flag indicating whether all coefficients are statistically significant (`true` if all p-values < 0.05).
"""
struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,T<:Float64,I<:Int64,B<:Bool} <: RegressionModel
  formula::F        # Formula
  cols::N           # Data (NamedTuple))
  β::Vector{T}      # Coefficients
  Σ::Matrix{T}      # Covariance Matrix
  σ²::T             # Residual Variance
  n::I              # Number of Observations
  ν::I              # Degrees of Freedom (Residual)
  p::I              # Number of Parameters
  sse::T            # Sum of Squared Errors
  sst::T            # Total Sum of Squares
  r²::T             # Generalized R²
  adjr²::T          # Adjusted Generalized R²
  d::T              # Willmott's Index of Agreement
  mae::T            # Mean Absolute Error
  s²ᵧₓ::T           # Mean Squared Error
  sᵧₓ::T            # Standard Error of Estimate (Absolute)
  sᵧₓpct::T         # Standard Error of Estimate (%)
  normality::B      # Shapiro-Wilk / Jarque-Bera (on ε)
end
