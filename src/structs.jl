const S = Union{Symbol,String}

const TermTuple = Tuple{Vararg{Tuple{AbstractTerm,Vector{Float64}}}}

"""
    const β₀ = InterceptTerm{true}()

Represents an intercept term for linear models.
"""
const β₀ = InterceptTerm{true}()

"""
    struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,T<:Float64,I<:Int64,B<:Bool} <: RegressionModel

Represents a fitted allometric regression model using a lightweight ("lazy") architecture. 
Predictions and residuals vectors are not stored to conserve memory; they are recalculated on demand.

# Fields

### Model Specification & Inference
- `formula::F`: The formula specifying the relationship between dependent and independent variables.
- `cols::N`: The data columns (NamedTuple) used for fitting. Stored to allow lazy recalculation of predictions.
- `β::Vector{T}`: The estimated regression coefficients.
- `Σ::Matrix{T}`: The variance-covariance matrix of the coefficients.

### Statistical Properties (Transformed Scale)
- `σ²::T`: The residual variance (Mean Squared Error) of the linearized model. Used for bias correction (e.g., Meyer factor).
- `n::I`: The number of observations.
- `ν::I`: The residual degrees of freedom.
- `p::I`: The number of model parameters.

### Goodness-of-Fit Metrics (Original Scale)
These metrics are based on the back-transformed predictions (real units, e.g., m³), allowing valid comparison between different transformations.

- `sse::T`: The Sum of Squared Errors (Residual Sum of Squares) on the original scale.
- `sst::T`: The Total Sum of Squares on the original scale.
- `r²::T`: The Generalized Coefficient of Determination (Pseudo-R²).
- `adjr²::T`: The Adjusted Generalized R².
- `d::T`: Willmott’s Index of Agreement (0 to 1).
- `mae::T`: The Mean Absolute Error.
- `s²ᵧₓ::T`: The variance of the estimate error (Mean Squared Error on original scale).
- `sᵧₓ::T`: The Standard Error of the Estimate (Absolute). Root of `s²ᵧₓ`.
- `sᵧₓpct::T`: The Standard Error of the Estimate as a percentage of the mean response (CV of RMSE).

### Diagnostics
- `normality::B`: Boolean flag indicating if the residuals (on the transformed scale) follow a normal distribution (`true` if p > 0.05).
"""
struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,T<:Float64,I<:Int64,B<:Bool} <: RegressionModel
  formula::F        # Formula
  cols::N           # Data (NamedTuple)
  β::Vector{T}      # Coefficients
  Σ::Matrix{T}      # Covariance Matrix
  σ²::T             # Residual Variance (Transformed)
  n::I              # Number of Observations
  ν::I              # Degrees of Freedom (Residual)
  p::I              # Number of Parameters
  sse::T            # Sum of Squared Errors
  sst::T            # Total Sum of Squares
  r2::T             # Generalized R²
  adjr2::T          # Adjusted Generalized R²
  ev::T             # Explained Variance
  mae::T            # Mean Absolute Error
  mape::T           # Mean Absolute Percentage Error (%)
  mse::T            # Mean Squared Error
  rmse::T           # Root Mean Squared Error
  cv::T             # Coefficient of Variation of RMSE (%)
  normality::B      # Normality check on transformed residuals
end
