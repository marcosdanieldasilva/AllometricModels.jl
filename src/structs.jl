const S = Union{Symbol,String}

const TermTuple = Tuple{Vararg{Tuple{AbstractTerm,Vector{Float64}}}}

"""
    const β0 = InterceptTerm{true}()

Represents an intercept term for linear models.
"""
const β0 = InterceptTerm{true}()

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
struct AllometricModel{F<:FormulaTerm,N<:NamedTuple,T<:Float64,I<:Int64} <: RegressionModel
  formula::F
  cols::N
  β::Array{T,1}
  ŷ::Array{T,1}
  ε::Array{T,1}
  σ²::T
  n::I
  ν::I
end

"""
    LinearModel(formula::F, data::N, β::Array{T,1}, residuals::Array{T,1}, σ²::T, r²::T, adjr²::T, d::T, mse::T, rmse::T, mae::T, syx::T, aic::T, bic::T, normality::B, significance::B)

    Creates a new `LinearModel` instance.

# Arguments
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
- `mse::T`: The mean absolute error as the average absolute residual value.
- `syx::T`: The standard error of the estimate (Syx), expressed as a percentage of the mean response.
- `aic::T`: The Akaike Information Criterion, used for model comparison.
- `bic::T`: The Bayesian Information Criterion, penalizing model complexity more heavily than AIC.
- `normality::B`: Boolean flag indicating whether residuals follow a normal distribution (`true` or `false`).
- `significance::B`: Boolean flag indicating whether all coefficients are statistically significant (`true` if all p-values < 0.05).

# Returns
- `LinearModel{F, N, T, B}`: A new fitted linear model instance, encapsulating all model parameters and metrics.

# Type Parameters
- `F`: The type of the formula term (e.g., `FormulaTerm`).
- `N`: The type of the data structure (e.g., `NamedTuple`).
- `T`: The numeric type used for calculations (e.g., `Float64`).
- `B`: The boolean type (`Bool`).

"""
function LinearModel(
  formula::F,
  data::N,
  β::Array{T,1},
  residuals::Array{T,1},
  σ²::T,
  r²::T,
  adjr²::T,
  d::T,
  mse::T,
  rmse::T,
  mae::T,
  syx::T,
  aic::T,
  bic::T,
  normality::B,
  significance::B
) where {F,N,T,B}
  return LinearModel{F,N,T,B}(
    formula,
    data,
    β,
    residuals,
    σ²,
    r²,
    adjr²,
    d,
    mse,
    rmse,
    mae,
    syx,
    aic,
    bic,
    normality,
    significance
  )
end
