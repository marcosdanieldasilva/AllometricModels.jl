"""
    calculatescore(models, criteria)

Internal helper to calculate a ranking score.
Lower score = Better model.
"""
function calculatescore(models::Vector{AllometricModel}, criteria::Vector{Symbol})
  n = length(models)
  totalscore = zeros(Float64, n)

  for crit in criteria
    # 1. Extract values
    values = [getfield(m, crit) for m in models]

    # 2. Determine ranking direction
    # Higher is better: r2, adjr2, d, normality, ev
    # Lower is better: mae, mape, mse, rmse, cv, aic, bic
    higherisbetter = crit in (:r2, :adjr2, :d, :normality, :ev)

    # 3. Calculate ranks
    ranks = tiedrank(values)

    if higherisbetter
      # Invert rank: Highest value gets Rank 1 (Best score)
      ranks .= (n + 1) .- ranks
    end

    totalscore .+= ranks
  end

  return totalscore
end

"""
    criteriatable(models, criteria...; best=10)

Evaluates and ranks regression models. If `:normality` is included, it acts as a strict filter.
Returns a lightweight table (Vector of NamedTuples).
"""
function criteriatable(models::Vector{AllometricModel}, criteria::Symbol...; best::Int=10)

  # 1. Define allowed fields
  allowed = [:r2, :adjr2, :ev, :mae, :mape, :mse, :rmse, :cv, :normality]

  # 2. Validate criteria
  selected = if isempty(criteria)
    [:adjr2, :cv, :ev] # Default
  elseif :all in criteria
    allowed
  else
    collect(criteria)
  end

  if !issubset(selected, allowed)
    throw(ArgumentError("invalid criteria used. allowed: $allowed"))
  end

  # --- 3. HARD FILTER: Normality ---
  # We must work on a subset of models if normality is requested

  current_models = models

  if :normality in selected
    # Filter models where normality == true
    current_models = filter(m -> m.normality, models)

    # Validation: Did any model survive?
    if isempty(current_models)
      throw(ErrorException("No regression models passed the Normality test (Shapiro-Wilk/Jarque-Bera). Relax the criteria by removing :normality to see the best non-normal models."))
    end

    # Remove :normality from the ranking list (since they are all true now)
    # but keep it in 'selected' for display in the table later if desired
  end

  # 4. Calculate scores on the filtered list
  scores = calculatescore(current_models, selected)

  # 5. Sort models
  sortedindices = sortperm(scores)
  limit = min(best, length(current_models))
  topindices = sortedindices[1:limit]

  # 6. Build output table
  result = map(topindices) do i
    m = current_models[i] # Use the filtered list!

    # Extract metrics
    metricpairs = (k => getfield(m, k) for k in selected)

    # Construct row
    (; model=m, rank=scores[i], metricpairs...)
  end

  return result
end

"""
    bestmodel(models, criteria...)

Returns the single best `AllometricModel`.
"""
function bestmodel(models::Vector{AllometricModel}, criteria::Symbol...)
  # Reuse table logic to handle the filtering correctly
  # Get the top 1 model
  table = criteriatable(models, criteria...; best=1)

  # We need to find the actual model object corresponding to the winner.
  # The table gives us the formula string, but recreating it is hard.
  # It is safer/faster to re-run the filter logic here.

  selected = isempty(criteria) ? [:adjr2, :cv] : collect(criteria)

  current_models = models
  if :normality in selected
    current_models = filter(m -> m.normality, models)
    if isempty(current_models)
      throw(ErrorException("No regression models passed the Normality test."))
    end
  end

  scores = calculatescore(current_models, selected)
  bestidx = argmin(scores)

  return current_models[bestidx]
end
