using CairoMakie
using DataFrames
using Distributions
using Statistics
using LinearAlgebra

struct RegressionDiagnosticsData
  data::DataFrame
  qx::Vector{Float64}
  qy::Vector{Float64}
  rtype::Symbol
  col_names::Vector{String}
end

function _prepare_diagnostics(model)
  df = DataFrame(model.data)
  y_pred = predict(model)
  resid = residuals(model)

  n_cols = size(df, 2)
  rtype = if n_cols == 2
    :simple
  elseif n_cols >= 3
    :multiple
  else
    :multiple
  end

  sorted_res = sort(resid)
  n = length(resid)
  p = ((1:n) .- 0.5) ./ n
  qx = quantile(Normal(0, 1), p)
  qy = sorted_res

  plot_data = copy(df)
  plot_data.pred = y_pred
  plot_data.resid = resid

  return RegressionDiagnosticsData(plot_data, qx, qy, rtype, names(df))
end

function regressionplot(model; kwargs...)
  f = Figure()
  regressionplot!(f[1, 1], model; kwargs...)
  return f
end

function regressionplot!(pos::Union{Figure,GridPosition,GridLayout}, model;
  color=:steelblue,
  markersize=8,
  linecolor=:red,
  linewidth=2,
  alpha=0.8,
  hist_color=(:gray, 0.5),
  kwargs...
)

  rd = _prepare_diagnostics(model)
  data = rd.data
  plot_color = color

  if rd.rtype == :simple
    g = GridLayout(pos)

    ax1 = Axis(g[1, 1], title="Observed vs Fitted", xlabel=rd.col_names[2], ylabel=rd.col_names[1])
    ax2 = Axis(g[1, 2], title="Residuals vs Fitted", xlabel="Predicted", ylabel="Residuals")
    ax3 = Axis(g[2, 1], title="Histogram", xlabel="Residuals", ylabel="Density")
    ax4 = Axis(g[2, 2], title="Normal Q-Q", xlabel="Theoretical", ylabel="Empirical")

    scatter!(ax1, data[!, 2], data[!, 1]; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)
    perm = sortperm(data[!, 2])
    lines!(ax1, data[perm, 2], data[perm, :pred]; color=linecolor, linewidth=linewidth)

    scatter!(ax2, data[!, :pred], data[!, :resid]; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)
    hlines!(ax2, [0], color=:black, linestyle=:dash)

    hist!(ax3, data[!, :resid]; normalization=:pdf, color=hist_color, strokewidth=1)
    d_fit = fit(Normal, data[!, :resid])
    x_rng = range(extrema(data[!, :resid])..., length=100)
    lines!(ax3, x_rng, pdf.(d_fit, x_rng); color=linecolor, linewidth=linewidth)

    scatter!(ax4, rd.qx, rd.qy; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)
    q_theo = quantile(Normal(), [0.25, 0.75])
    q_emp = quantile(rd.qy, [0.25, 0.75])
    slope = (q_emp[2] - q_emp[1]) / (q_theo[2] - q_theo[1])
    intercept = q_emp[1] - slope * q_theo[1]
    ablines!(ax4, intercept, slope; color=linecolor, linestyle=:dash)

  else
    g = GridLayout(pos)

    ax3d = Axis3(g[1:3, 1], title="3D View",
      xlabel=rd.col_names[3], ylabel=rd.col_names[2], zlabel=rd.col_names[1],
      azimuth=1.275 * pi)

    ax2 = Axis(g[1, 2], title="Resid vs Fit", xlabel="Fit", ylabel="Resid")
    ax3 = Axis(g[2, 2], title="Hist Resid", xlabel="Resid")
    ax4 = Axis(g[3, 2], title="QQ Plot", xlabel="Theo", ylabel="Emp")

    scatter!(ax3d, data[!, 3], data[!, 2], data[!, 1]; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)

    scatter!(ax2, data[!, :pred], data[!, :resid]; color=plot_color, markersize=markersize, kwargs...)
    hlines!(ax2, [0], color=:black, linestyle=:dash)

    hist!(ax3, data[!, :resid]; normalization=:pdf, color=hist_color)

    scatter!(ax4, rd.qx, rd.qy; color=plot_color, markersize=markersize, kwargs...)
    q_theo = quantile(Normal(), [0.25, 0.75])
    q_emp = quantile(rd.qy, [0.25, 0.75])
    slope = (q_emp[2] - q_emp[1]) / (q_theo[2] - q_theo[1])
    intercept = q_emp[1] - slope * q_theo[1]
    ablines!(ax4, intercept, slope; color=linecolor, linestyle=:dash)
  end

  return g
end

##########################


using CairoMakie
using DataFrames
using Distributions
using Statistics
using LinearAlgebra
using GLMakie

function regressionplot(model; kwargs...)
  f = Figure()
  regressionplot!(f[1, 1], model; kwargs...)
  return f
end

function regressionplot!(pos::Union{Figure,GridPosition,GridLayout}, model;
  color=:steelblue,
  markersize=8,
  linecolor=:red,
  linewidth=2,
  alpha=0.8,
  hist_color=(:gray, 0.5),
  kwargs...
)

  df = DataFrame(model.data)
  n_cols = size(df, 2)
  y_pred = predict(model)
  resid = residuals(model)

  rtype = if n_cols == 2
    :simple
  elseif n_cols >= 3 # Adicione verificação de Categórica aqui se necessário (ex: eltype(df[!,3]) <: CategoricalValue)
    :multiple
  else
    :multiple
  end

  sorted_res = sort(resid)
  n = length(resid)
  p = ((1:n) .- 0.5) ./ n
  qx = quantile(Normal(0, 1), p)
  qy = sorted_res

  plot_data = copy(df)
  plot_data.pred = y_pred
  plot_data.resid = resid
  col_names = names(df)
  plot_color = color

  if rtype == :simple
    g = GridLayout(pos)

    ax1 = Axis(g[1, 1], title="Observed vs Fitted", xlabel=col_names[2], ylabel=col_names[1])
    ax2 = Axis(g[1, 2], title="Residuals vs Fitted", xlabel="Predicted", ylabel="Residuals")
    ax3 = Axis(g[2, 1], title="Histogram", xlabel="Residuals", ylabel="Density")
    ax4 = Axis(g[2, 2], title="Normal Q-Q", xlabel="Theoretical", ylabel="Empirical")

    scatter!(ax1, plot_data[!, 2], plot_data[!, 1]; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)
    perm = sortperm(plot_data[!, 2])
    lines!(ax1, plot_data[perm, 2], plot_data[perm, :pred]; color=linecolor, linewidth=linewidth)

    scatter!(ax2, plot_data[!, :pred], plot_data[!, :resid]; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)
    hlines!(ax2, [0], color=:black, linestyle=:dash)

    hist!(ax3, plot_data[!, :resid]; normalization=:pdf, color=hist_color, strokewidth=1)
    d_fit = fit(Normal, plot_data[!, :resid])
    x_rng = range(extrema(plot_data[!, :resid])..., length=100)
    lines!(ax3, x_rng, pdf.(d_fit, x_rng); color=linecolor, linewidth=linewidth)

    scatter!(ax4, qx, qy; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)
    q_theo = quantile(Normal(), [0.25, 0.75])
    q_emp = quantile(qy, [0.25, 0.75])
    slope = (q_emp[2] - q_emp[1]) / (q_theo[2] - q_theo[1])
    intercept = q_emp[1] - slope * q_theo[1]
    ablines!(ax4, intercept, slope; color=linecolor, linestyle=:dash)

  else
    g = GridLayout(pos)

    ax3d = Axis3(g[1:3, 1], title="3D View",
      xlabel=col_names[3], ylabel=col_names[2], zlabel=col_names[1],
      azimuth=1.275 * pi)

    ax2 = Axis(g[1, 2], title="Resid vs Fit", xlabel="Fit", ylabel="Resid")
    ax3 = Axis(g[2, 2], title="Hist Resid", xlabel="Resid")
    ax4 = Axis(g[3, 2], title="QQ Plot", xlabel="Theo", ylabel="Emp")

    scatter!(ax3d, plot_data[!, 3], plot_data[!, 2], plot_data[!, 1]; color=plot_color, markersize=markersize, alpha=alpha, kwargs...)

    scatter!(ax2, plot_data[!, :pred], plot_data[!, :resid]; color=plot_color, markersize=markersize, kwargs...)
    hlines!(ax2, [0], color=:black, linestyle=:dash)

    hist!(ax3, plot_data[!, :resid]; normalization=:pdf, color=hist_color)

    scatter!(ax4, qx, qy; color=plot_color, markersize=markersize, kwargs...)
    q_theo = quantile(Normal(), [0.25, 0.75])
    q_emp = quantile(qy, [0.25, 0.75])
    slope = (q_emp[2] - q_emp[1]) / (q_theo[2] - q_theo[1])
    intercept = q_emp[1] - slope * q_theo[1]
    ablines!(ax4, intercept, slope; color=linecolor, linestyle=:dash)
  end

  return g
end

p1 = regressionplot(f)

f2 = fit(AllometricModel, @formula(log(Btot) ~ 1 + log(dbh) + dens), data)

p2 = regressionplot(f2)
