using StatsModels
using CairoMakie
using AllometricModels
dry = (; d=[1.1, 1.1, 1.2, 1.4, 1.7, 3.5, 3.7, 3.9, 5.8, 6.9],
  g=[66, 131, 274, 28, 91, 762, 1085, 696, 1582, 3340])

reg = regression(dry, :g, :d, nmin=1, nmax=2)

best = bestmodel(reg)

gpred = predict(best)

summary(best)

f = fit(AllometricModel, @formula(log(g) ~ 1 + log(d)), dry)

fpred = predict(f)

summary(f)

scatter(dry[1], dry[2], rasterize=true, markersize=30.0)

lines!(dry[1], gpred)

lines!(dry[1], fpred)

current_figure()