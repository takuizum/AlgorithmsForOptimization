using Plots

x1 = [-2:0.01:2;]
x2 = copy(x1)

plot(x1, x2, log.([Rosenbrock([x, y]) for x in x1, y in x2]'))
