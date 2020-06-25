using Optim

function line_search(f, x, d)
    obj = α -> f(x + α*d)
    a, b = bracket_minimum(obj) # Bracket intervals
    α = optimize(obj, a, b).minimizer # Brent method
    return x + a*d
end

include.(pwd() .* ["/Chapter1/Rosenbrock.jl", "/Chapter3/BracketMinimum.jl"]);
# # Univariate line search
# g(x) = x^2 + 4x
# p = 0
# line_search(g, p, 0.1)
# plot(g)
#
# Multivariate line search
p = [0.0, 0.0]
line_search(Rosenbrock, p, [0.1, 0.1])
