f(x) = x[1]^2 - x[2]^2
∇f(x) = [2x[1], -2x[2]]

using Plots
x = [-4:0.1:4;]
y = copy(x)
z = [f([i, j]) for i in x, j in y]'

plot(x, y, z, st=:surface)


# Descent Method
include(pwd() * "/Chapter4/LineSearch.jl");
include(pwd() * "/Chapter5/DescentMethod.jl");

opt, history = gradient_descent(f, ∇f, [2, 0], GradientDescent(0.1))

plot(x, y, z)
plot!(history[1, :], history[2, :], c = :black, label = "Gradient Descent")


# Noisy Descent Method
include(pwd() * "/Chapter8/NoisyDescent.jl")
g(x) = 1/x;
using Random;
Random.seed!(02040204);
opt, history = noisy_descent(f, ∇f, [2, 0], NoisyDescent(GradientDescent(0.1), g, 1));
plot!(history[1, 1:10], history[2, 1:10], c = :blue, label = "Noisy Descent")

# Mesh Adaptive Direct Method
rand_positive_spanning_set(0.01, 2)

# SANN

iter = [10^0:1:10^4;]
plot(title = "Schedule")
plot!(iter, log_schedule.(iter), label = "logarithmic", xaxis=:log)
plot!(iter, exp_schedule.(iter, 1/4), label = "exponential, γ = 1/4", xaxis=:log)
plot!(iter, fast_schedule.(iter), label = "fast", xaxis=:log)

include(pwd() * "/Chapter8/SimulatedAnnealing")
include(pwd() * "/Chapter8/Ackely.jl")

using Distributions, LinearAlgebra
opt, history = simulated_annealing(ackley, [15, 15], MvNormal([0, 0], 25), x -> exp_schedule(x, 1/4), 10000)
x, y = [-15:0.1:15;], [-15:0.1:15;]
z = [ackley([i, j]) for i in x, j in y]'
plot(x, y, z, st = :surface)
plot!(history[:, 1], history[:, 2], [ackley(history[k, :]) for k in 1:1000], c = :green, label = "")

# opt, history = simulated_annealing(ackley, 15, Normal(0, 2), exp_schedule, 10000)
# plot(x, ackley.(x), xlims = (-4, 4))
# histogram!(history, xlims = (-4, 4), normalize = true)


# Natual Evolution Strategies
include(pwd() * "/Chapter8/NaturalEvolutionStrategies.jl")
include(pwd() * "/TestFunctions/Wheeler.jl")
x = [-2:0.1:2;]
y = copy(x)
z = [wheeler([i, j]) for i in x, j in y]'
plot(x, y, z)
μ, Σ = natural_evolution_strategies(wheeler, [1.0, 1.0], [3.0 0.0; 0.0 3.0], 1000)
