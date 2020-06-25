# Don't forget activate env

using Zygote
using LinearAlgebra
using Plots

include(pwd()*"/Chapter1/Rosenbrock.jl")
include(pwd()*"/Chapter3/BraketMinimum.jl")
include(pwd()*"/Chapter4/LineSearch.jl")
include(pwd()*"/Chapter5/DescentMethod.jl")
include(pwd()*"/Chapter6/LimitedMemoryBFGS.jl")


x = [0., 0.]
step!(LimitedMemoryBFGS(), Rosenbrock, Rosenbrock', [-2.0, -2.0])

function minimize(f, f′, inits; method = LimitedMemoryBFGS())
    if length(inits) == 1
        x1 = [(inits-2):0.01:(inits+2);]
        p = plot(x1, f.(x1))
    elseif length(inits) == 2
        x1 = [(inits[1]-2):0.01:(inits[1]+2);]
        x2 = [(inits[2]-2):0.01:(inits[2]+2);]
        p = plot(x1, x2, log.([f([x, y]) for x in x1, y in x2]'))
        p = scatter!(inits)
    else
        stop("More than 3 variate function was not acceptable.")
    end
    x = copy(inits)
    while true
        x′ = step!(method, f, f', x)
        p = scatter!(x)
        @show x, x′
        if x ≈ x′
            return p
        end
        x = copy(x′)
    end
end

minimize(Rosenbrock, Rosenbrock', [-0.0, -0.0])

[0.0, 0.1] ≈ [0.0, 0.0]
