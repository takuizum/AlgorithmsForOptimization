abstract type DescentMethod end

struct GradientDescent <: DescentMethod
    α
end

init!(M::GradientDescent, f, ∇f, x) = M
function step!(M::GradientDescent, f, ∇f, x)
    α, g = M.α, ∇f(x)
    return x - α*g
end

function gradient_descent(f, ∇f, x, M::GradientDescent)
    x₀ = copy(x)
    init!(M, f, ∇f, x₀)
    t = 0
    res = copy(x)
    while true
        t += 1
        x₁ = step!(M, f, ∇f, x₀)
        # @show x₁
        if all(abs.(x₀ - x₁) .< 0.001) || t == 100
            res = [res x₁]
            return x₁, res
        else
            x₀ = copy(x₁)
            res = [res x₁]
        end
    end
end
