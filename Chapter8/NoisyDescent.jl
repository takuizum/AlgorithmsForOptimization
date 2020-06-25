mutable struct NoisyDescent <: DescentMethod
    submethod
    σ
    k
end

function init!(M::NoisyDescent, f, ∇f, x)
    init!(M.submethod, f, ∇f, x)
    M.k = 1
    return M
end

function step!(M::NoisyDescent, f, ∇f, x)
    x = step!(M.submethod, f, ∇f, x)
    σ = M.σ(M.k)
    x += σ .* randn(length(x))
    M.k += 1
    return x
end

function noisy_descent(f, ∇f, x, M::NoisyDescent)
    x₀ = copy(x)
    init!(M, f, ∇f, x₀)
    t = 0
    res = copy(x)
    while true
        t += 1
        x₁ = step!(M, f, ∇f, x₀)
        @show x₁
        if all(abs.(x₀ - x₁) .< 0.001) || t == 50
            res = [res x₁]
            return x₁, res
        else
            x₀ = copy(x₁)
            res = [res x₁]
        end
    end
end
