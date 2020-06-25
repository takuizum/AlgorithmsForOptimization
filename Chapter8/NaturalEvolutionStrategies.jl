using Distributions
function natural_evolution_strategies(f, θ, k_max; m=100, α=0.01)
    for k in 1 : k_max
        samples = [rand(θ) for i in 1 : m]
        θ -= α*sum(f(x)*∇logp(x, θ) for x in samples)/m
    end
    return θ
end

# Gaussian Version
function natural_evolution_strategies(f, μ, Σ, k_max; m=100, α=0.01)
    ∇μlogp(x, θ, Σ) = inv(Σ) * (x - μ)
    ∇Σlogp(x, θ, Σ) = 1/2 * inv(Σ) * (x - μ) * (x - μ)' * inv(Σ) - 1/2 * inv(Σ)
    ∇Alogp(x, θ, Σ, A) = A * (∇Σlogp(x, θ, Σ) + ∇Σlogp(x, θ, Σ)')
    A = sqrt(Σ)
    for k in 1 : k_max
        samples = [rand(MvNormal(μ, Σ)) for i in 1:m]
        μ -= α*sum(f(x)*∇μlogp(x, μ, Σ) for x in samples)/m
        # Σ -= α*sum(f(x)*∇Σlogp(x, μ, Σ) for x in samples)/m
        A -= α*sum(f(x)*∇Alogp(x, μ, Σ, A) for x in samples)/m
        Σ = A'A
    end
    return μ, Σ
end
