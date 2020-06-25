mutable struct LimitedMemoryBFGS <: DescentMethod
    m
    δs
    γs
    qs
    LimitedMemoryBFGS(;m = 1, δ = [], γ = [], q = []) = new(m, δ, γ, q)
end

function init!(M::LimitedMemoryBFGS, f, ∇f, x)
    M.δs = []
    M.γs = []
    M.qs = []
    return M
end

function step!(M::LimitedMemoryBFGS, f, ∇f, x)
    δs, γs, qs, g = M.δs, M.γs, M.qs, ∇f(x)
    m = length(δs)
    if m > 0
        q=g
        for i in m : -1 : 1
            qs[i] = copy(q)
            q -= (δs[i]⋅q)/(γs[i]⋅δs[i])*γs[i]
        end
        z = (γs[m] .* δs[m] .* q) / (γs[m]⋅γs[m])
        for i in 1 : m
            z += δs[i]*(δs[i]⋅qs[i] - γs[i]⋅z)/(γs[i]⋅δs[i])
        end
        x′ = line_search(f, x, -z[1])
    else
        x′ = line_search(f, x, -g[1])
    end
    g′ = ∇f(x′)
    push!(δs, x′ - x)
    push!(γs, g′ - g)
    push!(qs, zeros(length(x)))
    while length(δs) > M.m
        popfirst!(δs)
        popfirst!(γs)
        popfirst!(qs)
    end
    return x′
end
