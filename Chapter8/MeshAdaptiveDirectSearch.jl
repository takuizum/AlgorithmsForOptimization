using LinearAlgebra
function rand_positive_spanning_set(α, n)
    δ = round(Int, 1/sqrt(α)) # Int(round(1/sqrt(α))) よりも効率的。
    L = Matrix(Diagonal(δ * rand([1, -1], n))) # 対角行列
    for i in 1:n-1, j in i+1:n
        L[i, j] = rand(-δ+1:δ-1)
    end
    @show L # Upper triangular.
    D = L[randperm(n), :]
    # D = L[:, randperm(n)]
    D = hcat(D, -sum(D, dims = 2)) # is identical to colsums() in R
    return [D[:, i] for i in 1:n+1]
end

function mesh_adaptive_direct_search(f, x, α, ϵ, γ = 0.5)
    y, n = f(x), length(x)
    while α > ϵ
        improved = false
        L = rand_positive_spanning_set(α, n) # L is an array of array, not a matrix.
        for (i, d) in enumerate(L)
            x′ = x + α*d
            y′ = f(x′)
            if y′ < y
                x, y, improved = x′, y′, ture
                x′ = x + 3α*d # the outside of the mesh
                y′ = f(x′)
                if y′ < y # もし新しい候補点を飛び越えた先が，候補点よりも最適であれば
                    x, y = x′, y′
                end
                break
            end
        end
        α = improved ? min(1, 4α) : α/4
    end
    return x
end
