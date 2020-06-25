function corana_update!(v, a, c, ns)
    for i in 1 : length(v)
        ai, ci = a[i], c[i]
        if ai > 0.6ns
            v[i] *= (1 + ci*(ai/ns - 0.6)/0.4)
        elseif ai < 0.4ns
            v[i] /= (1 + ci*(0.4-ai/ns)/0.4)
        end
    end
    return v
end

function adaptive_simulated_annealing(f, x, v, t, ε; ns=20, nε=4, nt=max(100,5length(x)), γ=0.85, c=fill(2,length(x)) )
    y = f(x)
    x_best, y_best = x, y
    y_arr, n, U = [], length(x), Uniform(-1.0,1.0)
    a, counts_cycles, counts_resets = zeros(n), 0, 0
    while true
        for i in 1:n
            x′ = x + basis(i,n)*rand(U)*v[i]
            y′ = f(x′)
            Δy = y′ - y
            if Δy < 0 || rand() < exp(-Δy/t)
                x, y = x′, y′
                a[i] += 1
                if y′ < y_best
                    x_best, y_best = x′, y′
                end
            end
        end
        counts_cycles += 1
        counts_cycles ≥ ns || continue

        counts_cycles = 0
        corana_update!(v, a, c, ns)
        fill!(a, 0)
        counts_resets += 1
        counts_resets ≥ nt || continue

        t *= γ counts_resets = 0
        push!(y_arr, y)

        if !(length(y_arr) > nε && y_arr[end] - y_best ≤ ε && all(abs(y_arr[end]-y_arr[end-u]) ≤ ε for u in 1:nε))
            x, y = x_best, y_best
        else
            break
        end
    end
    return x_best
end
