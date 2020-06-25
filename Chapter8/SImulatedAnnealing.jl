function simulated_annealing(f, x, T, t, k_max)
    y = f(x)
    x_best, y_best = x, y
    res = zeros(k_max, length(x))
    for k in 1 : k_max
        x′ = x + rand(T)
        y′ = f(x′)
        Δy = y′ - y
        if Δy ≤ 0 || rand() < exp(-Δy/t(k))
            x, y = x′, y′
        end
        if y′ < y_best
            x_best, y_best = x′, y′
        end
        length(x) != 1 ? res[k, :] = x_best[:] : res[k, 1] = x_best[1]
    end
    return x_best, res
end

log_schedule(k) = k == 1 ? 10.0 : log_schedule(1) * log(2)/log(k + 1)
exp_schedule(k, γ = 0.5) = k == 1 ? 10.0 : exp_schedule(k-1, γ) * γ
fast_schedule(k) = k == 1 ? 10.0 : fast_schedule(1) / k
