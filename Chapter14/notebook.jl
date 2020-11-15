### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 2fc78330-1c47-11eb-070e-c1c2af8518d8
using LinearAlgebra, Plots, Distributions, StatsFuns, LaTeXStrings

# ╔═╡ 670e146a-265d-11eb-1df3-a9d2a75aa52e
using PlutoUI

# ╔═╡ 1dfcac48-275a-11eb-0070-01dcd3166db4
using ImageContainers

# ╔═╡ 38fe2652-1c47-11eb-3d46-55f50c523dbf
md"
# Chapter 14. Surrogate Models

- 目的関数(Objective function)を置き換え $\rightarrow$ 近似できる関数 = **Surrogate model**

    - なめらかで評価コストが低い関数
    - 目的関数は評価コストが高いか，評価が難しい関数
    - 13章のサンプリング法によって，いい感じのサンプル点で評価


- 多くの代理モデルは基底関数の線形結合で表現できる。
    - 非線形な関数でもOK


- モデル選択
    - A bias-variance tradeoff


- 汎化誤差
    - 代理モデルと観測したデータとのズレや乖離度合い
    - もっとも小さくなるようなモデルを選択する
"

# ╔═╡ ccdaadca-2115-11eb-05fe-e9ab0a063828
md"
## 代理モデル(Surrogate model)をフィッティングする

- 目的関数は直接評価が難しい関数（）
- 代理モデル$\hat{f}$はパラメタ$\theta$で評価したい関数を近似$f$する。
- 未知の確率法則に従うデータを手元の関数で近似する**$\rightarrow$回帰（regression）**と同じ。


m個の評価点，

$X=\left\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(m)}\right\}$

で代理モデルを評価した値，

$\hat{\mathbf{y}}=\left\{\hat{f}_{\boldsymbol{\theta}}\left(\mathbf{x}^{(1)}\right), \hat{f}_{\boldsymbol{\theta}}\left(\mathbf{x}^{(2)}\right), \ldots, \hat{f}_{\boldsymbol{\theta}}\left(\mathbf{x}^{(m)}\right)\right\}$

と目的関数の評価点における値,

$\mathbf{y}=\left\{y^{(1)}, y^{(2)}, \ldots, y^{(m)}\right\}$

の誤差関数を最小化する（任意の$L_p$ノルムで）:

$\begin{array}{ll}
\operatorname{minimize}_{\boldsymbol{\theta}} & \|\mathbf{y}-\hat{\mathbf{y}}\|_{p}
\end{array}.$


"

# ╔═╡ 3c9f0f24-1c49-11eb-1f52-f7773a4af43c
md"
## 線形回帰
"

# ╔═╡ 447d4c88-1c47-11eb-044a-d9248e608219
function design_matrix(X)
	n, m = length(X[1]), length(X)
	return [j==0 ? 1.0 : X[i][j] for i in 1:m, j in 0:n]
end

# ╔═╡ 26a41d8a-1c48-11eb-098a-91029e77c90e
md"
$\hat{f}=\boldsymbol{\theta}^{\top} \mathbf{x}$
$\boldsymbol{\theta}=\mathbf{X}^{+} \mathbf{y}$

ただし，$\mathbf{X}^{+}$は一般化逆行列。

- 一般化逆行列$\rightarrow$正則でない行列に対しても逆行列のような性質を持つ行列を定義できる。
- サンプリングしてきた点が線形従属であってもとりあえず解を，求めることができる
"

# ╔═╡ 8c1629ea-1c47-11eb-2e49-fd54779e9157
function linear_regression(X, y)
    θ = pinv(design_matrix(X))*y
    return x -> θ⋅[1; x]
end

# ╔═╡ 9242c56a-1c47-11eb-1b78-735e356f529a
begin
	X′= collect(-2:1:2) # m > n+1
	X = [-2; -2] # nonindependent points
end

# ╔═╡ ab9eb25e-1c48-11eb-307b-c1decac9c507
begin
	y′ = rand(5) # m > n+1
	y = [-2; 2] # nonindependent points
end

# ╔═╡ b5a2d03a-1c48-11eb-1eb8-65b883dbec6f
fun1 = linear_regression(X, y)

# ╔═╡ da730c04-1c48-11eb-025b-c170b0967328
begin
	p1 = plot(-3:0.1:3, fun1.(-3:0.1:3), label = "nonindependent points")
	scatter!(X, y; label = "train point(s)")
	ylims!(-3, 3)
	xlims!(-3, 3)
	fun′ = linear_regression(X′, y′)
	p2 = plot(-3:0.1:3, fun′.(-3:0.1:3), label = "m < n+1")
	scatter!(X′, y′; label = "train point(s)")
	ylims!(-3, 3)
	xlims!(-3, 3)
	plot(p1, p2)
end

# ╔═╡ 7d7ff996-1c49-11eb-0fe7-f954f850c43d
md"
## 基底関数

- 直線関係以外の，基底関数(basis function)を，全体に適用する
- 基底関数: 多項，正弦波，動径
"

# ╔═╡ d1f8a354-1c48-11eb-26ed-2f4a49d0af71
function regression(X, y, bases)
	B = [b(x) for x in X, b in bases]
	θ = pinv(B)*y
	return x -> sum(θ[i] * bases[i](x) for i in 1 : length(θ))
end

# ╔═╡ fe3a2cce-21b8-11eb-35ec-276326891263
md"
**regression**関数について

- 評価点を入力として基底関数を評価
- 重み付けのための重みを推定する

"

# ╔═╡ ad27a028-1c4a-11eb-3b6f-215ca56c9b13
X2 = collect(-2:1:2)

# ╔═╡ f5de951a-1c4a-11eb-3680-efb59a0a78e4
y2 = exp.(X2) + rand(Uniform(-1, 1), 5)

# ╔═╡ cefa80ae-1c4b-11eb-2d1a-352cd7557b6c
md"
## 多項基底関数

- 入力値を累乗した要素を組み合わせることで，より柔軟な関数表現を可能にする。
- 微分可能な関数であれば，十分な次数の多項式を用いて，精度良く近似できる（テイラー展開）。
"

# ╔═╡ 649ffa28-1c4b-11eb-30e5-17cd968a00d7
polynomial_bases_1d(i, k) = [x->x[i]^p for p in 0:k]

# ╔═╡ bf8d7910-1c48-11eb-2a0c-dfc43b63322d
function polynomial_bases(n, k)
	bases = [polynomial_bases_1d(i, k) for i in 1 : n]
	terms = Function[]
	for ks in Iterators.product([0:k for i in 1:n]...)
		if sum(ks) ≤ k 
			push!(terms,　x->prod(b[j+1](x) for (j,b) in zip(ks,bases)))
		end
	end
	return terms
end

# ╔═╡ fd476fae-1c4d-11eb-239e-c98cdd083eef
X3 = rand(Uniform(-2, 2), 10)

# ╔═╡ 8f8845c0-1c51-11eb-0e5c-e73fc4c96dd3
true_function(x) = 2sin(x) + cos(2x)

# ╔═╡ 17c7f506-1c4e-11eb-27cb-9335ac5e28aa
y3 = @. true_function(X3)

# ╔═╡ a3ebea9a-1c50-11eb-24e8-6b0b173bb53b
y3

# ╔═╡ 6c82b2a8-2760-11eb-3b83-791e7fb54e4b
@bind degree_poly Slider(1:1:20)  

# ╔═╡ 21203976-1c51-11eb-2fd4-4d6a8e723279
md"
## 正弦波基底関数(Sinusoidal basis function)

- フーリエ解析がベースのアイディア
- 区間が決まっている連続関数であれば，無限の正弦波基底の組み合わせで表現できる。
- 不連続な関数であっても，連続点においては収束する。

## フーリエ解析

$f(x)=\frac{\theta_{0}}{2}+\sum_{i=1}^{\infty} \theta_{i}^{(\sin )} \sin \left(\frac{2 \pi i x}{b-a}\right)+\theta_{i}^{(\cos )} \cos \left(\frac{2 \pi i x}{b-a}\right)$

ただし，

$\begin{aligned}
\theta_{i}^{(\sin )} &=\frac{2}{b-a} \int_{a}^{b} f(x) \sin \left(\frac{2 \pi i x}{b-a}\right) d x \\
\theta_{i}^{(\mathrm{cos})} &=\frac{2}{b-a} \int_{a}^{b} f(x) \cos \left(\frac{2 \pi i x}{b-a}\right) d x
\end{aligned}$

- 三角関数を関数系とする級数展開で関数の挙動を確かめる。
- 三角関数の直行性（周波数の異なる関数同士では内積が$0$）を利用して係数が決定される。
- なお，$\theta^{sin(0)}_0 = \frac{\theta_{0}}{2}$

"

# ╔═╡ fce5f744-1c50-11eb-00da-d145006971dd
function sinusoidal_bases_1d(j, k, a, b) 
	T = b[j] - a[j]
	bases = Function[x->1/2]
	for i in 1 : k
		push!(bases, x->sin(2π*i*x[j]/T))
		push!(bases, x->cos(2π*i*x[j]/T))
	end
	return bases
end

# ╔═╡ f5a51dde-1c50-11eb-10bf-cd75698bf29a
function sinusoidal_bases(k, a, b)
	n = length(a)
	bases = [sinusoidal_bases_1d(i, k, a, b) for i in 1 : n]
	terms = Function[]
	for ks in Iterators.product([0:2k for i in 1:n]...)
		powers = [div(k+1,2) for k in ks]
		if sum(powers) ≤ k
			push!(terms,　x->prod(b[j+1](x) for (j,b) in zip(ks,bases)))
		end
	end
	return terms
end

# ╔═╡ 0f491b92-265a-11eb-100d-53b340da82f8
function f4(x)
	if x < 0
		return -1
	else
		return 1
	end
end

# ╔═╡ 76007f5e-265a-11eb-33b9-83b37e84d71b
X4 = collect(-2:0.01:2)

# ╔═╡ 422444b4-265b-11eb-25b4-8f7507d6cccc
y4 = @. f4(X4)

# ╔═╡ f8ef561a-265c-11eb-1de7-9b2de47596fa
@bind degree_fourier Slider(1:1:200)

# ╔═╡ db650888-265c-11eb-30cc-a1726f5cb4fc
md"
# 動径基底関数(radial basis function)

- ある中心点$c$からの距離に依存する関数$\phi$の組み合わせで関数を表現する。
- 中心点をどこに取るか，いくつ取るかは事前に決めておく
    - とりあえずデータ点を中心としてとっておく
" 

# ╔═╡ 04a149c2-2762-11eb-17bc-9324f4ca97a8
storeimage("figure/pic0.png")

# ╔═╡ 5918e0c0-2673-11eb-05f8-c585faf72881
radial_bases(Ψ, C, p = 2) = [x -> Ψ(norm(x - c, p)) for c in C]

# ╔═╡ 4a0e7ae8-2677-11eb-3bb4-0b18cf522caf
md"
# Fitting Noisy Objective Function

- 複雑な関数$\rightarrow$オーバーフィットの問題
- 予測するという観点からは，滑らかな関数のほうが良いケース
- **正則化**手法の利用によって解決

$\underset{\theta}{\operatorname{minimize}}\|\mathbf{y}-\mathbf{B} \theta\|_{2}^{2}+\lambda\|\theta\|_{2}^{2}$

$\boldsymbol{\theta}=\left(\mathbf{B}^{\top} \mathbf{B}+\lambda \mathbf{I}\right)^{-1} \mathbf{B}^{\top} \mathbf{y}$

"

# ╔═╡ d2dc20cc-2678-11eb-18b0-e51fc51274da
function regression(X, y, bases, λ)
	B = [b(x) for x in X, b in bases]
	θ = (B'B + λ*I)\B'y
	return x -> sum(θ[i] * bases[i](x) for i in 1:length(θ))
end

# ╔═╡ f700b248-1c4a-11eb-0162-cbf016bc842f
fun_exp = regression(X2, y2, fill(exp, length(y)))

# ╔═╡ d4ec5f76-1c4b-11eb-3fe5-2bc430381ba6
begin
	scatter(X2, y2; label = "train point(s)", shape = :X, color = :black)
	plot!(-3:0.1:3, fun_exp.(-3:0.1:3), label = "exp basis")
	plot!(-3:0.1:3, linear_regression(X2, y2).(-3:0.1:3), label = "linear basis")
	ylims!(-2, 10)
	xlims!(-3, 3)
end

# ╔═╡ ec541108-1c4e-11eb-2ad8-33f7cac210d5
begin
	scatter(X3, y3; label = "train point(s)", shape = :X, color = :black, legend = :bottomright)
	plot!(-2:0.1:2, true_function.(-2:0.1:2); label = "true", lc = :red)
	plot!(-2:0.1:2, regression(X3, y3, polynomial_bases(1, degree_poly)).(-2:0.1:2); label = "degree = $degree_poly", linestyle = :dash)
end

# ╔═╡ 0ecd6aba-265a-11eb-0d12-7764ecd93cc1
begin
	plot(X4, y4; c = :black, label = "original", legend = :bottomright)
	plot!(-2:0.01:2, regression(X4, y4, sinusoidal_bases(degree_fourier, -2, 2)).(-2:0.01:2); label = "max degree = $degree_fourier", linestyle = :dash)
	ylims!(-1.2, 1.2)
end

# ╔═╡ b0484392-2673-11eb-1185-69f0222cbf70
begin
	scatter(X3, y3; label = "train point(s)", shape = :X, color = :black)
	plot!(-2:0.1:2, true_function.(-2:0.1:2); label = "true", lc = :red)
	plot!(-2:0.1:2, regression(X3, y3, radial_bases(x -> exp(-2x), -2:0.1:2)).(-2:0.1:2); label = L"\exp(-2r^2)", linestyle = :dash)
	plot!(-2:0.1:2, regression(X3, y3, radial_bases(x -> exp(-5x), -2:0.1:2)).(-2:0.1:2); label = L"\exp(-5r^2)", linestyle = :dash)
	plot!(-2:0.1:2, regression(X3, y3, radial_bases(x -> pdf(Normal(), x), -2:0.1:2)).(-2:0.1:2); label = L"Normal(0, 1)", linestyle = :dash)
	plot!(-2:0.1:2, regression(X3, y3, radial_bases(x -> pdf(Normal(0, 0.1), x), -2:0.1:2)).(-2:0.1:2); label = L"Normal(0, 0.1)", linestyle = :dash)
end

# ╔═╡ 64868eae-2679-11eb-14e1-c3d3d68f0b7a
X5 = rand(10)

# ╔═╡ 85f1e94e-2679-11eb-07ca-17e66ca002e3
y5 = X5 .* sin.(5X5) .+ rand(Uniform(-0.1, 0.1),10)

# ╔═╡ a4095066-2679-11eb-169a-c592386cbd16
@bind λvalue Slider(0:0.01:1)

# ╔═╡ 5bdee922-2679-11eb-0ad2-95d75c728c55
begin
	scatter(X5, y5; shape = :X, color = :black, label = "train points")
	plot!(x -> x*sin(5x); label = L"x\sin(x)")
	plot!(0:0.01:1, regression(X5, y5, radial_bases(x -> pdf(Normal(0, 0.1), x), 0:0.01:1), λvalue).(0:0.01:1); label = "Normal", linestyle = :dash)
	xlims!(0, 1)
	ylims!(-2, 2)
end

# ╔═╡ 4ef01b96-267e-11eb-068c-1f41a35836ed
md"
# モデル選択

## 汎化誤差
- 汎化誤差(generalization error)が小さくなるようにモデルを選択する。

$\epsilon_{\mathrm{gen}}=\mathbb{E}_{\mathbf{x} \sim \mathcal{X}}\left[(f(\mathbf{x})-\hat{f}(\mathbf{x}))^{2}\right]$

- 汎化誤差そのものは厳密に計算することができない$\rightarrow$有限個の評価点における平均2乗誤差で近似する。

$\epsilon_{\text {train }}=\frac{1}{m} \sum_{i=1}^{m}\left(f\left(\mathbf{x}^{(i)}\right)-\hat{f}\left(\mathbf{x}^{(i)}\right)\right)^{2}$

## 平均2乗誤差の問題点
- 複雑な関数では，必ずしも汎化誤差を最小化するとは限らない。

"

# ╔═╡ d8fc4552-2689-11eb-0ebc-09af91a266e9
storeimage("figure/pic1.png")

# ╔═╡ ebdf24d4-267f-11eb-2ccb-c50f256c017f
md"
## Train and Validate

訓練データと評価用データを分割して，モデルの妥当性を検証する。

- Holdout
- Cross validation
- Bootstrap

### Holdout

データを適当な比率で二等分$\rightarrow$訓練用データでモデルパラメタを推定し，テスト用データで汎化誤差を推定する。

- モデルの推定時に使用されなかった点において，モデルを評価できる可能性
- 何回かランダムに分割＆評価を繰り返して，その平均を取る

$\epsilon_{\text {holdout }}=\frac{1}{h} \sum_{(\mathbf{x}, y) \in \mathcal{D}_{h}}(y-\hat{f}(\mathbf{x}))^{2}$
"

# ╔═╡ f083ce52-275b-11eb-2108-1b1ba71d9595
storeimage("figure/pic2.png")

# ╔═╡ f3348e0c-275b-11eb-2617-a55e85dad110
md"
### Cross Validation

Holdoutではすべてのデータ点における誤差の最小化を保証しきれない。

$\begin{array}{l}
\epsilon_{\text {cross-validation }}=\frac{1}{k} \sum_{i=1}^{k} \epsilon_{\text {cross-validation }}^{(i)} \\
\epsilon_{\text {cross-validation }}^{(i)}=\frac{1}{\left|\mathcal{D}_{\text {test }}^{(i)}\right|} \sum_{(\mathbf{x}, y) \in \mathcal{D}_{\text {test }}^{(i)}}\left(y-\hat{f}^{(i)}(\mathbf{x})\right)^{2}
\end{array}$

**k-fold cross validation**

- データをk分割して，訓練データとテストデータの組み合わせをまんべんなく試す。
- k個の分割に対して，k-1個で訓練し，残りの一個をテストとして，汎化誤差の推定値を得る。
- 全推定値の平均と分散を計算
"

# ╔═╡ 106b3430-275c-11eb-185d-bf3f9c6136a9
storeimage("figure/pic3.png")

# ╔═╡ cad24368-275c-11eb-3f32-61ac35d70830
md"
**leava-one-out(LOO) cross-validation**

- 可能なすべてのパタンでk-fold cross validationを実行
- 計算コストが増える

※複数のcross-validationの平均を取る以外に，単一のデータセットで複数のモデルを平均化(Averaging)することもある。

### The Bootstrap

- bootstrap sample $\rightarrow$ 重複を許して，元のデータと同じサイズだけサンプリング
- 複数のbootstrap sampleでモデルパラメタを推定し，元のデータをテストデータとして利用

$\begin{aligned}
\epsilon_{\text {boot }} &=\frac{1}{b} \sum_{i=1}^{b} \epsilon_{\text {test }}^{(i)} \\
&=\frac{1}{m} \sum_{j=1}^{m} \frac{1}{b} \sum_{i=1}^{b}\left(y^{(j)}-\hat{f}^{(i)}\left(\mathbf{x}^{(j)}\right)\right)^{2}
\end{aligned}$

**Leave-one-out bootstrap estimate**

- Bootstrap法では，テストデータに訓練データ中のindexが含まれていた
- 元データ中のindex$j$ による汎化誤差の計算は，index $j$ を含まないモデルのみで

$\epsilon_{\text {leave-one-out-boot }}=\frac{1}{m} \sum_{j=1}^{m} \frac{1}{c_{-j}} \sum_{i=1}^{b}\left\{\begin{array}{ll}
\left(y^{(j)}-\hat{f}^{(i)}\left(\mathbf{x}^{(j)}\right)\right)^{2} & \text { if } j \text { th index was not in the } i \text { th bootstrap sample } \\
0 & \text { otherwise }
\end{array}\right.$

**0.632 bootstrap estimator**

- LOO bootstrapでは，テストサイズに依存するバイアスが含まれる。
- データ中にnon-distinctなサンプルが$m\rightarrow \infty$の場合，平均で1/3は存在
- distinct indicesは2/3

$\epsilon_{0.632-\mathrm{boot}}=0.632 \epsilon_{\text {leave-one-out-boot }}+0.368 \epsilon_{\mathrm{boot}}$


"

# ╔═╡ cb265048-275c-11eb-0b05-135effb519d4


# ╔═╡ Cell order:
# ╠═38fe2652-1c47-11eb-3d46-55f50c523dbf
# ╠═ccdaadca-2115-11eb-05fe-e9ab0a063828
# ╠═3c9f0f24-1c49-11eb-1f52-f7773a4af43c
# ╠═2fc78330-1c47-11eb-070e-c1c2af8518d8
# ╠═447d4c88-1c47-11eb-044a-d9248e608219
# ╟─26a41d8a-1c48-11eb-098a-91029e77c90e
# ╠═8c1629ea-1c47-11eb-2e49-fd54779e9157
# ╠═9242c56a-1c47-11eb-1b78-735e356f529a
# ╠═ab9eb25e-1c48-11eb-307b-c1decac9c507
# ╠═b5a2d03a-1c48-11eb-1eb8-65b883dbec6f
# ╠═da730c04-1c48-11eb-025b-c170b0967328
# ╠═7d7ff996-1c49-11eb-0fe7-f954f850c43d
# ╠═d1f8a354-1c48-11eb-26ed-2f4a49d0af71
# ╠═fe3a2cce-21b8-11eb-35ec-276326891263
# ╠═ad27a028-1c4a-11eb-3b6f-215ca56c9b13
# ╠═f5de951a-1c4a-11eb-3680-efb59a0a78e4
# ╠═f700b248-1c4a-11eb-0162-cbf016bc842f
# ╠═d4ec5f76-1c4b-11eb-3fe5-2bc430381ba6
# ╠═cefa80ae-1c4b-11eb-2d1a-352cd7557b6c
# ╠═649ffa28-1c4b-11eb-30e5-17cd968a00d7
# ╠═bf8d7910-1c48-11eb-2a0c-dfc43b63322d
# ╠═fd476fae-1c4d-11eb-239e-c98cdd083eef
# ╠═8f8845c0-1c51-11eb-0e5c-e73fc4c96dd3
# ╠═17c7f506-1c4e-11eb-27cb-9335ac5e28aa
# ╠═a3ebea9a-1c50-11eb-24e8-6b0b173bb53b
# ╠═6c82b2a8-2760-11eb-3b83-791e7fb54e4b
# ╠═ec541108-1c4e-11eb-2ad8-33f7cac210d5
# ╟─21203976-1c51-11eb-2fd4-4d6a8e723279
# ╠═fce5f744-1c50-11eb-00da-d145006971dd
# ╠═f5a51dde-1c50-11eb-10bf-cd75698bf29a
# ╠═0f491b92-265a-11eb-100d-53b340da82f8
# ╠═76007f5e-265a-11eb-33b9-83b37e84d71b
# ╠═422444b4-265b-11eb-25b4-8f7507d6cccc
# ╠═670e146a-265d-11eb-1df3-a9d2a75aa52e
# ╠═f8ef561a-265c-11eb-1de7-9b2de47596fa
# ╠═0ecd6aba-265a-11eb-0d12-7764ecd93cc1
# ╠═db650888-265c-11eb-30cc-a1726f5cb4fc
# ╠═04a149c2-2762-11eb-17bc-9324f4ca97a8
# ╠═5918e0c0-2673-11eb-05f8-c585faf72881
# ╠═b0484392-2673-11eb-1185-69f0222cbf70
# ╠═4a0e7ae8-2677-11eb-3bb4-0b18cf522caf
# ╠═d2dc20cc-2678-11eb-18b0-e51fc51274da
# ╠═64868eae-2679-11eb-14e1-c3d3d68f0b7a
# ╠═85f1e94e-2679-11eb-07ca-17e66ca002e3
# ╠═a4095066-2679-11eb-169a-c592386cbd16
# ╠═5bdee922-2679-11eb-0ad2-95d75c728c55
# ╠═1dfcac48-275a-11eb-0070-01dcd3166db4
# ╠═4ef01b96-267e-11eb-068c-1f41a35836ed
# ╠═d8fc4552-2689-11eb-0ebc-09af91a266e9
# ╠═ebdf24d4-267f-11eb-2ccb-c50f256c017f
# ╠═f083ce52-275b-11eb-2108-1b1ba71d9595
# ╠═f3348e0c-275b-11eb-2617-a55e85dad110
# ╠═106b3430-275c-11eb-185d-bf3f9c6136a9
# ╠═cad24368-275c-11eb-3f32-61ac35d70830
# ╠═cb265048-275c-11eb-0b05-135effb519d4
