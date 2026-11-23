# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # 防止画图弹窗卡住终端
import numpy as np

from zyquant.model.inputs import OptionInputs, OptionType, ExerciseStyle
from zyquant.model.binomial import (
    binomial_option, binomial_option_closed_form, BinomialModel, plot_binomial_tree
)
from zyquant.model.blackscholes import (
    black_scholes, black_76, garman_kohlhagen, bs_greeks, implied_vol, BlackScholes
)
from zyquant.model.asian import (
    geometric_asian_option, geometric_asian_forward, AsianModel
)
from zyquant.model.montecarlo import (
    montecarlo_option, MonteCarloModel, plot_montecarlo_paths
)
# AI生成的测试代码
# 基础参数
S, K, r, sigma, T, q = 100.0, 100.0, 0.05, 0.2, 1.0, 0.02

inputs = OptionInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)


# =========================================================
# 模块 1：二叉树 (binomial)
# =========================================================
print("="*50)
print("模块 1：二叉树 (binomial)")
print("="*50)

print("\n--- 1. 纯函数：binomial_option ---")
value_tree = binomial_option(S, K, r, sigma, T, n=10, output="value", method="loop")
print("期权价值矩阵左上角（今天价格）:", value_tree[0, 0])

delta_tree = binomial_option(S, K, r, sigma, T, n=100, output="delta", method="vectorized")
print("初始 Delta:", delta_tree[0, 0])

am_put_tree = binomial_option(S, K, r, sigma, T, n=100, kind="put", style="american", output="value")
print("美式看跌期权价格:", am_put_tree[0, 0])

print("\n--- 2. 纯函数：binomial_option_closed_form ---")
cf_price = binomial_option_closed_form(S, K, r, sigma, T, n=100)
print(f"欧式看涨闭式解价格 (100步): {cf_price:.4f}")

print("\n--- 3. 纯函数：plot_binomial_tree ---")
plot_binomial_tree(
    S=S, K=K, r=r, sigma=sigma, T=T, n=4, 
    kind="call", style="european", output="value", max_steps_plot=4
)

print("\n--- 4. 面向对象：BinomialModel ---")
bt_model = BinomialModel(inputs=inputs, n=100, option_type=OptionType.CALL)
print(f"现价: {bt_model.price:.4f}")
print(f"内在价值: {bt_model.payoff:.4f}")
print(f"期权理论价值: {bt_model.value:.4f}")
print(f"初始 Delta: {bt_model.delta:.4f}")
bt_model.plot(output="delta", max_steps_plot=4)


# =========================================================
# 模块 2：Black-Scholes / Black-76 / GK (blackscholes)
# =========================================================
print("\n" + "="*50)
print("模块 2：Black-Scholes / B76 / GK (blackscholes)")
print("="*50)

print("\n--- 1. 纯函数：三大模型对比 ---")
bs_price = float(black_scholes(S, K, r, sigma, T, q=q))
b76_price = float(black_76(S, K, r, sigma, T))
gk_price = float(garman_kohlhagen(S, K, r_d=r, r_f=0.03, sigma=sigma, T=T))
print(f"Black-Scholes 价格: {bs_price:.4f}")
print(f"Black-76 价格: {b76_price:.4f}")
print(f"Garman-Kohlhagen 价格: {gk_price:.4f}")

print("\n--- 2. 纯函数：Greeks 与隐含波动率 ---")
greeks = bs_greeks(S, K, r, sigma, T, q=q)
print(f"Delta: {float(greeks['delta']):.4f}, Gamma: {float(greeks['gamma']):.4f}, Vega: {float(greeks['vega']):.4f}")

# 用刚才算出的理论价反解波动率
iv_solved = float(implied_vol(bs_price, S, K, r, T, q=q))
print(f"反解隐含波动率: {iv_solved:.6f} (目标: {sigma})")

print("\n--- 3. 数组批量计算 ---")
strikes = np.array([90.0, 100.0, 110.0])
prices_batch = black_scholes(S, strikes, r, sigma, T, q=q)
print(f"三个行权价的 BS 价格: {np.round(prices_batch, 4)}")

print("\n--- 4. 面向对象：BlackScholes ---")
bs_model = BlackScholes(inputs=inputs, option_type=OptionType.CALL)
print(bs_model)


# =========================================================
# 模块 3：几何亚式 (asian)
# =========================================================
print("\n" + "="*50)
print("模块 3：几何亚式 (asian)")
print("="*50)

print("\n--- 1. 纯函数：geometric_asian_option ---")
fwd_price = geometric_asian_forward(S, r, sigma, T, q=q, n_steps=252)
asian_price = geometric_asian_option(S, K, r, sigma, T, q=q, n_steps=252)
print(f"几何平均期望 (Forward): {fwd_price:.4f}")
print(f"几何亚式看涨期权价格: {asian_price:.4f}")

print("\n--- 2. 面向对象：AsianModel ---")
asian_model = AsianModel(inputs=inputs, option_type=OptionType.CALL, n_steps=252)
print(asian_model)


# =========================================================
# 模块 4：蒙特卡洛 (montecarlo)
# =========================================================
print("\n" + "="*50)
print("模块 4：蒙特卡洛 (montecarlo)")
print("="*50)

print("\n--- 1. 纯函数：montecarlo_option ---")
# 欧式（注意：返回的是元组，需要拆包）
eu_price, eu_se, eu_lo, eu_hi = montecarlo_option(S, K, r, sigma, T, q=q, n_paths=50_000)
print(f"欧式看涨价格: {eu_price:.4f} (标准误: {eu_se:.4f}, 95%CI: [{eu_lo:.4f}, {eu_hi:.4f}])")

# 美式（看跌）
am_put_price, _, _, _ = montecarlo_option(S, K, r, sigma, T, q=q, kind="put", style="american", n_paths=50_000)
print(f"美式看跌价格: {am_put_price:.4f}")

# 亚式（算术平均）
as_price, _, _, _ = montecarlo_option(S, K, r, sigma, T, q=q, style="asian", average="arithmetic", n_paths=50_000)
print(f"算术亚式价格: {as_price:.4f}")

# 障碍（向上敲出）
bar_price, _, _, _ = montecarlo_option(S, K, r, sigma, T, q=q, style="barrier", barrier=130, barrier_type="up_and_out", n_paths=50_000)
print(f"向上敲出障碍价格: {bar_price:.4f}")

print("\n--- 2. Sobol 拟随机抽样对比 ---")
sobol_price, sobol_se, _, _ = montecarlo_option(S, K, r, sigma, T, q=q, sampler="sobol", n_paths=8192)
print(f"Sobol 抽样价格 (8192路径): {sobol_price:.4f} (标准误: {sobol_se:.4f})")

print("\n--- 3. 面向对象：MonteCarloModel ---")
mc_model = MonteCarloModel(inputs=inputs, n_paths=50_000, exercise_style=ExerciseStyle.EUROPEAN)
print(mc_model)

print("\n--- 4. 纯函数：plot_montecarlo_paths ---")
plot_montecarlo_paths(
    S=S, K=K, r=r, sigma=sigma, T=T, q=q, style="barrier", barrier=130, barrier_type="up_and_out",
    n_paths=1000, max_paths_plot=40, seed=42
)
print("路径图表已生成完毕。")