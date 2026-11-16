from zyquant.model.inputs import OptionInputs, OptionType, ExerciseStyle
from zyquant.model.binomial import (
    binomial_option, 
    binomial_option_closed_form, 
    BinomialModel, 
    plot_binomial_tree
)

inputs = OptionInputs(
    S=100,
    K=100,
    T=1.0,
    r=0.05,
    sigma=0.2,
    q=0.0
)

print("--- 1. 纯函数：binomial_option ---")
value_tree = binomial_option(100, 100, 0.05, 0.2, 1, 10, output="value", method="loop")
print("期权价值矩阵左上角（今天价格）:", value_tree[0, 0])

delta_tree = binomial_option(100, 100, 0.05, 0.2, 1, 100, output="delta", method="vectorized")
print("初始 Delta:", delta_tree[0, 0])

am_put_tree = binomial_option(100, 100, 0.05, 0.2, 1, 100, kind="put", style="american", output="value")
print("美式看跌期权价格:", am_put_tree[0, 0])

print("\n--- 2. 纯函数：binomial_option_closed_form ---")
cf_price = binomial_option_closed_form(100, 100, 0.05, 0.2, 1, 100)
print(f"欧式看涨闭式解价格 (100步): {cf_price:.4f}")


print("\n--- 3. 纯函数：plot_binomial_tree ---")
plot_binomial_tree(
    S=100, K=100, r=0.05, sigma=0.2, T=1, n=4, 
    kind="call", style="european", output="value", max_steps_plot=4)


print("\n--- 4. 面向对象：BinomialModel ---")
model = BinomialModel(inputs=inputs, n=100, option_type=OptionType.CALL)

print(f"现价: {model.price}")
print(f"内在价值: {model.payoff}")
print(f"期权理论价值: {model.value}")
print(f"初始 Delta: {model.delta}")

model.plot(output="delta", max_steps_plot=4)

# 如果你想画所有的指标，循环即可
# for out in ["price", "payoff", "value", "delta"]:
#     model.plot(output=out)
