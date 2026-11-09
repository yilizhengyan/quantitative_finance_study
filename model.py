import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tushare as ts
import akshare as ak

def binomial_option(S, K, r, sigma, t, steps, kind='call', american=False, output=0):
    '''
    构建期权二叉树
    
    Params
    ------
    S           int/float   标的现价
    K           int/float   行权价
    r           float       无风险利率
    sigma       float       波动率
    t           int/float   到期时间（年）
    steps       int         步数
    kind        str         'call' or 'put'
    american    bool        True=美式（可以提前行权）, False=欧式
    output      int         0=价格矩阵 1=到期损益 2=期权价值 3=Delta
    ------
    '''
    
    dt = t / steps
    u = np.exp(sigma * np.sqrt(dt))
    v = 1 / u
    p = (np.exp(r * dt) - v) / (u - v)
    df = np.exp(-r * dt)
    
    px = np.zeros((steps + 1, steps + 1))     # 标的价格
    cp = np.zeros((steps + 1, steps + 1))     # 到期内在价值
    V = np.zeros((steps + 1, steps + 1))      # 期权价值
    delta = np.zeros((steps + 1, steps + 1))  # Delta
    
    # 前向方程：逐节点算价格与内在价值
    for j in range(steps + 1):
        for i in range(j + 1):
            px[i, j] = S * np.power(u, j - i) * np.power(v, i)
            if kind == 'call':
                cp[i, j] = np.maximum(px[i, j] - K, 0)
            else:
                cp[i, j] = np.maximum(K - px[i, j], 0)
    
    # 后向方程：从到期日贴现回今天
    for j in range(steps, -1, -1):
        for i in range(j + 1):
            if j == steps:
                V[i, j] = cp[i, j]          # 到期日：期权价值=内在价值

                if kind == 'call':
                    delta[i, j] = 1.0 if px[i, j] >= K else 0.0
                else:
                    delta[i, j] = -1.0 if px[i, j] <= K else 0.0
                
            else:
                continuation_value = df * (p * V[i, j + 1] + (1 - p) * V[i + 1, j + 1])

                if american:
                    V[i, j] = np.maximum(continuation_value, cp[i, j])
                else:
                    V[i, j] = continuation_value

                delta[i, j] = (V[i, j + 1] - V[i + 1, j + 1]) / (px[i, j + 1] - px[i + 1, j + 1])
    
    results = (px, cp, V, delta)
    
    return results[output]


def plot_binomial_tree(S, K, r, sigma, t, steps, kind='call', american=False, output=2, max_steps_plot=5):
    '''
    画出二叉树图（自动调用 binomial_option 获取数据）
    
    Params
    ------
    kind            str   'call' or 'put'
    american        bool  True=美式(可以提前行权), False=欧式
    output          int   0=价格 1=损益 2=期权价值 3=Delta
    max_steps_plot  int   最多画前几步（防止节点太密看不清）
    ------
    '''
    
    px    = binomial_option(S, K, r, sigma, t, steps, kind=kind, american=american, output=0)
    cp    = binomial_option(S, K, r, sigma, t, steps, kind=kind, american=american, output=1)
    V     = binomial_option(S, K, r, sigma, t, steps, kind=kind, american=american, output=2)
    delta = binomial_option(S, K, r, sigma, t, steps, kind=kind, american=american, output=3)
    
    output_matrix = (px, cp, V, delta)[output]
    output_name = ["Price", "Payoff", "Value", "Delta"][output]
    plot_steps = min(steps, max_steps_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    title_style = "American" if american else "European"
    ax.set_title(f"Binomial Tree ({kind.title()} - {title_style}) - {output_name}")
    
    for j in range(plot_steps + 1):
        for i in range(j + 1):
            x = j
            y = (j - 2 * i) / 2  # 神奇的Y轴坐标，让树散开
            node_text = f"S: {px[i, j]:.2f}\n{output_name}: {output_matrix[i, j]:.2f}"
            ax.text(x, y, node_text, ha="center", va="center", 
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
            
            # 画节点之间的连线
            if j < plot_steps:
                ax.plot([x, x + 1], [y, y + 0.5], "k-", lw=1) # 上行线
                ax.plot([x, x + 1], [y, y - 0.5], "k-", lw=1) # 下行线
                
    ax.set_xlim(-0.5, plot_steps + 0.5)
    ax.set_ylim(-plot_steps / 2 - 0.5, plot_steps / 2 + 0.5)
    ax.axis("off")
    plt.show()


def black_scholes_option(S, K, r, sigma, t, kind, output=0):
    '''
    Black-Scholes欧式期权闭式解
    ------
    S       int/float/ndarray   标的现价
    K       int/float/ndarray   行权价
    r       float               无风险利率
    sigma   float/ndarray       波动率
    t       int/float/ndarray   到期时间（年）
    kind    str                 'call'或'put'
    output  int                 0=价格 1=Delta 2=Gamma 3=Vega 4=Theta 5=Rho
    ------ 
    '''
    
    if sigma <= 0:
        raise ValueError("波动率sigma必须大于0")
    
    if t <= 0:
        intrinsic = np.maximum(S - K, 0.0) if kind == 'call' else np.maximum(K - S, 0.0)
        delta_expiry = 1.0 if (kind == 'call' and S >= K) else (-1.0 if(kind == 'put' and S <= K) else 0.0)
        results = (intrinsic, delta_expiry, 0.0, 0.0, 0.0, 0.0)
        return results[output]
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    df = np.exp(-r * t)
    pdf_d1 = norm.pdf(d1)
      
    if kind == 'call':
        price = S * norm.cdf(d1) - K * df * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * t * df * norm.cdf(d2) / 100.0
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(t)) - r * K * df * norm.cdf(d2)) / 365.0
    else:
        price = K * df * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        rho   = -K * t * df * norm.cdf(-d2) / 100.0
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(t)) + r * K * df * norm.cdf(-d2)) / 365.0
    
    gamma = pdf_d1 / (S * sigma * np.sqrt(t))
    vega  = S * pdf_d1 * np.sqrt(t) / 100.0

    results = (price, delta, gamma, vega, theta, rho)
    
    return results[output]


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 标的价格矩阵px，第j列是第j步的所有可能价格，第i行表示跌了i次，右上三角以外都是0
    px = binomial_option(100, 100, 0.05, 0.2, 1, 4, output=0)
    print(px)

    # 到期内在价格矩阵cp = max{S_T - K, 0}
    cp = binomial_option(100, 100, 0.05, 0.2, 1, 4, output=1)
    print(cp)

    # 期权价格矩阵opx，左上角就是当天期权价格
    opx = binomial_option(100, 100, 0.05, 0.2, 1, 4, output=2)
    print(opx)

    # 对冲比率Delta
    dlt = binomial_option(100, 100, 0.05, 0.2, 1, 4, output=3)
    print(dlt)

    S, K, r, sigma, t = 100, 100, 0.05, 0.2, 1
    plot_binomial_tree(S, K, r, sigma, t, steps=5, kind='call', american=False, output=2)
    plot_binomial_tree(S, K, r, sigma, t, steps=5, kind='put', american=True, output=2)

    eu_put = binomial_option(60, 100, 0.05, 0.2, 1, steps=500, kind='put', american=False, output=2)[0, 0]
    am_put = binomial_option(60, 100, 0.05, 0.2, 1, steps=500, kind='put', american=True, output=2)[0, 0]

    print(f"欧式Put价值: {eu_put:.4f}")
    print(f"美式Put价值: {am_put:.4f}")
    print(f"提前行权溢价: {am_put - eu_put:.4f}")
