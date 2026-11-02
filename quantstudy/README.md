# cqfm1 —— CQF Module 1 自建量化工具箱

把原课程里依赖 `quantmod` / `yfinance` / `Supabase` 的接口，换成
**tushare（A 股数据）+ SQLite（本地存储）+ plotly（交互图表）**，
并且把每个"黑箱函数"重新实现了一遍。

配套 notebook：`../totalm1.ipynb`

---

## 快速开始

```python
import sys; sys.path.insert(0, ".")
import cqfm1
from cqfm1 import config

config.describe()                       # 看 token / 缓存路径 / 限速
df = cqfm1.getData("600519.SH", period="2y", adjust="qfq")
df["Close"].iplot(kind="line")          # 一行出图
```

不需要安装：把 `cqfm1/` 放在 notebook 同级目录即可。想安装成包：

```bash
pip install -e .            # 开发模式，改代码立即生效
pip install -e ".[fast]"    # 顺带装 numba
```

---

## 模块地图

| 模块 | 职责 | 关键 API |
|---|---|---|
| `config.py` | token / 路径 / 限速 / 日志 | `settings`, `set_token()`, `describe()` |
| `_cache.py` | SQLite 连接、接口缓存、物化序列、限流状态 | `connect()`, `cache_get/put`, `series_get/put` |
| `datasource.py` | tushare 适配 + akshare 兜底 + 错误分类 | `query()`, `ak_daily`, `ak_bond_rate`, `ak_names` |
| `marketdata.py` | 行情 / 公司快照 / 期权链 / 收益率曲线 / 新闻 | `getData`, `getTicker`, `get_option_chain`, `get_yield_curve` |
| `timeseries.py` | 收益率、波动率、贝塔、回撤、夏普 | `periodReturn`, `volatility`, `beta`, `maxDrawdown` |
| `db.py` | 本地版 QuantmodDB（表设计对齐原课程） | `QuantmodDB().register/load_history/get_asset_prices` |
| `charts.py` | 给 pandas 注册 `.iplot()` | `df.iplot(kind="candlestick")` 等 14 种 |
| `options.py` | 二叉树 / Black-Scholes / 隐含波动率 / Greeks / MC | `binomial_option`, `implied_vol`, `BinomialOptionPricing` |
| `fast.py` | 同一算法的四种实现 + 基准测试 | `binomial_py/numpy/numba/european_cf`, `benchmark()` |

---

## 数据源现实（重要）

本机 `~/tk.csv` 里的 token 是**免费档**，实测额度：

| 接口 | 额度 | 影响 |
|---|---|---|
| `daily`, `adj_factor` | 1 次 / **分钟** | 行情够用 |
| `stock_basic`, `dividend`, `opt_basic`, `opt_daily`, `us_tycr`, `cn_gdp`, `cn_cpi`, `trade_cal` | 1 次 / **小时** | 公司概况、期权链、宏观要排队 |
| `income`, `balancesheet`, `cashflow`, `news`, `fund_daily`, `yc_cb` | **无权限** | 需 2000 积分 |

所以库内置了**三级降级链**：

```
tushare（主源）
  └─ akshare 东财（次选）
       └─ akshare 新浪（最稳，实测可用）
            └─ 本地 SQLite 缓存（彻底离线）
                 └─ 明确抛出"不可用"，由上层决定怎么展示
```

限流状态带 TTL 记录在 `api_state` 表里，命中限流的接口会被**直接短路**，
不会再浪费 60 秒去重试。

---

## 性能：同一算法的四档实现

```python
from cqfm1 import fast
fast.benchmark(sizes=(100, 500, 1000, 3000))
```

| 写法 | 原理 | n=1000 耗时 |
|---|---|---|
| `binomial_py` | Python 双循环 + 列表 | ~29 ms |
| `binomial_numpy` | 节点维向量化，O(n) 滚动缓冲 | ~2.2 ms（13×） |
| `binomial_european_cf` | 二项分布闭式解，一次 `np.dot` | ~0.17 ms（170×） |
| `binomial_numba` | `@njit` 编译成机器码 | 需 `pip install numba` |

**结论：先向量化，再 JIT，最后才考虑 C 扩展。**

---

## 已知边界

- 期权链依赖 `opt_basic`，被限流时 notebook 改用构造的期权链演示 IV 反解
- `Ticker.info` 里 PE / PB / 市值 / 行业 需要 2000 积分，无权限时返回 `None`
- 分钟线走 akshare 东财接口，该域名在某些网络下会 `RemoteDisconnected`
- 复权优先用 `adj_factor`；拿不到时改用新浪的前复权价（`stock_zh_a_daily`）

---

## 自测

```bash
pytest tests/ -v
```
