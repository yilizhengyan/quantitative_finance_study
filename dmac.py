import numpy as np
import pandas as pd
import tushare as ts

TS_TOKEN = "YOURTOKEN"
COMMISSION = 0.0003     #佣金万三，双边
STAMP_TAX  = 0.001      #印花税千一，仅卖出
SLIPPAGE   = 0.001      #滑点千一

pro = ts.pro_api(TS_TOKEN) if False else None

#数据
def load_daily(code, start="20140101", end="20241231", token=None):
    """
    取日线并按 pre_close 反推后复权因子。
    tushare的pro.daily()没有adj参数，返回的是不复权价；
    前复权要用pro_bar(adj='qfq')，但它依赖adj_factor接口（免费版额度低），
    且新版本tushare与pandas 3.x存在fillna(method=)兼容问题。
    这里用pre_close反推：除权日pre_close != 上一日 close，比值就是当天的复权因子。
    """
    if token:
        ts.set_token(token)
    pro = ts.pro_api()
    df = pro.daily(ts_code=code, start_date=start, end_date=end)
    if df is None or df.empty:
        raise ValueError(f"{code} 无数据")
    df = df.sort_values("trade_date").reset_index(drop=True)

    prev_close = df["close"].shift(1)
    ratio = (prev_close / df["pre_close"]).fillna(1.0)   #除权日<1，倒数>1
    df["adj_factor"] = ratio.cumprod()
    for col in ["open", "high", "low", "close"]:
        df["adj_" + col] = df[col] * df["adj_factor"]
    return df


#回测
def backtest(df, fast=5, slow=10, comm=COMMISSION, tax=STAMP_TAX,
             slip=SLIPPAGE, cost_on=True):
    """
    信号在t日收盘产生，t+1日开盘成交。
    P&L拆成两段：
        隔夜段prev_pos * (open_t / close_{t-1} - 1)
        日内段new_pos  * (close_t / open_t - 1)
    这样成交时点和收益计算才是一致的。
    """
    x = df.copy().reset_index(drop=True)
    x["ma_f"] = x["adj_close"].rolling(fast).mean()
    x["ma_s"] = x["adj_close"].rolling(slow).mean()

    #信号：1=持有，0=空仓。A股个股不能做空，所以没有-1。
    sig = (x["ma_f"] > x["ma_s"]).astype(float)
    x["pos"] = sig.shift(1).fillna(0.0)          # t日的仓位由t-1日收盘决定
    x["prev_pos"] = x["pos"].shift(1).fillna(0.0)

    overnight = x["adj_open"] / x["adj_close"].shift(1) - 1
    intraday = x["adj_close"] / x["adj_open"] - 1
    gross = x["prev_pos"] * overnight + x["pos"] * intraday

    turn = (x["pos"] - x["prev_pos"]).abs()
    rate = np.where(x["pos"] > x["prev_pos"], comm + slip, comm + tax + slip)
    cost = turn * rate if cost_on else 0.0

    x["ret_gross"] = gross
    x["ret_net"] = gross - cost
    x["nav"] = (1 + x["ret_net"]).cumprod()
    x["bh"] = (1 + x["adj_close"].pct_change().fillna(0)).cumprod()
    x = x.iloc[slow:].reset_index(drop=True)
    return x


def stats(x, name=""):
    n = len(x)
    yrs = n / 242
    nav = x["nav"].iloc[-1]
    r = x["ret_net"]
    return {
        "标的": name,
        "策略累计": nav - 1,
        "基准累计": x["bh"].iloc[-1] - 1,
        "策略CAGR": nav ** (1 / yrs) - 1,
        "基准CAGR": x["bh"].iloc[-1] ** (1 / yrs) - 1,
        "年化波动": r.std() * np.sqrt(242),
        "Sharpe": r.mean() / r.std() * np.sqrt(242) if r.std() > 0 else np.nan,
        "最大回撤": (x["nav"] / x["nav"].cummax() - 1).min(),
        "交易次数": int((x["pos"].diff().abs() > 1e-12).sum()),
        "年换手": x["pos"].diff().abs().sum() / yrs,
        "持仓占比": x["pos"].mean(),
    }


def vol_quintile(x, col="vol20"):
    """
    按 20 日已实现波动率分组，看低波动是否失效
    """
    y = x.copy()
    y["vol20"] = y["adj_close"].pct_change().rolling(20).std()
    y = y.dropna(subset=["vol20"]).copy()
    y["组"] = pd.qcut(y["vol20"], 5, labels=["Q1最低", "Q2", "Q3", "Q4", "Q5最高"])
    g = y.groupby("组", observed=True)["ret_net"].agg(["mean", "std", "count"])
    g["年化收益"] = g["mean"] * 242
    g["Sharpe"] = g["mean"] / g["std"] * np.sqrt(242)
    return g


if __name__ == "__main__":
    TOKEN = "YOURTOKEN"
    rows = []
    for code in ["601398.SH", "600519.SH", "000001.SZ", "601318.SH"]:
        try:
            df = load_daily(code, token=TOKEN)
            x = backtest(df, 5, 10)
            rows.append(stats(x, code))
            if code == "601398.SH":
                print("\n=== 601398 波动率分组 ===")
                print(vol_quintile(x).round(4).to_string())
        except Exception as e:
            print(code, "失败:", e)

    out = pd.DataFrame(rows).set_index("标的")
    pd.set_option("display.width", 200)
    print("\n=== 主结果 ===")
    print(out.round(4).to_string())