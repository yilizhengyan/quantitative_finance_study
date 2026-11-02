# -*- coding: utf-8 -*-
"""cqfm1.config —— 全局配置：Token、缓存目录、接口限速、日志。

设计原则
--------
1. 所有可变状态集中在一个 Settings 对象里，其它模块只读不写（单一事实来源）。
2. 默认值必须能开箱即用：没配 Token 时给出可读提示，而不是抛 ImportError。
3. 路径与环境变量优先：CQFM1_HOME / TUSHARE_TOKEN 可在不改代码的情况下覆盖。

用法
----
>>> from cqfm1 import config
>>> config.set_token("你的 tushare token")      # 传空串则重新自动探测
>>> config.describe()                            # 打印当前生效配置
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Optional

__all__ = ["settings", "Settings", "set_token", "describe", "logger"]


def _project_root() -> Path:
    """库所在目录的上一级，通常就是 notebook 所在文件夹。"""
    return Path(__file__).resolve().parent.parent


def _default_home() -> Path:
    """缓存目录：环境变量 > notebook 同级目录。"""
    env = os.environ.get("CQFM1_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "m1_cache"


def _read_tk_csv(path: Path) -> Optional[str]:
    """解析 tushare 官方写法 ~/tk.csv（一行表头 + 一行取值）。"""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return None
    for row in rows:
        for cell in row:
            cell = cell.strip()
            if len(cell) >= 32 and " " not in cell and cell.lower() != "token":
                return cell
    return None


def _discover_token() -> Optional[str]:
    """优先级：环境变量 -> ~/tk.csv -> ~/.tushare_token -> 缓存目录/token.txt"""
    for env_name in ("TUSHARE_TOKEN", "TS_TOKEN"):
        val = os.environ.get(env_name)
        if val and val.strip():
            return val.strip()
    candidates = [
        Path.home() / "tk.csv",
        Path.home() / ".tushare_token",
        _default_home() / "token.txt",
    ]
    for p in candidates:
        if not p.exists():
            continue
        tok = _read_tk_csv(p) if p.suffix.lower() == ".csv" else (p.read_text(encoding="utf-8-sig").strip() or None)
        if tok:
            return tok
    return None


class Settings:
    """运行期配置：类属性即默认值，实例属性可覆盖。"""

    # ---- 数据源 ----
    token: Optional[str] = None
    source: str = "tushare"          # tushare | akshare | cache
    fallback_akshare: bool = True    # tushare 无权限时是否降级到 akshare

    # ---- 缓存 ----
    home: Path = _default_home()
    db_name: str = "market.db"

    # ---- 网络与限速 ----
    rate_limit: float = 61.0         # 免费 token 每接口 1 次/分钟；0 表示不限速
    timeout: float = 30.0
    max_retry: int = 2
    offline: bool = False            # True: 只用本地缓存，绝不联网
    stale_days: int = 3              # 尾部缺失不超过 N 天时直接用手上的缓存，不联网

    # ---- 展示 ----
    verbose: bool = True
    tz: str = "Asia/Shanghai"

    def __init__(self) -> None:
        self.token = _discover_token()
        self.logger = logging.getLogger("cqfm1")

    @property
    def cache_dir(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        return self.home

    @property
    def db_path(self) -> Path:
        return self.cache_dir / self.db_name

    @property
    def has_token(self) -> bool:
        return bool(self.token)

    def describe(self) -> str:
        """打印当前配置，排查问题时的第一站。"""
        tok = (self.token[:6] + "..." + self.token[-4:]) if self.token else "未设置"
        lines = [
            "cqfm1 配置",
            "  token           : " + tok,
            "  source          : " + self.source,
            "  fallback_akshare: " + str(self.fallback_akshare),
            "  cache_dir       : " + str(self.cache_dir),
            "  db_path         : " + str(self.db_path),
            "  rate_limit      : %.0fs / 接口" % self.rate_limit,
            "  offline         : " + str(self.offline),
        ]
        text = "\n".join(lines)
        if self.verbose:
            print(text)
        return text


settings = Settings()

logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("cqfm1")


def set_token(token: str, persist: bool = False) -> None:
    """设置 tushare token。

    Parameters
    ----------
    token : str
        tushare.pro 个人主页上的 API Token；传空字符串表示重新自动探测。
    persist : bool
        是否写入缓存目录的 token.txt，方便下次自动读取。
    """
    token = (token or "").strip()
    settings.token = token or _discover_token()
    if persist and token:
        (settings.cache_dir / "token.txt").write_text(token, encoding="utf-8")
    if settings.verbose:
        state = "已设置" if settings.token else "未设置（将只能读取本地缓存）"
        print("[cqfm1] token " + state)


def describe() -> str:
    """模块级快捷方式，等价于 settings.describe()。"""
    return settings.describe()
