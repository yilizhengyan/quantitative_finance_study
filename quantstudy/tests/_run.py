# -*- coding: utf-8 -*-
"""不用 pytest 的极简测试跑分器：逐个调用 test_* 函数。"""
import sys, traceback, types
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\yiliz\Desktop\cqfm1")

# pytest 只是用来提供 raises 的，这里给个假模块
fake = types.ModuleType("pytest")
class _Raises:
    def __init__(self, exc): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError("expected %s but nothing raised" % self.exc.__name__)
        return issubclass(t, self.exc)
fake.raises = lambda exc: _Raises(exc)
sys.modules["pytest"] = fake

import test_smoke as T

ok = fail = 0
for name in dir(T):
    if not name.startswith("test_"):
        continue
    fn = getattr(T, name)
    try:
        fn()
        print("PASS", name); ok += 1
    except Exception as exc:
        print("FAIL", name, "->", type(exc).__name__, str(exc)[:150]); fail += 1
print("\n%d passed, %d failed" % (ok, fail))
