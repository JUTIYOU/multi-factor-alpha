# -*- coding: utf-8 -*-
"""BaoStock 连接管理: 登录上下文 + 重试 + 结果集转DataFrame."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import baostock as bs
import pandas as pd

MAX_RETRY = 3
RETRY_SLEEP = 2.0  # 秒,重试间隔


class Session:
    """with Session() as s: rs = s.query(...)
    并发登录有服务端上限, 登录失败按退避重试.
    """

    def __init__(self, login_retry: int = 8):
        self.login_retry = login_retry
        # 账号从环境变量读,默认匿名; 凭证不落代码/配置
        self.user = os.environ.get("BAOSTOCK_USER", "anonymous")
        self.pw = os.environ.get("BAOSTOCK_PASSWORD", "123456")

    def _login(self):
        return bs.login(user_id=self.user, password=self.pw)

    def __enter__(self):
        last = None
        for i in range(self.login_retry):
            try:
                lg = self._login()
                if lg.error_code == "0":
                    return self
                last = f"{lg.error_code} {lg.error_msg}"
            except Exception as e:
                last = e
            time.sleep(3.0 * (i + 1))
        raise ConnectionError(f"baostock login failed after retries: {last}")

    def __exit__(self, *exc):
        bs.logout()
        return False

    def relogin(self) -> bool:
        """会话被服务端踢下(10001001)后重登; 成功返回True."""
        try:
            bs.logout()
        except Exception:
            pass
        for i in range(self.login_retry):
            try:
                lg = self._login()
                if lg.error_code == "0":
                    return True
            except Exception:
                pass
            time.sleep(3.0 * (i + 1))
        return False


def rs_to_df(rs) -> pd.DataFrame:
    """BaoStock结果集迭代为DataFrame; 空结果集返回带列名的空表."""
    if rs.error_code != "0":
        raise RuntimeError(f"query failed: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def query_with_retry(fn, *args, **kwargs) -> pd.DataFrame:
    """对bs.query_*做有限重试; error_code!=0或异常时重试, 最终失败抛出."""
    last = None
    for i in range(MAX_RETRY):
        try:
            rs = fn(*args, **kwargs)
            if rs.error_code == "0":
                return rs_to_df(rs)
            last = RuntimeError(f"{rs.error_code} {rs.error_msg}")
        except Exception as e:  # 网络层异常
            last = e
        time.sleep(RETRY_SLEEP * (i + 1))
    raise last if isinstance(last, Exception) else RuntimeError(str(last))


class Checkpoint:
    """断点续传: JSON记录已完成键. ckpt.done(key) / ckpt.mark(key)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self._done = set(json.loads(self.path.read_text(encoding="utf-8")))
        else:
            self._done = set()

    def done(self, key: str) -> bool:
        return key in self._done

    def mark(self, key: str) -> None:
        self._done.add(key)
        self.path.write_text(json.dumps(sorted(self._done)), encoding="utf-8")

    def __len__(self):
        return len(self._done)


def load_yaml(path: str | Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def atomic_to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """先写.tmp再os.replace, 避免并发读到写了一半的parquet."""
    import os
    path = Path(path)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
