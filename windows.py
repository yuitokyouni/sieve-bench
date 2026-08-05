"""実データの窓を、**暦の日付を持ったまま**作る。

以前は窓が「指数名と配列」だけだった。そのため窓どうしの依存を扱うときに、
`robustness.py` が **窓番号 k を「同じ時期」の代理**に使っていた。これは誤りである：

  - 指数ごとに営業日数が違う（FTSE 6317 日、日経 6122 日）ので、
    同じ k でも暦の上では最大 1 年近くずれる
  - EURO STOXX 50 は 2007-03 開始なので、k=0 が他の指数の k=0 と
    **5年半ずれる**

つまり実効標本数の推定が、揃っていないものを揃っていると見なして計算されていた。
ここでは各窓に実際の開始日・終了日を持たせ、依存の単位を暦で定義し直す。

窓の依存には2種類ある。分けて扱う：

  1. **同一指数の窓どうしの重なり。**窓長 1000・ストライド 250 なので、
     隣の窓とは 75% を共有する。
  2. **異なる指数の同時期の窓どうしの相関。**2008年や2020年の窓は6指数すべてに
     同時に現れる。EURO STOXX 50 は DAX 構成銘柄を含むので一部は二重計上。

`calendar_blocks()` は開始日を暦の幅で区切ってブロックにする。幅を窓長（約4年）に
とれば 1 も 2 も同時に畳み込め、幅を1年にとれば 2 だけを畳み込む。

**完全に独立なブロックは作れない。**ストライド 250・窓長 1000 なので窓は指数ごとに
数珠つなぎになっており、「暦で重ならない窓の集合」に分けようとすると全体が
1個の連結成分になる。区切りの直前と直後の窓は最大 99% を共有する。
ブロック化はこの継ぎ目を許した近似である。だからこそ幅を変えた版を並べて、
**結論が依存の仮定にどれだけ左右されるか**を出す（`separation.py` の3段）。
"""

import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

WINDOW = 1000      # 1本の系列の長さ（約4年の営業日）
STRIDE = 250       # 実データの窓をずらす幅


def load_series():
    """指数ごとに (対数リターン, 各リターンの日付) を返す。

    リターンは各指数の標準偏差で割る。全統計量はスケール不変なので
    結果は変わらないが、振幅の違いを「差」として拾わないことを明示するため。
    """
    if not os.path.isdir(DATA) or not any(
            f.endswith(".json") for f in os.listdir(DATA)):
        sys.exit("data/ が空。先に `python3 fetch.py` を実行すること。\n"
                 "（指数データは再配布しないので同梱されていない）")
    out = {}
    for fn in sorted(os.listdir(DATA)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(DATA, fn)))["chart"]["result"][0]
        ts = d["timestamp"]
        cl = d["indicators"]["quote"][0]["close"]
        good = [(t, c) for t, c in zip(ts, cl) if c is not None]
        c = np.array([g[1] for g in good], dtype=float)
        days = [dt.datetime.utcfromtimestamp(g[0]).date() for g in good]
        r = np.diff(np.log(c))
        # r[i] は days[i] → days[i+1] のリターン。到着日を日付とする。
        d1 = days[1:]
        m = np.isfinite(r)
        r, d1 = r[m], [x for x, k in zip(d1, m) if k]
        out[fn[:-5]] = (r / r.std(), d1)
    return out


class Window:
    """1本の実データ窓。統計量の計算対象と、その素性。"""

    __slots__ = ("index", "values", "start", "end", "block")

    def __init__(self, index, values, start, end):
        self.index = index
        self.values = values
        self.start = start
        self.end = end
        self.block = None

    def __repr__(self):
        return f"<{self.index} {self.start}..{self.end}>"


def real_windows(series, window=WINDOW, stride=STRIDE):
    """日付つきの窓を作る。並びは (指数, 開始日) 順。"""
    ws = []
    for name in sorted(series):
        r, days = series[name]
        for s in range(0, len(r) - window + 1, stride):
            ws.append(Window(name, r[s:s + window], days[s], days[s + window - 1]))
    ws.sort(key=lambda w: (w.start, w.index))
    return ws


# ブロック幅（暦日）。窓長 1000 営業日はおよそ 1450 暦日。
BLOCK_WIDTHS = {
    "span": 1450,   # 窓長と同じ幅。重なりも指数間相関も畳み込む（最も保守的）
    "year": 365,    # 1年幅。指数間相関だけを畳み込む
}


def calendar_blocks(windows, width_days=BLOCK_WIDTHS["span"]):
    """**開始日**を暦の幅 `width_days` で区切ってブロック番号を返す。

    同じブロックに入るのは
      - 同時期の別指数の窓（指数間の相関）
      - 同一指数の隣接した窓（時間的な重なり）
    の両方である。幅を窓長にとれば後者もほぼ畳み込まれる。

    返り値はブロック番号の配列（`windows` と同じ長さ）。番号は 0 から詰める。
    """
    t0 = min(w.start for w in windows)
    raw = np.array([(w.start - t0).days // width_days for w in windows])
    uniq = {v: i for i, v in enumerate(sorted(set(raw.tolist())))}
    labels = np.array([uniq[v] for v in raw.tolist()], dtype=int)
    for w, l in zip(windows, labels):
        w.block = int(l)
    return labels


def describe_blocks(windows, labels):
    """ブロックごとの (期間, 本数, 指数) を人が読める形で返す。"""
    rows = []
    for b in sorted(set(labels)):
        sel = [w for w, l in zip(windows, labels) if l == b]
        rows.append({
            "block": int(b),
            "start": str(min(w.start for w in sel)),
            "end": str(max(w.end for w in sel)),
            "n": len(sel),
            "indices": sorted({w.index for w in sel}),
        })
    return rows
