"""金融リターンのスタイライズドファクツを、1本の系列から1つの数に落とす関数群。

各関数は長さ N の対数リターン配列を受け取り、スカラーを返す。
全てスケール不変にしてある（系列を定数倍しても値が変わらない）ので、
実データと生成データの比較で「振幅が違うだけ」の差を拾わない。

参照：Cont (2001), Empirical properties of asset returns: stylized facts and
statistical issues, Quantitative Finance 1(2).
"""

import numpy as np


def _acf(x, lag):
    x = x - x.mean()
    d = (x * x).sum()
    if d <= 0:
        return np.nan
    return float((x[lag:] * x[:-lag]).sum() / d)


def excess_kurtosis(r):
    """裾の重さ。正規分布なら 0。"""
    z = (r - r.mean()) / r.std()
    return float((z ** 4).mean() - 3.0)


def hill_right(r, frac=0.05):
    """右裾の Hill 推定量の逆数（テール指数 alpha）。小さいほど裾が重い。"""
    return _hill(r[r > 0], frac)


def hill_left(r, frac=0.05):
    """左裾。"""
    return _hill(-r[r < 0], frac)


def _hill(x, frac):
    x = np.sort(x)[::-1]
    k = max(int(len(x) * frac), 10)
    if len(x) <= k or x[k] <= 0:
        return np.nan
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def acf_return_1(r):
    """リターンそのものの1次自己相関。実データではほぼ 0 になる。"""
    return _acf(r, 1)


def acf_abs_1(r):
    """絶対値リターンの1次自己相関。ボラティリティ・クラスタリングの直接指標。"""
    return _acf(np.abs(r), 1)


def acf_abs_20(r):
    """ラグ20での絶対値自己相関。実データでは「まだ残っている」のが特徴。"""
    return _acf(np.abs(r), 20)


def acf_abs_decay(r, lags=50):
    """絶対値自己相関の減衰の遅さ。log-log 回帰の傾き（負、絶対値が小さいほど遅い）。"""
    a = np.array([_acf(np.abs(r), l) for l in range(1, lags + 1)])
    ls = np.arange(1, lags + 1)
    m = np.isfinite(a) & (a > 1e-6)
    if m.sum() < 10:
        return np.nan
    return float(np.polyfit(np.log(ls[m]), np.log(a[m]), 1)[0])


def ljung_box_sq(r, lags=20):
    """二乗リターンの Ljung-Box 統計量の log。分散の時間依存の総量。"""
    n = len(r)
    x = r ** 2
    s = 0.0
    for l in range(1, lags + 1):
        a = _acf(x, l)
        if np.isfinite(a):
            s += a * a / (n - l)
    return float(np.log(n * (n + 2) * s + 1e-12))


def aggregational_gaussianity(r, scale=5):
    """時間集約でどれだけ正規に近づくか。1日と scale 日の超過尖度の差。

    実データでは集約すると尖度が落ちる。落ちない系列は集約構造が誤っている。
    """
    n = (len(r) // scale) * scale
    agg = r[:n].reshape(-1, scale).sum(axis=1)
    if len(agg) < 50:
        return np.nan
    return float(excess_kurtosis(r) - excess_kurtosis(agg))


def leverage(r, lags=5):
    """レバレッジ効果。r_t と将来の |r| の相関。実データでは負。"""
    a = np.abs(r)
    out = []
    for k in range(1, lags + 1):
        x, y = r[:-k], a[k:]
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            out.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)))
    return float(np.mean(out)) if out else np.nan


def vol_of_vol(r, win=21):
    """ボラティリティ自体のばらつき（間欠性）。21日実現ボラの log の標準偏差。"""
    n = (len(r) // win) * win
    v = r[:n].reshape(-1, win).std(axis=1)
    v = v[v > 0]
    if len(v) < 10:
        return np.nan
    return float(np.log(v).std())


def multiscaling(r, qs=(1.0, 2.0), taus=(1, 2, 5, 10, 20)):
    """マルチスケーリング。E|r_tau|^q ~ tau^zeta(q) の zeta が q に非線形か。

    単純な拡散なら zeta(q)=q/2 で線形。実データはここから外れる。
    返すのは zeta(1) - zeta(2)/2（単一スケーリングなら 0）。
    """
    z = {}
    for q in qs:
        m = []
        for t in taus:
            n = (len(r) // t) * t
            agg = r[:n].reshape(-1, t).sum(axis=1)
            if len(agg) < 30:
                return np.nan
            m.append(np.log((np.abs(agg) ** q).mean()))
        z[q] = float(np.polyfit(np.log(taus), m, 1)[0])
    return float(z[1.0] - z[2.0] / 2.0)


def gain_loss_asymmetry(r):
    """利得と損失の非対称。標準化リターンの3次モーメント。"""
    z = (r - r.mean()) / r.std()
    return float((z ** 3).mean())


def variance_ratio_20(r, q=20):
    """20日でまとめた分散が、1日分散の20倍からどれだけ外れるか。

    リターンが独立なら分散は期間に比例するので 1 になる。
    1 未満は平均回帰（まとめると打ち消し合う）、1 超はトレンド持続。

    **他の13個を作った後で足した14個目である。**ABM を通したところ、
    Franke-Westerhoff と Lux-Marchesi が実データの3〜5倍も強く平均回帰
    しているのに、13個のどれもそれを直接には見ていなかった。
    重なりのない20日ブロックを使う（長さ1000なら50ブロック）。
    """
    n = (len(r) // q) * q
    if n < q * 10:
        return np.nan
    a = r[:n]
    v1 = a.var()
    if v1 <= 0:
        return np.nan
    return float((a.reshape(-1, q).sum(axis=1).var() / q) / v1)


def drift(r):
    """1標準偏差あたりの平均リターン。窓のシャープレシオに相当する。

    **15個目。他の14個が全滅した穴を塞ぐために足した。**

    既存14個のうち6個（尖度・gain_loss_asymmetry・acf_return_1・
    aggregational_gaussianity・vol_of_vol・variance_ratio_20）は平均を
    引いてから計算するのでドリフトに完全に盲目、残り8個も |r| 経由で
    間接的に動くだけである。つまり **バッテリー全体がドリフトを見ていなかった。**

    較正した ABM を累積リターンで描いたところ、3モデルが実データの
    4〜10倍のドリフトを持つほぼ直線のランプになっていた。目で見れば
    一発で分かる異常を、14個のどれも検出できなかった。

    スケール不変性は保たれている（分子・分母がともに定数倍される）。
    """
    s = float(np.std(r))
    if s <= 0 or not np.isfinite(s):
        return np.nan
    return float(np.mean(r) / s)


# 名前 → 関数。この並びがそのまま出力表の行になる。
BATTERY = {
    "excess_kurtosis": excess_kurtosis,
    "hill_right": hill_right,
    "hill_left": hill_left,
    "gain_loss_asymmetry": gain_loss_asymmetry,
    "acf_return_1": acf_return_1,
    "acf_abs_1": acf_abs_1,
    "acf_abs_20": acf_abs_20,
    "acf_abs_decay": acf_abs_decay,
    "ljung_box_sq": ljung_box_sq,
    "aggregational_gaussianity": aggregational_gaussianity,
    "leverage": leverage,
    "vol_of_vol": vol_of_vol,
    "multiscaling": multiscaling,
    "variance_ratio_20": variance_ratio_20,
    "drift": drift,
}


def evaluate(r):
    """全統計量を1本の系列に対して計算する。"""
    out = {}
    for name, fn in BATTERY.items():
        try:
            out[name] = fn(np.asarray(r, dtype=float))
        except Exception:
            out[name] = np.nan
    return out
