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
    """右裾の Hill 推定量の逆数（テール指数 alpha）。小さいほど裾が重い。

    **k（=frac）の選択に依存する。**極値理論でよく知られた罠で、
    実測でも k を 2.5%→10% と振ると実データの推定値が 3.54→2.97 に動き、
    garch_t に対する KS が 0.47→0.70 に変わる（`robustness.py`）。
    ここでは 5% に固定しているが、これは選択であって最適化ではない。
    """
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


def return_skewness(r):
    """標準化リターンの3次モーメント。ただの歪度である。

    **v0.1 では `gain_loss_asymmetry` という名前だった。**中身は昔から
    3次モーメントで、金融でいう gain/loss asymmetry —— 利得側と損失側の
    到達時間の非対称（Jensen, Johansen & Simonsen 2003 の inverse statistics）
    —— とは別物である。名前が実装を誤って表していた。

    改名しただけで値は変わらない。本来の gain/loss asymmetry は
    `gain_loss_asymmetry` として別に実装してある。
    """
    z = (r - r.mean()) / r.std()
    return float((z ** 3).mean())


def gain_loss_asymmetry(r, rho=4.0, horizon=250, tmax=30):
    """本来の意味での利得／損失の非対称。**到達時間**で測る。

    出典: Jensen, M. H., Johansen, A. & Simonsen, I. (2003).
    "Inverse statistics in economics: the gain-loss asymmetry."
    Physica A 324, 338-343.

    ある日から見て、累積リターンが +rho·sigma / -rho·sigma に達するまでの
    日数を比べる。株価指数では**短期では損失側のほうが早く到達する**。

    返すのは、短期（t <= tmax）での

        F_down(t) - F_up(t)      F は「t 日以内に到達した起点の割合」

    の平均。正なら「下げの方が早い」。

    ## v0.3 で2つのバグを直した。**両方とも値の符号を変えるほどのものだった。**

    **1. 選抜バイアス。**旧版は「両方向とも horizon 以内に到達した起点」だけを
    使っていた。しかし「片側だけ届かない」起点こそが、測りたい非対称そのものである。
    それを「観測できなかった」という理由で捨てていた。実測では horizon を
    250 → 2000 と延ばす（打ち切り 55% → 14%）だけで実データの値が
    **+0.398 → +0.043 に落ちた。**現象ではなく捨て方が値を作っていた。
    ここでは**全起点を使う部分分布**で比べるので、この選抜が無い。

    **2. ドリフト汚染。**長い地平ではドリフトが非対称性を上書きし、実データで
    符号が反転していた（t=200 日で -0.158）。**`invariance.py` はこれを検出して
    いた**（location 変換への反応が四分位幅の 0.63 倍、全16統計量中3位）のに、
    出力を読んでいなかった。ここでは**平均を引いてから測る**ので位置不変になり、
    `drift` との役割分担もはっきりする。

    直した後はパラメータに鈍い（horizon 150〜1000 で 0.0424〜0.0440）。
    **鈍いことは要件であって、たまたまではない**（`SPEC` で宣言し検査する）。

    **3次モーメントとは別物である。**対称な分布でも到達時間は非対称になりうる。
    """
    r = np.asarray(r, float)
    r = r - r.mean()                       # 位置不変にする（ドリフトは drift が見る）
    s = float(np.std(r))
    n = len(r)
    K = min(horizon, n - 1)
    if s <= 0 or not np.isfinite(s) or n - K < 100:
        return np.nan
    c = np.concatenate([[0.0], np.cumsum(r)])
    starts = np.arange(n - K)
    base = c[starts]
    lvl = rho * s
    up = np.zeros(len(starts), dtype=int)
    dn = np.zeros(len(starts), dtype=int)
    for k in range(1, K + 1):
        d = c[starts + k] - base
        up[(up == 0) & (d >= lvl)] = k
        dn[(dn == 0) & (d <= -lvl)] = k
    m = len(starts)
    T = min(tmax, K)
    return float(np.mean([(np.sum((dn > 0) & (dn <= t))
                           - np.sum((up > 0) & (up <= t))) / m
                          for t in range(1, T + 1)]))


def variance_ratio_20(r, q=20):
    """20日でまとめた分散が、1日分散の20倍からどれだけ外れるか。

    リターンが独立なら分散は期間に比例するので 1 になる。
    1 未満は平均回帰（まとめると打ち消し合う）、1 超はトレンド持続。

    **これは Lo & MacKinlay (1988) の分散比検定そのものである。**
    独立に作ってから気づいた。漸近分布を含む理論が既にあるので、
    次の版では彼らの標準誤差を使うべきである（現状は置換検定で代用）。

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

    **これは「有限のバッテリーには穴がある」という話ではない。**見落としたのは
    一次モーメントで、走らせる前に分かる設計ミスである。スケール不変にすると
    決めたのは意図的だったが、**位置に対しても不変になっていたことを誰も
    数え上げなかった。**統計量を足すときは、そのバッテリーが何に対して不変かを
    先に列挙すること。不変群に入る量は、定義上どうやっても見えない。

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
    "return_skewness": return_skewness,
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


# --------------------------------------------------------------- 設計契約
#
# **再発防止のための宣言。**v0.3 で `gain_loss_asymmetry` に2つのバグが出た。
# どちらも「既に測っていたのに読まなかった」種類である：
#
#   ・ドリフト汚染 → `invariance.py` が location 反応 0.63 と出していた
#   ・打ち切り依存 → 誰も horizon を振っていなかった
#
# そこで **意図を機械可読で宣言し、実測と食い違ったら落ちるようにする。**
# 報告書は読み飛ばせるが、失敗するテストは読み飛ばせない。
#
#   must_invariant … 設計上「不変であるべき」変換。破れていたらバグ
#                    （`invariance.py` が検査）
#   params         … 実装の自由度。**値がこれに鈍いことが要件。**
#                    振って四分位幅比で効果を測る（`sensitivity.py` が検査）
#
# スケール不変は全統計量の設計方針なので、全部に入っている。
# 位置不変は「水準ではなく形・依存を測る」統計量にだけ課す。
# `drift` は位置を測るのが仕事なので、位置不変を課さない。

SPEC = {
    "excess_kurtosis":           dict(must_invariant=("location", "scale"), params={}),
    "hill_right":                dict(must_invariant=("scale",),
                                      params={"frac": (0.025, 0.05, 0.10)}),
    "hill_left":                 dict(must_invariant=("scale",),
                                      params={"frac": (0.025, 0.05, 0.10)}),
    "return_skewness":           dict(must_invariant=("location", "scale"), params={}),
    "gain_loss_asymmetry":       dict(must_invariant=("location", "scale"),
                                      params={"rho": (3.0, 4.0, 6.0),
                                              "horizon": (150, 250, 500),
                                              "tmax": (20, 30, 50)}),
    "acf_return_1":              dict(must_invariant=("location", "scale"), params={}),
    "acf_abs_1":                 dict(must_invariant=("scale",), params={}),
    "acf_abs_20":                dict(must_invariant=("scale",), params={}),
    "acf_abs_decay":             dict(must_invariant=("scale",),
                                      params={"lags": (30, 50, 80)}),
    "ljung_box_sq":              dict(must_invariant=("scale",),
                                      params={"lags": (10, 20, 40)}),
    "aggregational_gaussianity": dict(must_invariant=("location", "scale"),
                                      params={"scale": (3, 5, 10)}),
    "leverage":                  dict(must_invariant=("scale",),
                                      params={"lags": (3, 5, 10)}),
    "vol_of_vol":                dict(must_invariant=("location", "scale"),
                                      params={"win": (10, 21, 42)}),
    "multiscaling":              dict(must_invariant=("scale",), params={}),
    "variance_ratio_20":         dict(must_invariant=("location", "scale"),
                                      params={"q": (10, 20, 40)}),
    "drift":                     dict(must_invariant=("scale",), params={}),
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
