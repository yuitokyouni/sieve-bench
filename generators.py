"""比較対象になる生成器の梯子。

狙いは「LLMシミュレーションを落とすこと」ではなく、
**バッテリーに識別力があるかどうかを先に測ること**である。
そのために、正解が分かっている生成器を並べる。

| 生成器 | 持っているもの | 役割 |
|---|---|---|
| `gaussian` | 何もない | ここで分けられない統計量は使い物にならない |
| `student_t` | 重い裾のみ | |
| `iid_bootstrap` | 実データの値そのもの。時間構造なし | |
| `block_bootstrap` | 短期の時間構造（ブロック長20） | |
| `garch_norm` | 対称なクラスタリング | |
| `garch_t` | 対称なクラスタリング＋重い裾 | |
| `gjr_t` | **非対称**なクラスタリング＋重い裾 | **後から足した。下を見よ** |
| `egarch_t` | 非対称（対数分散・指数型） | **後から足した** |

## gjr_t / egarch_t を足した理由（と、それが検証の分離になること）

v0.1 の結論は「効いたのはレバレッジ効果だけだった」だった。しかし
`garch_norm` / `garch_t` はどちらも**対称な**ボラティリティ・モデルである。
対称なモデルが非対称性の指標で落ちるのは、ほとんど定義から従う。
**梯子に非対称なモデルが1つも無い状態で「leverage が効く」と言っても、
何も試されていない。**

GJR-GARCH（Glosten, Jagannathan & Runkle 1993）と EGARCH（Nelson 1991）は
負のショックだけボラティリティを大きく上げる。**この2つは、`leverage` を
選ぶときには存在しなかった**ので、v0.1 の統計量選択に対して純粋に
held-out である。README の「統計量の選び方について」で言っている
選択用／評価用の分離が、ここで初めて1つ実現している。

## garch_t の推定を直した

v0.1 の `garch_t` は、**正規尤度で推定した**パラメータに後から t 分布の
イノベーションを載せていた。QMLE なので一致性はあるが、名前が示す
Student-t GARCH の同時最尤推定ではない。ここでは (omega, alpha, beta, nu) を
同時に当てる。gjr_t / egarch_t も同様。

推定は一度だけ行い `baselines.json` に書き出す（`separation.py`・
`robustness.py`・`run_power.py` が同じ値を使うため、かつ再現のため）。
"""

import hashlib
import json
import os

import numpy as np
from scipy import optimize, special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "baselines.json")
BURN = 500


# ------------------------------------------------------------------- 生成器


def gaussian(n, rng, ctx):
    return rng.standard_normal(n)


def student_t(n, rng, ctx):
    df = ctx["t_df"]
    x = rng.standard_t(df, n)
    return x / np.sqrt(df / (df - 2.0))


def iid_bootstrap(n, rng, ctx):
    pool = ctx["pool"]
    return rng.choice(pool, size=n, replace=True)


def block_bootstrap(n, rng, ctx, block=20):
    pool = ctx["pool"]
    nb = n // block + 1
    starts = rng.integers(0, len(pool) - block, size=nb)
    return np.concatenate([pool[s:s + block] for s in starts])[:n]


def _std_t(rng, df, m):
    """分散1に標準化した Student-t。"""
    return rng.standard_t(df, m) / np.sqrt(df / (df - 2.0))


def _garch_path(n, rng, omega, alpha, beta, innov):
    total = n + BURN
    e = innov(total)
    r = np.empty(total)
    s2 = omega / max(1e-8, 1 - alpha - beta)
    for i in range(total):
        r[i] = np.sqrt(s2) * e[i]
        s2 = omega + alpha * r[i] ** 2 + beta * s2
    return r[BURN:]


def garch_norm(n, rng, ctx):
    o, a, b = ctx["garch"]
    return _garch_path(n, rng, o, a, b, lambda m: rng.standard_normal(m))


def garch_t(n, rng, ctx):
    o, a, b, df = ctx["garch_t"]
    return _garch_path(n, rng, o, a, b, lambda m: _std_t(rng, df, m))


def gjr_t(n, rng, ctx):
    """GJR-GARCH(1,1,1) with Student-t。負のショックだけ係数が (alpha+gamma)。

        h_t = omega + (alpha + gamma·1[r_{t-1}<0]) r_{t-1}^2 + beta h_{t-1}
    """
    o, a, g, b, df = ctx["gjr_t"]
    total = n + BURN
    e = _std_t(rng, df, total)
    r = np.empty(total)
    h = o / max(1e-8, 1 - a - g / 2 - b)
    for i in range(total):
        r[i] = np.sqrt(h) * e[i]
        h = o + (a + (g if r[i] < 0 else 0.0)) * r[i] ** 2 + b * h
    return r[BURN:]


def egarch_t(n, rng, ctx):
    """EGARCH(1,1) with Student-t（Nelson 1991）。

        log h_t = omega + beta·log h_{t-1} + alpha(|z|-E|z|) + gamma·z
    """
    o, a, g, b, df = ctx["egarch_t"]
    total = n + BURN
    e = _std_t(rng, df, total)
    r = np.empty(total)
    eabs = _abs_mean_std_t(df)
    lh = o / max(1e-8, 1 - b)
    for i in range(total):
        h = np.exp(min(lh, 50.0))
        r[i] = np.sqrt(h) * e[i]
        lh = o + b * lh + a * (abs(e[i]) - eabs) + g * e[i]
    return r[BURN:]


GENERATORS = {
    "gaussian": gaussian,
    "student_t": student_t,
    "iid_bootstrap": iid_bootstrap,
    "block_bootstrap": block_bootstrap,
    "garch_norm": garch_norm,
    "garch_t": garch_t,
    "gjr_t": gjr_t,
    "egarch_t": egarch_t,
}

# 統計量を選んだ後に足した生成器。v0.1 の統計量選択に対して held-out である。
HELD_OUT = ("gjr_t", "egarch_t")


# --------------------------------------------------------------------- 推定


def _abs_mean_std_t(df):
    """分散1に標準化した Student-t の E|z|。"""
    return (2.0 * np.sqrt(df - 2.0) * np.exp(special.gammaln((df + 1) / 2)
                                             - special.gammaln(df / 2))
            / ((df - 1.0) * np.sqrt(np.pi)))


def _t_logpdf_const(df):
    return (special.gammaln((df + 1) / 2) - special.gammaln(df / 2)
            - 0.5 * np.log(np.pi * (df - 2.0)))


def fit_garch11(r):
    """GARCH(1,1) を正規尤度で当てる。omega, alpha, beta を返す。"""
    r = np.asarray(r, dtype=float)
    v = r.var()

    def nll(p):
        o, a, b = p
        if o <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        s2 = v
        s = 0.0
        for x in r:
            s += np.log(s2) + x * x / s2
            s2 = o + a * x * x + b * s2
        return s

    best, bv = (v * 0.05, 0.08, 0.90), np.inf
    for a0, b0 in ((0.08, 0.90), (0.05, 0.93), (0.12, 0.85)):
        res = optimize.minimize(
            nll, [v * (1 - a0 - b0), a0, b0], method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-8},
        )
        if res.fun < bv:
            best, bv = tuple(res.x), res.fun
    return best


def _nll_garch_t(p, r, v):
    o, a, b, df = p
    if o <= 0 or a < 0 or b < 0 or a + b >= 0.999 or df <= 2.05 or df > 50:
        return 1e10
    c = _t_logpdf_const(df)
    k = (df + 1) / 2
    s2, s = v, 0.0
    for x in r:
        z2 = x * x / s2
        s += 0.5 * np.log(s2) - c + k * np.log1p(z2 / (df - 2.0))
        s2 = o + a * x * x + b * s2
    return s


def _nll_gjr_t(p, r, v):
    o, a, g, b, df = p
    if (o <= 0 or a < 0 or b < 0 or a + g < 0 or a + g / 2 + b >= 0.999
            or df <= 2.05 or df > 50):
        return 1e10
    c = _t_logpdf_const(df)
    k = (df + 1) / 2
    h, s = v, 0.0
    for x in r:
        z2 = x * x / h
        s += 0.5 * np.log(h) - c + k * np.log1p(z2 / (df - 2.0))
        h = o + (a + (g if x < 0 else 0.0)) * x * x + b * h
    return s


def _nll_egarch_t(p, r, v):
    o, a, g, b, df = p
    if abs(b) >= 0.999 or df <= 2.05 or df > 50 or abs(a) > 2 or abs(g) > 2:
        return 1e10
    c = _t_logpdf_const(df)
    k = (df + 1) / 2
    eabs = _abs_mean_std_t(df)
    lh, s = np.log(v), 0.0
    for x in r:
        if lh > 50 or lh < -50:
            return 1e10
        h = np.exp(lh)
        z = x / np.sqrt(h)
        s += 0.5 * lh - c + k * np.log1p(z * z / (df - 2.0))
        lh = o + b * lh + a * (abs(z) - eabs) + g * z
    return s


def _minimize(fn, starts, r, v):
    best, bv = None, np.inf
    for x0 in starts:
        res = optimize.minimize(fn, x0, args=(r, v), method="Nelder-Mead",
                                options={"maxiter": 6000, "maxfev": 6000,
                                         "xatol": 1e-8, "fatol": 1e-8})
        if res.fun < bv:
            best, bv = tuple(float(x) for x in res.x), float(res.fun)
    return best, bv


def fit_all(r):
    """t 分布イノベーションのモデル群を同時最尤で当てる。"""
    r = np.asarray(r, float)
    v = float(r.var())
    out = {}
    out["garch"] = [float(x) for x in fit_garch11(r)]
    out["t_df"] = float(np.clip(stats.t.fit(r, floc=0)[0], 2.5, 30.0))
    out["garch_t"], _ = _minimize(
        _nll_garch_t,
        [[v * 0.05, 0.08, 0.90, 6.0], [v * 0.02, 0.05, 0.93, 4.0]], r, v)
    out["gjr_t"], _ = _minimize(
        _nll_gjr_t,
        [[v * 0.05, 0.02, 0.12, 0.90, 6.0], [v * 0.03, 0.05, 0.08, 0.91, 8.0]],
        r, v)
    out["egarch_t"], _ = _minimize(
        _nll_egarch_t,
        [[-0.10, 0.12, -0.08, 0.98, 6.0], [-0.05, 0.15, -0.05, 0.97, 8.0]],
        r, v)
    return out


def build_context(pool, cache=True, verbose=False):
    """実データから、生成器が必要とするパラメータを推定する。

    ここで当てたものが「合わせ込んだ量」になる。判定はここで使っていない
    統計量でやらないと意味が無い ── 力場検証と同じ作法。

    推定は重いので `baselines.json` に載せる。系列のハッシュを鍵にしてあるので、
    データが変われば自動で当て直す。
    """
    pool = np.asarray(pool, dtype=float)
    key = hashlib.sha256(np.round(pool, 10).tobytes()).hexdigest()[:32]

    fits = None
    if cache and os.path.exists(CACHE):
        blob = json.load(open(CACHE))
        if blob.get("key") == key and blob.get("version") == 2:
            fits = blob["fits"]
    if fits is None:
        if verbose:
            print("  基準線を推定中（初回のみ、1分ほど）…", flush=True)
        fits = fit_all(pool)
        if cache:
            json.dump({"key": key, "version": 2, "n": len(pool), "fits": fits},
                      open(CACHE, "w"), indent=1)

    ctx = {"pool": pool}
    ctx.update(fits)
    return ctx
