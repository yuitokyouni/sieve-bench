"""比較対象になる生成器。

狙いは「LLMシミュレーションを落とすこと」ではなく、
**バッテリーに識別力があるかどうかを先に測ること**である。

そのために、正解が分かっている生成器を並べる。

- gaussian      : 何もかも間違っている。ここで分けられない統計量は使い物にならない
- student_t     : 裾は重いが時間構造が無い
- iid_bootstrap : **周辺分布は実データと完全に同一**。時間構造だけが無い
                  → これと分けられない統計量は「裾しか見ていない」ことの証明になる
- block_bootstrap: 短期の時間構造だけ持つ
- garch_norm    : クラスタリングを持つ標準的な計量経済モデル
- garch_t       : 上に加えて裾も重い。**これが強い基準線**
                  → これと分けられない統計量は、実務で情報を持たない
"""

import numpy as np
from scipy import optimize, stats


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


def _garch_path(n, rng, omega, alpha, beta, innov):
    burn = 500
    total = n + burn
    e = innov(total)
    r = np.empty(total)
    s2 = omega / max(1e-8, 1 - alpha - beta)
    for i in range(total):
        r[i] = np.sqrt(s2) * e[i]
        s2 = omega + alpha * r[i] ** 2 + beta * s2
    return r[burn:]


def garch_norm(n, rng, ctx):
    o, a, b = ctx["garch"]
    return _garch_path(n, rng, o, a, b, lambda m: rng.standard_normal(m))


def garch_t(n, rng, ctx):
    o, a, b = ctx["garch"]
    df = ctx["t_df"]
    sc = np.sqrt(df / (df - 2.0))
    return _garch_path(n, rng, o, a, b, lambda m: rng.standard_t(df, m) / sc)


GENERATORS = {
    "gaussian": gaussian,
    "student_t": student_t,
    "iid_bootstrap": iid_bootstrap,
    "block_bootstrap": block_bootstrap,
    "garch_norm": garch_norm,
    "garch_t": garch_t,
}


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


def build_context(pool):
    """実データから、生成器が必要とするパラメータを推定する。

    ここで当てたものが「合わせ込んだ量」になる。判定はここで使っていない
    統計量でやらないと意味が無い ── 力場検証と同じ作法。
    """
    df = float(np.clip(stats.t.fit(pool, floc=0)[0], 2.5, 30.0))
    return {"pool": np.asarray(pool, dtype=float), "t_df": df,
            "garch": fit_garch11(pool)}
