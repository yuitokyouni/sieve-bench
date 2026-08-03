"""識別力の測り直し。

**power = 2|AUC-0.5| は壊れている。**AUC = P(生成 > 実データ) は確率的順序、
つまり位置ずれしか見ない。中心が同じで形が違う分布に対してほぼゼロを返す：

    中央値同一・分散5倍差   power 0.138   KS p = 2.7e-13
    二峰性 vs 単峰         power 0.003   KS p = 2.2e-17

さらに **雑音床が測られていなかった。**同一分布同士でも 124 対 200 なら
power は平均 0.052、95%点 0.129 になる（絶対値で折り返すため、真値ゼロでも
正に膨らむ）。旧 power.json の 90 セル中 11 セルがこの帯域に入っており、
「その統計量は無力」と読める形で報告されていた。

ここでは3つ入れ替える:

  1. 検定統計量を **2標本 Kolmogorov-Smirnov** にする。分布関数の最大乖離なので
     位置・尺度・形のいずれのずれにも反応する。
  2. **置換検定で p 値**を出す。ラベルを入れ替えた帰無分布に対する位置で測るので、
     標本数の違いによる床が自動的に織り込まれる。
  3. **実データ同士の対照（天井）**を併記する。米欧の窓 vs アジアの窓で同じ量を
     測り、「実市場同士ですらこれだけ違う」水準を示す。これを超えられない
     統計量に生成器のゼロを要求しても意味がない。

出力は `separation.json`。旧 `power.json` は AUC 版として残す（比較のため）。
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facts import BATTERY, evaluate            # noqa: E402
from generators import GENERATORS, build_context  # noqa: E402
from run_power import load_returns, real_windows, WINDOW, STRIDE, N_RUNS, SEED  # noqa: E402

N_PERM = 2000

# 指数を地域で二分し、実データ同士の対照に使う
GROUP_A = {"gspc", "ftse", "gdaxi", "sx5e"}   # 米欧
GROUP_B = {"n225", "hsi"}                      # アジア


def ks_stat(a, b):
    """2標本 KS 統計量。分布関数の最大乖離。"""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def perm_test(a, b, rng, n_perm=N_PERM):
    """置換検定。返り値 (KS統計量, p値, 帰無の95%点)。

    ラベルを入れ替えた分布に対する位置で測るので、標本数の違いに由来する
    床は自動的に織り込まれる。
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan
    obs = ks_stat(a, b)
    pool = np.concatenate([a, b])
    na = len(a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pool)
        null[i] = ks_stat(p[:na], p[na:])
    pval = float((1.0 + np.sum(null >= obs)) / (n_perm + 1.0))
    return float(obs), pval, float(np.percentile(null, 95))


def main():
    rng = np.random.default_rng(SEED)
    series = load_returns()
    pool = np.concatenate(list(series.values()))
    print(f"指数 {len(series)} 本、リターン {len(pool)} 点", flush=True)

    ctx = build_context(series["gspc"])
    ctx["pool"] = pool

    wins = real_windows(series)
    real = [evaluate(w) for _, w in wins]
    names = [n for n, _ in wins]
    print(f"実データの窓 {len(wins)} 本", flush=True)

    results = {}
    for gname, gfn in GENERATORS.items():
        results[gname] = [evaluate(gfn(WINDOW, rng, ctx)) for _ in range(N_RUNS)]
        print(f"  {gname}: {N_RUNS} 回", flush=True)

    stats_names = list(BATTERY.keys())
    gen_names = list(GENERATORS.keys())

    ia = [i for i, n in enumerate(names) if n in GROUP_A]
    ib = [i for i, n in enumerate(names) if n in GROUP_B]
    print(f"実データ同士の対照: 米欧 {len(ia)} 本 vs アジア {len(ib)} 本", flush=True)

    out = {"config": {"window": WINDOW, "stride": STRIDE, "n_runs": N_RUNS,
                      "n_perm": N_PERM, "seed": SEED, "n_real_windows": len(wins),
                      "group_a": sorted(GROUP_A), "group_b": sorted(GROUP_B),
                      "n_a": len(ia), "n_b": len(ib)},
           "ks": {}, "pvalue": {}, "null95": {}, "real_vs_real": {}}

    for s in stats_names:
        rv = [d[s] for d in real]
        out["ks"][s], out["pvalue"][s], out["null95"][s] = {}, {}, {}
        for g in gen_names:
            k, p, n95 = perm_test(rv, [d[s] for d in results[g]], rng)
            out["ks"][s][g] = k
            out["pvalue"][s][g] = p
            out["null95"][s][g] = n95
        ka, pa, _ = perm_test([rv[i] for i in ia], [rv[i] for i in ib], rng)
        out["real_vs_real"][s] = {"ks": ka, "pvalue": pa}
        print(f"  {s}", flush=True)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "separation.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ------------------------------------------------------------------ 表示
    w0 = max(len(s) for s in stats_names) + 1
    print("\nKS 統計量（分布関数の最大乖離）。括弧内は置換検定の p 値\n")
    hdr = "統計量".ljust(w0) + "".join(g[:11].rjust(16) for g in gen_names) + "実データ同士".rjust(16)
    print(hdr)
    print("-" * len(hdr))
    for s in stats_names:
        row = s.ljust(w0)
        for g in gen_names:
            k = out["ks"][s][g]
            p = out["pvalue"][s][g]
            row += (f"{k:.2f}({p:.3f})" if np.isfinite(k) else "—").rjust(16)
        rr = out["real_vs_real"][s]
        row += (f"{rr['ks']:.2f}({rr['pvalue']:.3f})"
                if np.isfinite(rr["ks"]) else "—").rjust(16)
        print(row)
    print("-" * len(hdr))
    print("\n右端が天井：実市場同士でもこれだけ違う。"
          "これを超えない生成器を『現実的』とは言えない一方、\n"
          "これを下回る差を統計量が拾えないことは欠陥ではない。")

    print("\n5% 水準で全生成器を棄却できた統計量：")
    for s in stats_names:
        ps = [out["pvalue"][s][g] for g in gen_names if np.isfinite(out["pvalue"][s][g])]
        if ps and max(ps) < 0.05:
            worst = max(ps)
            print(f"  {s:<28} 最悪 p = {worst:.4f}")


if __name__ == "__main__":
    main()
