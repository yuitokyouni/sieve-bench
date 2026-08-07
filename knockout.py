"""機構を1つだけ抜いて、どの統計量が気づくかを測る。

    python3 knockout.py     # → knockout.json（3分ほど）

## なぜこれが要るのか

`separation.py` の行列は「実データ vs 生成器」である。そこで
KS(実データ, GJR-t) = 0.53 が出ても、**その 0.53 が何に由来するかは識別できない。**
生成器と実データは同時に何もかも違うからである：機構の欠落、市場ごとの
ボラティリティ水準、裾の重さ、持続性、構造変化、推定パラメータのずれ。

つまりあれは **「生成器 × 統計量」の行列**であって、まだ
**「既知の欠落 × 検出器」の行列ではない。**

ここでは比較の両側を、**1つのパラメータ以外まったく同じ**にする。
基準は指数ごとに当てた GJR-t（クラスタリング・重い裾・非対称性を全部持つ）。
そこから機構を1つだけ抜いた版と比べる。

    no_asymmetry   (alpha, gamma) → (alpha + gamma/2, 0)
    no_fat_tails   nu → 30（実質正規）
    no_clustering  (alpha, gamma, beta) → (0, 0, 0)

**`no_asymmetry` は α に γ/2 を足し戻している。**単に γ=0 にすると
持続性 α+γ/2+β が 0.988 から 0.876 に落ち、クラスタリングまで一緒に
抜けてしまう。γ/2 を α に移せば、r² に掛かる係数の期待値も持続性も
無条件分散も変わらず、**符号への依存だけ**が消える。これが正しい ablation である。

`no_clustering` は ω を無条件分散に合わせ直す（全統計量はスケール不変なので
水準自体は効かないが、系列の性質を揃えるため）。

## この設計の利点

**依存の問題が消える。**両側とも独立な生成器の実行なので、窓の重なりも
指数間の相関も無い。標準の置換検定がそのまま厳密に使える。
`separation.py` の6ブロック制約（p の刻み 0.016、検出力 8.7%）はここには無い。
だから小さな効果も見える。

## この設計の限界

言えるのは「**GJR-t という族の中で**、その機構を抜いたことに気づくか」までである。
現実の非対称性が GJR の形をしている保証は無い。実データとの比較
（`separation.py`）を置き換えるものではなく、そちらの解釈を可能にするものである。
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from facts import BATTERY, evaluate                       # noqa: E402
from generators import build_contexts, gjr_t              # noqa: E402
from resampling import benjamini_hochberg, iid_perm_test, ks_stat  # noqa: E402
from windows import WINDOW, load_series                   # noqa: E402

N_RUNS = 400          # 片側あたり。両側とも独立なので大きく取れる
N_PERM = 4000
SEED = 20260802

KNOCKOUTS = {
    "no_asymmetry":  "非対称性だけを抜く（持続性・無条件分散は保つ）",
    "no_fat_tails":  "裾の重さだけを抜く（nu → 30、実質正規）",
    "no_clustering": "クラスタリングだけを抜く（alpha=gamma=beta=0）",
}


def ablate(params, which):
    """GJR-t の (omega, alpha, gamma, beta, nu) から機構を1つ抜く。"""
    o, a, g, b, nu = params
    if which == "full":
        return (o, a, g, b, nu)
    if which == "no_asymmetry":
        # gamma/2 を alpha に移す。r^2 の係数の期待値も持続性も変わらない。
        return (o, a + g / 2.0, 0.0, b, nu)
    if which == "no_fat_tails":
        return (o, a, g, b, 30.0)
    if which == "no_clustering":
        uncond = o / max(1e-8, 1.0 - a - g / 2.0 - b)
        return (uncond, 0.0, 0.0, 0.0, nu)
    raise ValueError(which)


def run_side(ctxs, which, alloc, rng):
    """各指数のパラメータから機構を抜いた系列を、割り当てぶん生成して評価する。"""
    out = []
    for name, k in alloc.items():
        base = ctxs[name]["gjr_t"]
        p = ablate(base, which)
        c = {"gjr_t": p}
        out += [evaluate(gjr_t(WINDOW, rng, c)) for _ in range(k)]
    return out


def main():
    rng = np.random.default_rng(SEED)
    series = load_series()
    ctxs = build_contexts(series, verbose=True)

    names = sorted(series)
    per = N_RUNS // len(names)
    alloc = {n: per for n in names}
    for n in names[:N_RUNS - per * len(names)]:
        alloc[n] += 1
    print(f"\n基準 = 指数ごとに当てた GJR-t、片側 {sum(alloc.values())} 実行"
          f"（指数あたり {per} 前後）\n")
    print(f"{'指数':<8}{'omega':>9}{'alpha':>9}{'gamma':>9}{'beta':>9}{'nu':>8}"
          f"{'持続性':>9}")
    print("-" * 61)
    for n in names:
        o, a, g, b, nu = ctxs[n]["gjr_t"]
        print(f"{n:<8}{o:>9.4f}{a:>9.4f}{g:>9.4f}{b:>9.4f}{nu:>8.2f}"
              f"{a + g / 2 + b:>9.4f}")

    full = run_side(ctxs, "full", alloc, rng)
    stats_names = list(BATTERY.keys())

    out = {"config": {"n_runs": N_RUNS, "n_perm": N_PERM, "seed": SEED,
                      "window": WINDOW, "alloc": alloc,
                      "base": {n: list(ctxs[n]["gjr_t"]) for n in names}},
           "knockouts": KNOCKOUTS, "ks": {}, "pvalue": {}, "q": {}, "shift": {}}

    for kname in KNOCKOUTS:
        side = run_side(ctxs, kname, alloc, rng)
        out["ks"][kname], out["pvalue"][kname], out["shift"][kname] = {}, {}, {}
        for s in stats_names:
            a = [d[s] for d in full]
            b = [d[s] for d in side]
            k, p, _ = iid_perm_test(a, b, ks_stat, rng, N_PERM)
            out["ks"][kname][s] = k
            out["pvalue"][kname][s] = p
            fa = np.nanmedian(a)
            fb = np.nanmedian(b)
            out["shift"][kname][s] = float(fb - fa)
        print(f"  {kname} 済み", flush=True)

    flat = [(kn, s) for kn in KNOCKOUTS for s in stats_names]
    q = benjamini_hochberg([out["pvalue"][kn][s] for kn, s in flat])
    for (kn, s), qq in zip(flat, q):
        out["q"].setdefault(kn, {})[s] = float(qq) if np.isfinite(qq) else None

    json.dump(out, open(os.path.join(HERE, "knockout.json"), "w"),
              ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------ 表示
    w0 = max(len(s) for s in stats_names) + 1
    print("\n\n機構を1つだけ抜いたときに、統計量が気づくか")
    print("KS（括弧内は置換検定の p 値）。両側とも独立な生成器なので厳密。\n")
    hdr = "統計量".ljust(w0) + "".join(k[:16].rjust(20) for k in KNOCKOUTS)
    print(hdr)
    print("-" * len(hdr))
    for s in stats_names:
        row = s.ljust(w0)
        for kn in KNOCKOUTS:
            k, p = out["ks"][kn][s], out["pvalue"][kn][s]
            row += (f"{k:.2f}({p:.4f})" if np.isfinite(k) else "—").rjust(20)
        print(row)
    print("-" * len(hdr))

    for kn, desc in KNOCKOUTS.items():
        rank = sorted(((out["ks"][kn][s], s) for s in stats_names
                       if np.isfinite(out["ks"][kn][s])), reverse=True)
        print(f"\n{kn} —— {desc}")
        print("  よく気づく: " + ", ".join(f"{s} {v:.2f}" for v, s in rank[:4]))
        blind = [s for v, s in rank if out["q"][kn][s] is not None
                 and out["q"][kn][s] >= 0.05]
        print(f"  気づかない（BH q≥0.05）: "
              + (", ".join(blind) if blind else "なし"))

    print("\n**これは「既知の欠落 × 検出器」の行列である。**両側の違いは")
    print("パラメータ1つ分だけなので、差が出たならその機構に反応したと言える。")
    print("ただし言えるのは GJR-t 族の中での話に限られる。")


if __name__ == "__main__":
    main()
