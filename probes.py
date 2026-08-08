"""プローブ族 —— 「学習ベースの統計量」のための高速な当てはめ。

`reversal_learnability_gap`（`facts.py`）が使う。統計量の側の説明はそちらに、
**観測者としてのプローブ族の宣言**はここに置く。

## なぜ「プローブ族」を宣言するのか

Finzi et al. (2026) の枠組みでは、情報量は観測者の計算資源に相対的である
（epiplexity / time-bounded entropy）。学習ベースの統計量も同じで、
**測定値はどのモデル族で測るかに依存する。**これは欠陥ではなく定義の一部なので、
プローブ族は隠さず宣言する：

    PROBE = GJR-GARCH(1,1,1) + Student-t イノベーション、5 パラメータ
            (omega, alpha, gamma, beta, nu)、最尤推定（Nelder-Mead、多点開始）

GJR-t を選ぶ理由：
  - 符号チャネル（gamma·1[r<0]）を持つので、**時間の向きを利用できる**
    （対称な族は向きの主要な手がかりを構造的に使えない）
  - 5 パラメータと小さく、窓 1000 点で当てても過学習しない
  - この梯子の最強格の基準線なので、「この族にとっての学びやすさ」が
    そのまま計量経済の実務の言葉になる

## 実装の注意

- NLL は `generators._nll_gjr_t` と**同じ式**。numba で JIT してあるだけで、
  値は一致する（`python3 probes.py` が突き合わせる）。numba が無い環境では
  純 Python 版に自動で落ちる（遅いが同じ値）。
- 逆方向の当てはめは順方向の解から温間開始する。帰無（可逆な系列）では
  両方向の最適解がほぼ同じ場所にあるため、これで最適化ノイズが大きく減る。
"""

import math

import numpy as np
from scipy import optimize

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True)
def _gjr_t_nll_core(p0, p1, p2, p3, p4, r, v):
    """GJR-t の負対数尤度。generators._nll_gjr_t と同じ式（JIT 版）。"""
    o, a, g, b, df = p0, p1, p2, p3, p4
    if (o <= 0.0 or a < 0.0 or b < 0.0 or a + g < 0.0
            or a + g / 2.0 + b >= 0.999 or df <= 2.05 or df > 50.0):
        return 1e10
    # Student-t の対数密度の定数項。自然対数で持つ（generators と同じ）
    c = (math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0)
         - 0.5 * math.log(math.pi * (df - 2.0)))
    k = (df + 1.0) / 2.0
    h = v
    s = 0.0
    for i in range(r.shape[0]):
        x = r[i]
        z2 = x * x / h
        s += 0.5 * math.log(h) - c + k * math.log(1.0 + z2 / (df - 2.0))
        if x < 0.0:
            h = o + (a + g) * x * x + b * h
        else:
            h = o + a * x * x + b * h
    return s


def gjr_t_nll(params, r, v):
    o, a, g, b, df = params
    return _gjr_t_nll_core(float(o), float(a), float(g), float(b), float(df),
                           np.ascontiguousarray(r, dtype=np.float64), float(v))


DEFAULT_STARTS = ((0.02, 0.02, 0.12, 0.90, 6.0),
                  (0.05, 0.05, 0.08, 0.88, 8.0))


def fit_gjr_t(r, starts=DEFAULT_STARTS, maxiter=3000):
    """GJR-t を最尤で当てる。返り値 (1点あたり NLL, パラメータ)。"""
    r = np.ascontiguousarray(r, dtype=np.float64)
    v = float(r.var())
    best, bx = np.inf, None
    for x0 in starts:
        res = optimize.minimize(gjr_t_nll, x0, args=(r, v),
                                method="Nelder-Mead",
                                options={"maxiter": maxiter, "maxfev": maxiter,
                                         "xatol": 1e-7, "fatol": 1e-7})
        if res.fun < best:
            best, bx = float(res.fun), tuple(float(x) for x in res.x)
    return best / len(r), bx


def reversal_gap(r, starts=DEFAULT_STARTS, maxiter=3000):
    """順方向と逆方向に別々に当てた 1点あたり NLL の差（逆 − 順）。

    正 = 逆向きのほうがこの族には当てにくい = 時間の矢がこの観測者に見える。

    構成上の恒等式として **gap(rev(x)) = −gap(x)**（反対称）。したがって
    可逆な過程（順逆が同分布）では期待値が厳密に 0 になる。帰無が
    統計量の定義自体から出る——参照データが要らない。

    ## プロトコルは向きについて厳密に対称にしてある

    最初の実装は「順方向を先に当て、その解を逆方向の開始点に足す」だった。
    すると**逆方向だけ開始点が1個多く**、わずかに低い NLL を見つけやすくなり、
    可逆な帰無で中央値が −0.0002 に偏った（`--calibrate` で実測）。
    ここでは両方向を同じ開始点で当ててから、**互いの解を相手の温間開始として
    交換し**、それぞれ最小を取る。x ↔ rev(x) の入れ替えが手続き全体を入れ替える
    ので、反対称性 gap(rev x) = −gap(x) が構成から厳密に成り立つ。
    """
    r = np.ascontiguousarray(r, dtype=np.float64)
    # 位置・尺度不変を**ビット単位で**厳密にする。標準化だけだと x→x+c や
    # x→a·x の後に最終ビットが揺れ、Nelder-Mead の経路が分かれて値が 1e-6
    # 程度動く。設計契約（invariance.py の EXACT 判定）はビット一致を要求する
    # ので、標準化後に 1e-8 で量子化して入力を同一にする。分散1の系列に
    # 対する 1e-8 は統計的な分解能のはるか下である。
    r = np.round((r - r.mean()) / r.std(), 8)
    rb = r[::-1].copy()
    f1, f_par = fit_gjr_t(r, starts, maxiter)
    b1, b_par = fit_gjr_t(rb, starts, maxiter)
    f2, _ = fit_gjr_t(r, (b_par,), maxiter)
    b2, _ = fit_gjr_t(rb, (f_par,), maxiter)
    return min(b1, b2) - min(f1, f2)


# ---------------------------------------------------------------- 較正

def _calibrate():
    """統計量の帰無を、答えの分かっているデータで測る。

        python3 probes.py --calibrate     # → probes_calibration.json（10分ほど）

    3層を測る：

      1. **最適化ノイズの床** — 反対称性 gap(x) + gap(rev x) は恒等的に 0 の
         はずなので、その残差がそのまま最適化ノイズの大きさである
      2. **厳密に可逆な帰無** — iid な系列（gaussian / student-t / iid_bootstrap）
         は順逆が同分布なので、gap の期待値は 0。分布の幅が「窓1本の帰無」
      3. **理論値が非自明なもの** — 対称 GARCH は弱く不可逆でありうる
         （r² が非ガウス ARMA になるため）。ここは測って報告するだけで、
         0 を仮定しない
    """
    import json
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from windows import load_series, real_windows, WINDOW
    from generators import GENERATORS, build_contexts

    rng = np.random.default_rng(20260802)
    s = load_series()
    ctxs = build_contexts(s)
    ctx = dict(ctxs["gspc"]); ctx["pool"] = s["gspc"][0]
    N = 120

    out = {"n_per_cell": N, "window": WINDOW, "gap": {}}
    t0 = time.time()

    print("=== 1. 最適化ノイズの床（反対称性の残差）===")
    resid = []
    for _ in range(40):
        x = GENERATORS["garch_t"](WINDOW, rng, ctx)
        resid.append(reversal_gap(x) + reversal_gap(x[::-1].copy()))
    resid = np.abs(resid)
    out["antisymmetry_residual"] = {"median": float(np.median(resid)),
                                    "p95": float(np.percentile(resid, 95))}
    print(f"  |gap(x)+gap(rev x)| 中央値 {np.median(resid):.5f}  "
          f"95%点 {np.percentile(resid,95):.5f}")

    print("\n=== 2-3. 生成器ごとの gap 分布 ===")
    print(f"{'系列':<16}{'中央値':>10}{'四分位幅':>10}{'|gap|>0 の向き':>14}")
    for g in ("gaussian", "student_t", "iid_bootstrap",
              "garch_norm", "garch_t", "gjr_t", "egarch_t"):
        v = np.array([reversal_gap(GENERATORS[g](WINDOW, rng, ctx))
                      for _ in range(N)])
        out["gap"][g] = {"median": float(np.median(v)),
                         "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                         "frac_positive": float(np.mean(v > 0))}
        print(f"{g:<16}{np.median(v):>+10.5f}{out['gap'][g]['iqr']:>10.5f}"
              f"{np.mean(v>0):>13.0%}")

    ws = [w.values for w in real_windows(s)]
    v = np.array([reversal_gap(w) for w in ws])
    out["gap"]["real"] = {"median": float(np.median(v)),
                          "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                          "frac_positive": float(np.mean(v > 0))}
    print(f"{'実データ124窓':<16}{np.median(v):>+10.5f}"
          f"{out['gap']['real']['iqr']:>10.5f}{np.mean(v>0):>13.0%}")

    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, "probes_calibration.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n{time.time()-t0:.0f}s → probes_calibration.json")
    return 0


# ---------------------------------------------------------------- 自己検査

def _selfcheck():
    import time
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generators import _nll_gjr_t

    rng = np.random.default_rng(0)
    r = rng.standard_t(6, 1000)
    v = float(r.var())
    p = (0.02, 0.03, 0.10, 0.88, 6.0)

    a = gjr_t_nll(p, r, v)
    b = _nll_gjr_t(p, r, v)
    print(f"JIT 版と generators 版の NLL 一致: |差| = {abs(a - b):.2e} "
          f"({'OK' if abs(a - b) < 1e-8 else 'NG'})")

    t0 = time.time(); gjr_t_nll(p, r, v); t1 = time.time()
    n = 200
    t0 = time.time()
    for _ in range(n):
        gjr_t_nll(p, r, v)
    dt = (time.time() - t0) / n
    print(f"NLL 1回 {dt*1e3:.3f} ms（JIT {'有効' if HAVE_NUMBA else '無効'}）")

    t0 = time.time()
    g = reversal_gap(r)
    print(f"gap 1窓 {time.time()-t0:.2f} s   iid-t での値 {g:+.5f}")

    # 反対称性: gap(rev) = −gap
    g2 = reversal_gap(r[::-1].copy())
    print(f"反対称性: gap(x)={g:+.5f}  gap(rev x)={g2:+.5f}  "
          f"和 {g+g2:+.2e} ({'OK' if abs(g+g2) < 5e-4 else 'NG — 最適化ノイズ超過'})")
    return 0


if __name__ == "__main__":
    import sys as _sys
    if "--calibrate" in _sys.argv:
        raise SystemExit(_calibrate())
    raise SystemExit(_selfcheck())
