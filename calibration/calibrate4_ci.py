"""Chiarella-Iori だけ v4 — 境界3個を解消する。

v3 で境界に張り付いた3個は、性質が異なる:

  price_impact = 3.0 (上限)   … 範囲が狭いだけ。広げる。
  chart_lag    = 250 (上限)   … 同上。広げる。
  alpha_fund   = 0.95 (上限)  … **範囲の問題ではない。**

戦略の構成比 (fundamentalist / chartist / noise) は合計 1 の単体上の点であって、
独立な箱ではない。v3 は各々を箱で振ってから build() の中で正規化していたため、
alpha_fund が「上限」に見えていた。実効値は 0.707 で、0.95 という数字自体に
意味がない。ここは**単体から直接サンプルする**形に書き換える。

noise_scale も 97% と端に近かったので併せて広げる。
"""

import json
import os
import time

import numpy as np
from scipy.stats import qmc

import calibrate as C
import calibrate3 as V3

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, SEED, FITTED = C.WINDOW, C.SEED, C.FITTED
N_DRAWS, N_SEEDS = 4000, 6

# 単体上の重み w_* は箱ではなく比率。正規化するので端が存在しない。
BOX = {
    "w_fund":         (0.02, 1.0, 0),
    "w_chart":        (0.02, 1.0, 0),
    "w_noise":        (0.02, 1.0, 0),
    "fund_speed":     (0.005, 4.0, 0),
    "chart_lag":      (2, 500, 1),
    "chart_strength": (0.02, 8.0, 0),
    "noise_scale":    (1e-4, 10.0, 0),
    "price_impact":   (1e-6, 50.0, 0),
}
KEYS = sorted(BOX)

# 論文/実装の既定値を、同じ表現に直したもの (0.4 / 0.3 / 0.3)
DEFAULT = {"w_fund": 0.4, "w_chart": 0.3, "w_noise": 0.3, "fund_speed": 0.05,
           "chart_lag": 10, "chart_strength": 0.8, "noise_scale": 0.01,
           "price_impact": 0.005}


def build(p, seed):
    from abm_models.chiarella_iori import ChiarellaIori, CIParams
    s = p["w_fund"] + p["w_chart"] + p["w_noise"]
    af, ac, an = p["w_fund"] / s, p["w_chart"] / s, p["w_noise"] / s
    pr = CIParams(n_steps=int(WINDOW * 1.25), alpha_fund=af, alpha_chart=ac,
                  alpha_noise=an, fund_speed=p["fund_speed"],
                  chart_lag=int(p["chart_lag"]), chart_strength=p["chart_strength"],
                  noise_scale=p["noise_scale"], price_impact=p["price_impact"])
    r = np.asarray(ChiarellaIori(n_steps=pr.n_steps, params=pr).run(seed=seed)["returns"],
                   dtype=float)
    r = r[np.isfinite(r)]
    return r[-WINDOW:]


def main():
    t0 = time.time()
    lf = open(os.path.join(HERE, "calib4.log"), "w")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    _, target, scale, _ = C.target_and_scale()
    seeds = list(range(N_SEEDS))
    rej = {}

    def ev(p):
        return C.loss_of(V3.stats_of(build, p, seeds, rej), target, scale)

    dl = ev(DEFAULT)
    log(f"既定パラメータ loss = {dl:.3f}")
    best, bl = (dict(DEFAULT), dl) if np.isfinite(dl) else (None, np.inf)

    lo = np.array([BOX[k][0] for k in KEYS], float)
    hi = np.array([BOX[k][1] for k in KEYS], float)
    isint = [BOX[k][2] for k in KEYS]
    draws = qmc.scale(qmc.Sobol(d=len(KEYS), scramble=True, seed=SEED).random(N_DRAWS),
                      lo, hi)
    for row in draws:
        p = {k: (int(round(v)) if isint[i] else float(v))
             for i, (k, v) in enumerate(zip(KEYS, row))}
        l = ev(p)
        if l < bl:
            bl, best = l, dict(p)
    log(f"探索後 loss = {bl:.3f}  ({time.time()-t0:.0f}s)")

    for _ in range(4):
        imp = False
        for i, k in enumerate(KEYS):
            span = (hi[i] - lo[i]) * 0.06
            for d in (-span, span):
                cand = dict(best)
                v = float(np.clip(best[k] + d, lo[i], hi[i]))
                cand[k] = int(round(v)) if isint[i] else v
                l = ev(cand)
                if l < bl - 1e-9:
                    bl, best, imp = l, cand, True
        if not imp:
            break
    log(f"精錬後 loss = {bl:.3f}  棄却 {rej}")

    # 境界判定：単体の重みは比率なので、端にいても意味を持たない
    at_bound = []
    for i, k in enumerate(KEYS):
        if k.startswith("w_"):
            continue
        f = (float(best[k]) - lo[i]) / (hi[i] - lo[i])
        if f <= 0.02 or f >= 0.98:
            at_bound.append(k)
    s = best["w_fund"] + best["w_chart"] + best["w_noise"]
    log(f"構成比  fund {best['w_fund']/s:.3f} / chart {best['w_chart']/s:.3f} "
        f"/ noise {best['w_noise']/s:.3f}")
    log("境界: " + (str(at_bound) if at_bound else "なし"))

    paths, rows = [], []
    for sd in range(1000, 1100):
        try:
            r = build(best, sd)
        except Exception:
            continue
        if not V3.valid(r)[0]:
            continue
        rn = r / r.std()
        paths.append(rn)
        rows.append(C.evaluate(rn))
    log(f"有効パス {len(paths)}/100")
    if paths:
        np.save(os.path.join(HERE, "v4_paths_chiarella_iori.npy"),
                np.array(paths, dtype=np.float32))
    summ = {k: {"median": float(np.median([x[k] for x in rows])),
                "target": float(target[k]),
                "z": float((np.median([x[k] for x in rows]) - target[k]) / scale[k]),
                "fitted": k in FITTED} for k in rows[0]}
    json.dump({"params": best, "loss": bl, "default_loss": dl, "at_bound": at_bound,
               "mix": {"fund": best["w_fund"] / s, "chart": best["w_chart"] / s,
                       "noise": best["w_noise"] / s},
               "n_paths": len(paths), "rejections": rej, "stats": summ},
              open(os.path.join(HERE, "calibration_v4_ci.json"), "w"),
              ensure_ascii=False, indent=1)
    log(f"完了 {time.time()-t0:.0f}s")
    lf.close()


if __name__ == "__main__":
    main()
