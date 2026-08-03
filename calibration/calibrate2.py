"""較正 v2 — 縮退の自動排除と、探索範囲の拡張。

v1 の欠陥:
  1. Zero Intelligence の最良解はリターンの36%が厳密にゼロ、走行の49%で
     市場が停止していた。当てはまりの良さ（loss 0.156）は動きを止めて
     得たものだった。**目視で気づいた。装置が気づくべきだった。**
  2. Chiarella-Iori は3個、Franke-Westerhoff は1個のパラメータが
     探索範囲の端に張り付いていた。範囲が狭すぎた。

v2 での対処:
  - 妥当性フィルタ: ゼロ収益率 > 3%、末尾50本が定数、分散ゼロ、
    非有限値を含む走行は「無効」として捨てる。有効 seed が半数を
    切ったパラメータは損失 = inf（＝採用しない）。
  - 端に張り付いた／近かったパラメータの範囲を広げる。
"""

import json
import os
import time

import numpy as np
from scipy.stats import qmc

import calibrate as C

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = C.WINDOW
SEED = C.SEED
FITTED = C.FITTED

# v1 で端に張り付いた／近かったものを広げる。確率など上限に意味がある量は上限を保つ。
OVERRIDE = {
    "chiarella_iori": {"fund_speed": (0.005, 4.0, 0), "noise_scale": (1e-4, 1.5, 0),
                       "price_impact": (1e-6, 3.0, 0), "chart_strength": (0.02, 8.0, 0)},
    "franke_westerhoff": {"alpha_p": (0.0, 10.0, 0), "alpha_w": (0.05, 12.0, 0),
                          "chi": (0.05, 8.0, 0), "phi": (0.005, 4.0, 0),
                          "price_impact": (0.01, 8.0, 0)},
    "zero_intelligence": {"price_impact": (1e-4, 2.0, 0), "tick_size": (1e-5, 0.2, 0),
                          "noise_scale": (1e-4, 0.6, 0), "n_agents": (5, 600, 1)},
    "cont_bouchaud": {"c": (0.05, 0.9995, 0), "lam": (0.02, 12.0, 0),
                      "a": (1e-4, 0.6, 0)},
    "lux_marchesi": {"sigma_mu": (1e-3, 0.8, 0), "beta": (0.2, 30.0, 0)},
}

MAX_ZERO_FRAC = 0.03   # リターンが厳密にゼロである割合の上限
FLAT_TAIL = 50         # 末尾この本数が定数なら停止とみなす


def valid(r):
    """縮退した走行を弾く。返り値 (有効か, 理由)。"""
    if r.size < WINDOW * 0.9:
        return False, "短い"
    if not np.isfinite(r).all():
        return False, "非有限"
    s = float(np.std(r))
    if s == 0 or not np.isfinite(s):
        return False, "分散ゼロ"
    if np.mean(np.abs(r) < 1e-12) > MAX_ZERO_FRAC:
        return False, "ゼロ収益過多"
    if np.std(r[-FLAT_TAIL:]) < 1e-9:
        return False, "末尾停止"
    return True, ""


def stats_of(build, p, seeds, rej=None):
    rows = []
    for s in seeds:
        try:
            r = build(p, int(s))
        except Exception:
            if rej is not None:
                rej["例外"] = rej.get("例外", 0) + 1
            continue
        ok, why = valid(r)
        if not ok:
            if rej is not None:
                rej[why] = rej.get(why, 0) + 1
            continue
        rows.append(C.evaluate(r / r.std()))
    if len(rows) < max(2, len(seeds) // 2):
        return None
    return {k: float(np.median([x[k] for x in rows])) for k in rows[0]}


def run_model(name, target, scale, log):
    make, n_draws, n_seeds = C.MODELS[name]
    box, build = make()
    box = dict(box)
    for k, v in OVERRIDE.get(name, {}).items():
        if k in box:
            log(f"  範囲拡張 {k}: {box[k][:2]} → {v[:2]}")
            box[k] = v
    keys = sorted(box)
    lo = np.array([box[k][0] for k in keys], float)
    hi = np.array([box[k][1] for k in keys], float)
    isint = [box[k][2] for k in keys]
    seeds = list(range(n_seeds))
    rej = {}

    draws = qmc.scale(qmc.Sobol(d=len(keys), scramble=True, seed=SEED).random(n_draws),
                      lo, hi)
    best, bl, n_eval, t0 = None, np.inf, 0, time.time()
    for row in draws:
        p = {k: (int(round(v)) if isint[i] else float(v))
             for i, (k, v) in enumerate(zip(keys, row))}
        l = C.loss_of(stats_of(build, p, seeds, rej), target, scale)
        n_eval += 1
        if l < bl:
            bl, best = l, dict(p)
            log(f"  [{name}] eval {n_eval:>5}  loss {l:9.3f}")
    if best is not None:
        for _ in range(3):
            improved = False
            for i, k in enumerate(keys):
                span = (hi[i] - lo[i]) * 0.08
                for d in (-span, span):
                    cand = dict(best)
                    v = float(np.clip(best[k] + d, lo[i], hi[i]))
                    cand[k] = int(round(v)) if isint[i] else v
                    l = C.loss_of(stats_of(build, cand, seeds, rej), target, scale)
                    n_eval += 1
                    if l < bl - 1e-9:
                        bl, best, improved = l, cand, True
            if not improved:
                break
    log(f"  [{name}] loss {bl:.3f}  ({n_eval} evals, {time.time()-t0:.0f}s)  "
        f"棄却 {rej}")

    if best is None or not np.isfinite(bl):
        return {"status": "failed", "rejections": rej}

    at_bound = []
    for i, k in enumerate(keys):
        f = (float(best[k]) - lo[i]) / (hi[i] - lo[i])
        if f <= 0.02 or f >= 0.98:
            at_bound.append(k)

    paths, rows = [], []
    for s in range(1000, 1100):
        try:
            r = build(best, s)
        except Exception:
            continue
        ok, _ = valid(r)
        if not ok:
            continue
        rn = r / r.std()
        paths.append(rn)
        rows.append(C.evaluate(rn))
    log(f"  [{name}] 有効パス {len(paths)}/100 本"
        + (f"  ◀境界 {at_bound}" if at_bound else ""))
    if paths:
        np.save(os.path.join(HERE, f"v2_paths_{name}.npy"),
                np.array(paths, dtype=np.float32))
    summ = {}
    for k in rows[0]:
        v = np.array([x[k] for x in rows])
        summ[k] = {"median": float(np.median(v)),
                   "lo": float(np.percentile(v, 2.5)),
                   "hi": float(np.percentile(v, 97.5)),
                   "target": float(target[k]),
                   "z": float((np.median(v) - target[k]) / scale[k]),
                   "fitted": k in FITTED}
    return {"status": "ok", "params": best, "loss": bl, "n_eval": n_eval,
            "n_paths": len(paths), "at_bound": at_bound, "rejections": rej,
            "stats": summ}


def main():
    t0 = time.time()
    lf = open(os.path.join(HERE, "calib2.log"), "w")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    win, target, scale, n_win = C.target_and_scale()
    log(f"S&P500 直近{WINDOW}本を目標／尺度は実窓{n_win}本")
    log(f"妥当性: ゼロ収益率≤{MAX_ZERO_FRAC:.0%}、末尾{FLAT_TAIL}本が定数でない、有限、分散>0")
    out = {"config": {"window": WINDOW, "seed": SEED, "fitted": list(FITTED),
                      "max_zero_frac": MAX_ZERO_FRAC, "n_real_windows": n_win},
           "target": target, "scale": scale, "models": {}}
    for name in C.MODELS:
        log(f"\n=== {name} ===")
        out["models"][name] = run_model(name, target, scale, log)
        json.dump(out, open(os.path.join(HERE, "calibration_v2.json"), "w"),
                  ensure_ascii=False, indent=1)
    log(f"\n完了 {time.time()-t0:.0f}s")
    lf.close()


if __name__ == "__main__":
    main()
