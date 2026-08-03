"""較正 v3。

v2 で分かった欠陥:
  1. 論文の既定パラメータより悪い解を採用していた（LM: 尖度 9.66 → 1.86）。
     探索の初期点に既定値が入っていなかったため。
  2. 妥当性フィルタが「決定論的な直線」を弾けなかった（CI/FW が一本帯に）。
  3. CI/FW/CB でまだ別のパラメータが範囲の端にいた。
  4. LM は9次元を176回しか評価しておらず、探索が薄すぎた。

v3 での対処:
  - **既定パラメータを初期点として必ず評価する。**これで既定値を下回る解は
    採用されない。
  - 縮退フィルタに「パス間がほぼ同一（＝確率的な広がりが無い）」を追加。
  - 端にいたパラメータの範囲をさらに広げる。
  - 遅いモデルの評価回数を増やす。
  - 全評価点の (尖度, acf_abs_1) を記録し、**達成可能な境界**を出す。
    単一の最適点ではなく、そのモデルに何ができて何ができないかを見せる。
"""

import json
import os
import time

import numpy as np
from scipy.stats import qmc

import calibrate as C
import calibrate2 as V2

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, SEED, FITTED = C.WINDOW, C.SEED, C.FITTED

# v2 で端にいたものをさらに広げる
OVERRIDE = dict(V2.OVERRIDE)
OVERRIDE["chiarella_iori"] = {**OVERRIDE["chiarella_iori"],
                              "alpha_fund": (0.01, 0.95, 0), "chart_lag": (2, 250, 1)}
OVERRIDE["franke_westerhoff"] = {**OVERRIDE["franke_westerhoff"],
                                 "noise_scale": (1e-6, 0.5, 0), "sigma_f": (1e-6, 0.3, 0),
                                 "sigma_c": (1e-6, 0.5, 0)}
OVERRIDE["cont_bouchaud"] = {**OVERRIDE["cont_bouchaud"], "c": (0.05, 0.99999, 0)}

# 論文／実装の既定パラメータ（初期点として必ず評価する）
DEFAULTS = {
    "chiarella_iori": {"alpha_fund": 0.4, "alpha_chart": 0.3, "fund_speed": 0.05,
                       "chart_lag": 10, "chart_strength": 0.8, "noise_scale": 0.01,
                       "price_impact": 0.005},
    "franke_westerhoff": {"phi": 0.12, "chi": 1.5, "alpha_w": 1.8, "alpha_o": 0.05,
                          "alpha_p": 0.2, "sigma_f": 0.01, "sigma_c": 0.02,
                          "noise_scale": 0.01, "price_impact": 0.5},
    "zero_intelligence": {"n_agents": 100, "noise_scale": 0.01, "price_impact": 0.01,
                          "tick_size": 0.01},
    "lux_marchesi": {"nu1": 3.0, "nu2": 2.0, "beta": 6.0, "Tc": 10.0, "Tf": 5.0,
                     "alpha1": 0.6, "alpha2": 0.2, "alpha3": 0.5, "sigma_mu": 0.05},
    "cont_bouchaud": {"N": 10000, "c": 0.9, "a": 0.01, "lam": 1.0},
}

BUDGET = {"chiarella_iori": (1400, 6), "franke_westerhoff": (1400, 6),
          "zero_intelligence": (900, 6), "lux_marchesi": (420, 3),
          "cont_bouchaud": (320, 3)}

MIN_PATH_SPREAD = 0.02   # パス間の終端ばらつき / sqrt(n) がこれ未満なら決定論的
MAX_ZERO_RUN = 50        # ゼロ収益がこれ以上連続したら「市場が止まった」


def longest_zero_run(r):
    z = np.abs(r) < 1e-12
    m = c = 0
    for v in z:
        c = c + 1 if v else 0
        if c > m:
            m = c
    return m


def valid(r):
    """縮退した走行を弾く。

    v2 は「ゼロ収益率 > 3%」で弾いていたが、これは誤りだった。
    Lux-Marchesi は価格が ±0.01 の離散ジャンプなので、動かない日の
    リターンが厳密に 0 になるのは**モデルの仕様**である（既定値で 8.9%、
    ただし最長連続は 3）。市場が止まったかどうかは総量ではなく
    **連続長**で見なければならない。
    """
    if r.size < WINDOW * 0.9:
        return False, "短い"
    if not np.isfinite(r).all():
        return False, "非有限"
    s = float(np.std(r))
    if s == 0 or not np.isfinite(s):
        return False, "分散ゼロ"
    if np.std(r[-50:]) < 1e-9:
        return False, "末尾停止"
    if longest_zero_run(r) > MAX_ZERO_RUN:
        return False, "長時間停止"
    return True, ""


def deterministic(paths):
    """100本がほぼ同一の軌跡なら True（確率的な広がりが無い）。"""
    if len(paths) < 5:
        return False
    cum = np.cumsum(np.array(paths), axis=1)
    return float(cum[:, -1].std()) / np.sqrt(cum.shape[1]) < MIN_PATH_SPREAD


def stats_of(build, p, seeds, rej):
    rows, raw = [], []
    for s in seeds:
        try:
            r = build(p, int(s))
        except Exception:
            rej["例外"] = rej.get("例外", 0) + 1
            continue
        ok, why = valid(r)
        if not ok:
            rej[why] = rej.get(why, 0) + 1
            continue
        rn = r / r.std()
        raw.append(rn)
        rows.append(C.evaluate(rn))
    if len(rows) < max(2, len(seeds) // 2):
        return None
    if deterministic(raw):
        rej["決定論的"] = rej.get("決定論的", 0) + 1
        return None
    return {k: float(np.median([x[k] for x in rows])) for k in rows[0]}


def run_model(name, target, scale, log):
    make, _, _ = C.MODELS[name]
    n_draws, n_seeds = BUDGET[name]
    box, build = make()
    box = dict(box)
    for k, v in OVERRIDE.get(name, {}).items():
        if k in box:
            box[k] = v
    keys = sorted(box)
    lo = np.array([box[k][0] for k in keys], float)
    hi = np.array([box[k][1] for k in keys], float)
    isint = [box[k][2] for k in keys]
    seeds = list(range(n_seeds))
    rej, frontier = {}, []

    def ev(p):
        s = stats_of(build, p, seeds, rej)
        l = C.loss_of(s, target, scale)
        if s is not None:
            frontier.append({"loss": l, "excess_kurtosis": s["excess_kurtosis"],
                             "acf_abs_1": s["acf_abs_1"]})
        return l

    # 1) 既定パラメータを必ず評価する
    d = {k: DEFAULTS[name][k] for k in keys if k in DEFAULTS[name]}
    for k in keys:
        d.setdefault(k, float((lo[keys.index(k)] + hi[keys.index(k)]) / 2))
    dl = ev(d)
    log(f"  [{name}] 既定パラメータ loss = {dl:.3f}")
    best, bl = (dict(d), dl) if np.isfinite(dl) else (None, np.inf)

    # 2) 準乱数探索
    draws = qmc.scale(qmc.Sobol(d=len(keys), scramble=True, seed=SEED).random(n_draws),
                      lo, hi)
    t0 = time.time()
    for row in draws:
        p = {k: (int(round(v)) if isint[i] else float(v))
             for i, (k, v) in enumerate(zip(keys, row))}
        l = ev(p)
        if l < bl:
            bl, best = l, dict(p)
    log(f"  [{name}] 探索後 loss = {bl:.3f}"
        + ("  ← 既定値を更新できず" if best is not None and bl >= dl - 1e-12 else ""))

    # 3) 局所精錬
    if best is not None:
        for _ in range(3):
            imp = False
            for i, k in enumerate(keys):
                span = (hi[i] - lo[i]) * 0.08
                for dd in (-span, span):
                    cand = dict(best)
                    v = float(np.clip(best[k] + dd, lo[i], hi[i]))
                    cand[k] = int(round(v)) if isint[i] else v
                    l = ev(cand)
                    if l < bl - 1e-9:
                        bl, best, imp = l, cand, True
            if not imp:
                break
    log(f"  [{name}] 最終 loss = {bl:.3f}  ({time.time()-t0:.0f}s)  棄却 {rej}")
    if best is None:
        return {"status": "failed", "rejections": rej}

    at_bound = [k for i, k in enumerate(keys)
                if (float(best[k]) - lo[i]) / (hi[i] - lo[i]) <= 0.02
                or (float(best[k]) - lo[i]) / (hi[i] - lo[i]) >= 0.98]

    paths, rows = [], []
    for s in range(1000, 1100):
        try:
            r = build(best, s)
        except Exception:
            continue
        if not valid(r)[0]:
            continue
        rn = r / r.std()
        paths.append(rn)
        rows.append(C.evaluate(rn))
    log(f"  [{name}] 有効パス {len(paths)}/100"
        + (f"  ◀境界 {at_bound}" if at_bound else "  境界なし"))
    if paths:
        np.save(os.path.join(HERE, f"v3_paths_{name}.npy"),
                np.array(paths, dtype=np.float32))
    summ = {k: {"median": float(np.median([x[k] for x in rows])),
                "target": float(target[k]),
                "z": float((np.median([x[k] for x in rows]) - target[k]) / scale[k]),
                "fitted": k in FITTED} for k in rows[0]}
    return {"status": "ok", "params": best, "loss": bl, "default_loss": dl,
            "beat_default": bool(bl < dl - 1e-12), "at_bound": at_bound,
            "rejections": rej, "n_paths": len(paths), "stats": summ,
            "frontier": frontier}


def main():
    t0 = time.time()
    lf = open(os.path.join(HERE, "calib3.log"), "w")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    win, target, scale, n_win = C.target_and_scale()
    log(f"S&P500 直近{WINDOW}本／実窓{n_win}本で標準化")
    out = {"config": {"window": WINDOW, "seed": SEED, "fitted": list(FITTED),
                      "min_path_spread": MIN_PATH_SPREAD},
           "target": target, "scale": scale, "models": {}}
    for name in C.MODELS:
        log(f"\n=== {name} ===")
        out["models"][name] = run_model(name, target, scale, log)
        json.dump(out, open(os.path.join(HERE, "calibration_v3.json"), "w"),
                  ensure_ascii=False, indent=1)
    log(f"\n完了 {time.time()-t0:.0f}s")
    lf.close()


if __name__ == "__main__":
    main()
