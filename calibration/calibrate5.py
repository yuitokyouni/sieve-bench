"""較正 v5 — ドリフト制約つき。

v3/v4 の較正は、3つの目標（尖度・acf_abs_1・acf_return_1）を合わせることには
成功したが、**累積系列が現実に見えなかった**。CI/FW/ZI はほぼ直線のランプで、
ドリフト（mean/σ）が実データの 4〜10 倍あった。

原因は目的関数の指定不足である。3目標のうち尖度と acf_return_1 は
**ドリフトに完全に盲目**で、acf_abs_1 も間接的にしか反応しない。
そして 14 統計量のどれもドリフトを直接測っていなかったため、
ベンチマーク自体がこの欠陥を見られなかった。

v5 では、実データ窓 124 本のドリフト 95% 範囲を**妥当性の門**として使う：

    DRIFT_LO ≤ median_seeds(mean/σ) ≤ DRIFT_HI

損失には入れない。「市場が死んでいる走行を捨てる」のと同じ種類の措置で、
最適化の目標ではなく、もっともらしさの下限である。

**代償:** ドリフトはこれ以降 held-out の判定材料には使えない（合わせ込んだ量に
なるため）。v3/v4（制約なし）と v5（制約あり）を並べて報告することで、
「モデルに有利を与えた上でも leverage が分けられるか」を見る。

    python3 calibrate5.py            # 未処理のモデルを順に、1つ終わるごとに保存
"""

import json
import os
import time

import numpy as np
from scipy.stats import qmc

import calibrate as C
import calibrate3 as V3
import calibrate4_ci as V4

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, SEED, FITTED = C.WINDOW, C.SEED, C.FITTED
OUTFILE = os.path.join(HERE, "calibration_v5.json")

ORDER = ["chiarella_iori", "franke_westerhoff", "zero_intelligence",
         "lux_marchesi", "cont_bouchaud"]
BUDGET = {"chiarella_iori": (1600, 6), "franke_westerhoff": (1400, 6),
          "zero_intelligence": (900, 6), "lux_marchesi": (70, 3),
          "cont_bouchaud": (210, 3)}


def drift(x):
    s = float(np.std(x))
    return float(np.mean(x) / s) if s > 0 else np.nan


def real_drift_range():
    ser = C.load_series()
    d = [drift(r[s:s + WINDOW])
         for r in ser.values() for s in range(0, len(r) - WINDOW + 1, C.STRIDE)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)


def make_boxes():
    """モデル名 -> (box, build)。CI だけ v4 の単体パラメータ化を使う。"""
    out = {}
    for name in ORDER:
        if name == "chiarella_iori":
            out[name] = (dict(V4.BOX), V4.build, dict(V4.DEFAULT))
            continue
        box, build = C.MODELS[name][0]()
        box = dict(box)
        for k, v in V3.OVERRIDE.get(name, {}).items():
            if k in box:
                box[k] = v
        d = {k: V3.DEFAULTS[name][k] for k in box if k in V3.DEFAULTS[name]}
        out[name] = (box, build, d)
    return out


def stats_of(build, p, seeds, rej, lo, hi):
    rows, raw, dr = [], [], []
    for s in seeds:
        try:
            r = build(p, int(s))
        except Exception:
            rej["例外"] = rej.get("例外", 0) + 1
            continue
        ok, why = V3.valid(r)
        if not ok:
            rej[why] = rej.get(why, 0) + 1
            continue
        dr.append(drift(r))
        rn = r / r.std()
        raw.append(rn)
        rows.append(C.evaluate(rn))
    if len(rows) < max(2, len(seeds) // 2):
        return None
    md = float(np.median(dr))
    if not (lo <= md <= hi):
        rej["ドリフト外"] = rej.get("ドリフト外", 0) + 1
        return None
    if V3.deterministic(raw):
        rej["決定論的"] = rej.get("決定論的", 0) + 1
        return None
    return {k: float(np.median([x[k] for x in rows])) for k in rows[0]}


def run_model(name, boxes, target, scale, lo, hi, log):
    box, build, dflt = boxes[name]
    n_draws, n_seeds = BUDGET[name]
    keys = sorted(box)
    blo = np.array([box[k][0] for k in keys], float)
    bhi = np.array([box[k][1] for k in keys], float)
    isint = [box[k][2] for k in keys]
    seeds = list(range(n_seeds))
    rej = {}

    def ev(p):
        return C.loss_of(stats_of(build, p, seeds, rej, lo, hi), target, scale)

    d = {k: dflt[k] for k in keys if k in dflt}
    for i, k in enumerate(keys):
        d.setdefault(k, float((blo[i] + bhi[i]) / 2))
    dl = ev(d)
    log(f"  [{name}] 既定値 loss = {dl:.3f}"
        + ("  （ドリフト制約で棄却）" if not np.isfinite(dl) else ""))
    best, bl = (dict(d), dl) if np.isfinite(dl) else (None, np.inf)

    # 探索は高価なので、終わったら即保存して再実行時に飛ばせるようにする
    ckpt = os.path.join(HERE, f"v5_search_{name}.json")
    if os.path.exists(ckpt):
        c = json.load(open(ckpt))
        best, bl = c["best"], c["loss"]
        log(f"  [{name}] 探索結果を再利用 loss = {bl:.3f}")
        draws = []
    else:
        draws = qmc.scale(
            qmc.Sobol(d=len(keys), scramble=True, seed=SEED).random(n_draws), blo, bhi)
    t0 = time.time()
    for row in draws:
        p = {k: (int(round(v)) if isint[i] else float(v))
             for i, (k, v) in enumerate(zip(keys, row))}
        l = ev(p)
        if l < bl:
            bl, best = l, dict(p)
    if len(draws):
        log(f"  [{name}] 探索後 loss = {bl:.3f}  ({time.time()-t0:.0f}s)")
        if best is not None:
            json.dump({"best": best, "loss": bl}, open(ckpt, "w"))
    if best is None:
        log(f"  [{name}] 有効な解なし  棄却 {rej}")
        return {"status": "failed", "rejections": rej}

    for _ in range(1):
        imp = False
        for i, k in enumerate(keys):
            span = (bhi[i] - blo[i]) * 0.07
            for dd in (-span, span):
                cand = dict(best)
                v = float(np.clip(best[k] + dd, blo[i], bhi[i]))
                cand[k] = int(round(v)) if isint[i] else v
                l = ev(cand)
                if l < bl - 1e-9:
                    bl, best, imp = l, cand, True
        if not imp:
            break
    log(f"  [{name}] 最終 loss = {bl:.3f}  棄却 {rej}")

    at_bound = [k for i, k in enumerate(keys)
                if not k.startswith("w_")
                and ((float(best[k]) - blo[i]) / (bhi[i] - blo[i]) <= 0.02
                     or (float(best[k]) - blo[i]) / (bhi[i] - blo[i]) >= 0.98)]

    paths, rows, dr = [], [], []
    for s in range(1000, 1100):
        try:
            r = build(best, s)
        except Exception:
            continue
        if not V3.valid(r)[0]:
            continue
        dr.append(drift(r))
        rn = r / r.std()
        paths.append(rn)
        rows.append(C.evaluate(rn))
    log(f"  [{name}] 有効パス {len(paths)}/100  drift中央値 {np.median(dr):+.4f}"
        + (f"  ◀境界 {at_bound}" if at_bound else "  境界なし"))
    if paths:
        np.save(os.path.join(HERE, f"v5_paths_{name}.npy"),
                np.array(paths, dtype=np.float32))
    summ = {k: {"median": float(np.median([x[k] for x in rows])),
                "target": float(target[k]),
                "z": float((np.median([x[k] for x in rows]) - target[k]) / scale[k]),
                "fitted": k in FITTED} for k in rows[0]}
    return {"status": "ok", "params": best, "loss": bl, "default_loss": dl,
            "at_bound": at_bound, "rejections": rej, "n_paths": len(paths),
            "drift_median": float(np.median(dr)), "stats": summ}


def main():
    lf = open(os.path.join(HERE, "calib5.log"), "a")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    lo, hi, n_win = real_drift_range()
    _, target, scale, _ = C.target_and_scale()
    if os.path.exists(OUTFILE):
        out = json.load(open(OUTFILE))
    else:
        out = {"config": {"window": WINDOW, "seed": SEED, "fitted": list(FITTED),
                          "drift_lo": lo, "drift_hi": hi, "n_real_windows": n_win},
               "target": target, "scale": scale, "models": {}}
    log(f"ドリフト制約 [{lo:+.4f}, {hi:+.4f}]（実窓{n_win}本の95%範囲）")

    boxes = make_boxes()
    for name in ORDER:
        if out["models"].get(name, {}).get("status") == "ok":
            log(f"=== {name} … 保存済み、飛ばす")
            continue
        log(f"\n=== {name} ===")
        out["models"][name] = run_model(name, boxes, target, scale, lo, hi, log)
        json.dump(out, open(OUTFILE, "w"), ensure_ascii=False, indent=1)
        log(f"  [{name}] 保存した")
    log("\n全モデル完了")
    lf.close()


if __name__ == "__main__":
    main()
