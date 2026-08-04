"""100本の世界線を走らせて、束として何が起きるかを出す。

1本の軌跡は seed 次第でどうにでも見えるので、主張は束で見せる必要がある。
ここでは3つのパラメータ集合を各100 seed 走らせ、作図用のデータを書き出す。

    paper        論文の数値そのまま（α=0.1, σ₁=10, σ₂=1.2, ノイズは字義通り τ 倍）
    paper_sqrt   ノイズだけ √τ に直したもの（α は論文のまま）
    calibrated   S&P500 較正値（価格がファンダに追随する制約付き）

書き出すもの:
    price/fundamental の軌跡（束の広がり）
    最終時点の price/fundamental の分布
    指値の中値からの距離（論文 Figure 7）
    leverage と excess_kurtosis の分布（S&P500 の値と並べる）

    python3 models/ensemble.py        # 約6分（4並列）
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
sys.path.insert(0, SB)

from facts import excess_kurtosis, leverage  # noqa: E402
from models.book_stats import BookProbe  # noqa: E402
from models.chiarella_iori import CIParams  # noqa: E402

N_SEEDS, N_STEPS, WARMUP, N_PROC = 100, 8000, 1000, 4
AGG = 3          # 較正で出た「1営業日 ≈ 数ステップ」に合わせた集約幅
KEEP = 400       # 軌跡は間引いて保存する

CAL = json.load(open(os.path.join(SB, "calibration", "calibration_cip.json")))["params"]

SETS = {
    "paper": CIParams(noise_horizon="linear"),
    "paper_sqrt": CIParams(noise_horizon="sqrt"),
    "calibrated": CIParams(
        alpha=float(CAL["alpha"]), sigma_1=float(CAL["sigma_1"]),
        sigma_2=float(CAL["sigma_2"]), tau=int(round(CAL["tau"])),
        sigma_eps=float(CAL["sigma_eps"]), sigma_fund=float(CAL["sigma_fund"]),
        noise_horizon="sqrt"),
}


def one(arg):
    name, seed = arg
    pr = SETS[name]
    m = BookProbe(n_steps=N_STEPS, warmup=WARMUP, params=pr)
    o = m.run(seed=seed)

    ratio = np.asarray(o["prices"], dtype=float) / np.asarray(o["fundamental"], dtype=float)
    idx = np.linspace(0, ratio.size - 1, KEEP).astype(int)

    cut = pr.tau_cap + WARMUP
    d = np.array([x[2] for x in m.placements if x[0] >= cut], dtype=float)

    r = np.asarray(o["returns"], dtype=float)
    r = r[np.isfinite(r)]
    k = (r.size // AGG) * AGG
    daily = r[:k].reshape(-1, AGG).sum(axis=1)
    daily = daily[np.isfinite(daily)]
    ok = daily.size > 50 and daily.std() > 0
    dn = daily / daily.std() if ok else daily

    return {
        "set": name, "seed": seed,
        "track": ratio[idx].tolist(),
        "final_ratio": float(ratio[-1]),
        "median_ratio": float(np.median(ratio)),
        "dist_median": float(np.median(d)) if d.size else float("nan"),
        "dist_q": (np.percentile(d, [50, 75, 90, 99]).tolist()
                   if d.size else [float("nan")] * 4),
        "leverage": float(leverage(dn)) if ok else float("nan"),
        "excess_kurtosis": float(excess_kurtosis(dn)) if ok else float("nan"),
    }


def main():
    jobs = [(n, s) for n in SETS for s in range(N_SEEDS)]
    with Pool(N_PROC) as pool:
        rows = pool.map(one, jobs)

    # S&P500 の実測値を並べるために読む
    sys.path.insert(0, os.path.join(SB, "calibration"))
    import calibrate as C
    _, target, scale, _ = C.target_and_scale()

    out = {"n_seeds": N_SEEDS, "n_steps": N_STEPS, "agg": AGG,
           "sp500": {"leverage": target["leverage"],
                     "excess_kurtosis": target["excess_kurtosis"],
                     "leverage_scale": scale["leverage"],
                     "excess_kurtosis_scale": scale["excess_kurtosis"]},
           "params": {k: vars(v) for k, v in SETS.items()},
           "runs": rows}
    p = os.path.join(HERE, "ensemble.json")
    json.dump(out, open(p, "w"), ensure_ascii=False)

    print(f"{'集合':<12} {'価格/ファンダ 中央値':>20} {'[10%,90%]':>22} "
          f"{'指値距離 中央値':>16} {'leverage':>12}")
    for name in SETS:
        rs = [r for r in rows if r["set"] == name]
        mr = np.array([r["median_ratio"] for r in rs])
        dm = np.array([r["dist_median"] for r in rs])
        lv = np.array([r["leverage"] for r in rs])
        print(f"{name:<12} {np.median(mr):20.3f} "
              f"{'[%.3f, %.3f]' % tuple(np.percentile(mr, [10, 90])):>22} "
              f"{100*np.nanmedian(dm):15.2f}% {np.nanmedian(lv):12.4f}")
    print(f"{'S&P500':<12} {'—':>20} {'—':>22} {'—':>16} "
          f"{out['sp500']['leverage']:12.4f}")
    print(f"\n書き出し: {p}")


if __name__ == "__main__":
    main()
