"""S&P500 に対する ABM の較正装置。

**合わせ込む量と、判定に使う量を分ける。**

較正するのは3つだけ:
    excess_kurtosis   — 裾の重さ
    acf_abs_1         — ボラティリティ・クラスタリング
    acf_return_1       — リターンの無相関性

これは ABM 論文が「fat tails / volatility clustering / no autocorrelation」として
ほぼ必ず挙げる3点そのものである。残り10個の統計量は較正に一切使わず、
較正後に初めて評価する。合わせ込んだ量で評価しても何も分からないため。

損失は、実データ窓124本におけるその統計量のばらつきで標準化した二乗和:

    loss = Σ_j ((sim_j - target_j) / s_j)^2

探索は Sobol 列による準乱数探索 + 座標方向の局所精錬。全て seed 固定。
"""

import json
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

# リポジトリの場所は自分の位置から引く。以前は書いた環境の絶対パスが
# 直書きされていて、その容れ物の外では import が落ちていた。
SB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SB)
for p in __import__("glob").glob("/workspace/financial-abm-lab/packages/*"):
    sys.path.insert(0, p)

from facts import BATTERY, evaluate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 1000
STRIDE = 250
SEED = 20260802
FITTED = ("excess_kurtosis", "acf_abs_1", "acf_return_1")


# ---------------------------------------------------------------- 実データ

def load_series():
    out = {}
    for fn in sorted(os.listdir(os.path.join(SB, "data"))):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(SB, "data", fn)))["chart"]["result"][0]
        c = np.array([x for x in d["indicators"]["quote"][0]["close"]
                      if x is not None], dtype=float)
        r = np.diff(np.log(c))
        r = r[np.isfinite(r)]
        out[fn[:-5]] = r / r.std()
    return out


def target_and_scale():
    """S&P500 直近1000本を目標に、統計量のばらつきは全指数の窓124本から取る。"""
    ser = load_series()
    target_win = ser["gspc"][-WINDOW:]
    target = evaluate(target_win)

    vals = {k: [] for k in BATTERY}
    for r in ser.values():
        for s in range(0, len(r) - WINDOW + 1, STRIDE):
            v = evaluate(r[s:s + WINDOW])
            for k in BATTERY:
                vals[k].append(v[k])
    scale = {k: float(np.std(vals[k])) for k in BATTERY}
    n_win = len(vals[BATTERY[0] if isinstance(BATTERY, (list, tuple)) else
                     list(BATTERY)[0]])
    return target_win, target, scale, n_win


# ---------------------------------------------------------------- モデル定義
# 各 spec: (パラメータ名 -> (下限, 上限, 整数か), 構築関数)
# 構築関数は seed を受けて長さ WINDOW のリターンを返す。

def _tail(r, n=WINDOW):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return r[-n:]


def make_ci():
    from abm_models.chiarella_iori import ChiarellaIori, CIParams

    box = {"alpha_fund": (0.05, 0.8, 0), "alpha_chart": (0.05, 0.8, 0),
           "fund_speed": (0.005, 0.5, 0), "chart_lag": (2, 60, 1),
           "chart_strength": (0.05, 3.0, 0), "noise_scale": (0.001, 0.15, 0),
           "price_impact": (0.0005, 0.5, 0)}

    def build(p, seed):
        af, ac = p["alpha_fund"], p["alpha_chart"]
        tot = af + ac
        if tot > 0.95:
            af, ac = af * 0.95 / tot, ac * 0.95 / tot
        pr = CIParams(n_steps=int(WINDOW * 1.25), alpha_fund=af, alpha_chart=ac,
                      alpha_noise=max(0.05, 1 - af - ac),
                      fund_speed=p["fund_speed"], chart_lag=int(p["chart_lag"]),
                      chart_strength=p["chart_strength"],
                      noise_scale=p["noise_scale"], price_impact=p["price_impact"])
        return _tail(ChiarellaIori(n_steps=pr.n_steps, params=pr).run(seed=seed)["returns"])
    return box, build


def make_fw():
    from abm_models.franke_westerhoff import FrankeWesterhoff, FWParams

    box = {"phi": (0.01, 1.2, 0), "chi": (0.1, 3.0, 0), "alpha_w": (0.1, 5.0, 0),
           "alpha_o": (0.0, 1.0, 0), "alpha_p": (0.0, 2.0, 0),
           "sigma_f": (0.001, 0.1, 0), "sigma_c": (0.001, 0.15, 0),
           "noise_scale": (0.001, 0.1, 0), "price_impact": (0.05, 2.0, 0)}

    def build(p, seed):
        pr = FWParams(n_steps=int(WINDOW * 1.25), **{k: p[k] for k in box})
        return _tail(FrankeWesterhoff(n_steps=pr.n_steps, params=pr).run(seed=seed)["returns"])
    return box, build


def make_zi():
    from abm_models.zero_intelligence import ZeroIntelligence

    box = {"n_agents": (10, 400, 1), "noise_scale": (0.001, 0.2, 0),
           "price_impact": (0.001, 0.5, 0), "tick_size": (0.001, 0.2, 0)}

    def build(p, seed):
        m = ZeroIntelligence(n_agents=int(p["n_agents"]), n_steps=int(WINDOW * 1.25),
                             noise_scale=p["noise_scale"],
                             price_impact=p["price_impact"], tick_size=p["tick_size"])
        return _tail(m.run(seed=seed)["returns"])
    return box, build


def make_lm():
    from abm_models.lux_marchesi import LuxMarchesi, Params

    box = {"nu1": (0.5, 8.0, 0), "nu2": (0.2, 6.0, 0), "beta": (1.0, 15.0, 0),
           "Tc": (2.0, 30.0, 0), "Tf": (1.0, 20.0, 0), "alpha1": (0.1, 2.0, 0),
           "alpha2": (0.02, 1.0, 0), "alpha3": (0.1, 2.0, 0),
           "sigma_mu": (0.005, 0.3, 0)}

    def build(p, seed):
        pr = Params(**{k: p[k] for k in box})
        m = LuxMarchesi(n_integer_steps=int(WINDOW * 1.25), steps_per_unit=100,
                        n_c_init=50, params=pr)
        return _tail(m.run(seed=seed)["returns"])
    return box, build


def make_cb():
    from abm_models.cont_bouchaud import ContBouchaud

    box = {"N": (500, 8000, 1), "c": (0.05, 0.995, 0), "a": (0.001, 0.2, 0),
           "lam": (0.1, 4.0, 0)}

    def build(p, seed):
        m = ContBouchaud(N=int(p["N"]), c=p["c"], a=p["a"], lam=p["lam"],
                         T=int(WINDOW * 1.25), report_every=10**9)
        return _tail(m.run(seed=seed)["returns"])
    return box, build


MODELS = {
    "chiarella_iori": (make_ci, 1400, 6),
    "franke_westerhoff": (make_fw, 1400, 6),
    "zero_intelligence": (make_zi, 900, 6),
    "lux_marchesi": (make_lm, 140, 3),
    "cont_bouchaud": (make_cb, 140, 3),
}


# ---------------------------------------------------------------- 較正

def stats_of(build, p, seeds):
    """複数 seed の統計量の中央値。失敗した seed は捨てる。"""
    rows = []
    for s in seeds:
        try:
            r = build(p, int(s))
            if r.size < WINDOW * 0.9 or not np.isfinite(r).all() or r.std() == 0:
                continue
            rows.append(evaluate(r / r.std()))
        except Exception:
            continue
    if len(rows) < max(2, len(seeds) // 2):
        return None
    return {k: float(np.median([x[k] for x in rows])) for k in rows[0]}


def loss_of(sim, target, scale):
    if sim is None:
        return np.inf
    return float(sum(((sim[k] - target[k]) / scale[k]) ** 2 for k in FITTED))


def calibrate(name, target, scale, log):
    make, n_draws, n_seeds = MODELS[name]
    box, build = make()
    keys = sorted(box)
    lo = np.array([box[k][0] for k in keys], float)
    hi = np.array([box[k][1] for k in keys], float)
    isint = [box[k][2] for k in keys]
    seeds = list(range(n_seeds))

    sob = qmc.Sobol(d=len(keys), scramble=True, seed=SEED)
    draws = qmc.scale(sob.random(n_draws), lo, hi)

    best, best_loss, n_eval, t0 = None, np.inf, 0, time.time()
    for row in draws:
        p = {k: (int(round(v)) if isint[i] else float(v))
             for i, (k, v) in enumerate(zip(keys, row))}
        l = loss_of(stats_of(build, p, seeds), target, scale)
        n_eval += 1
        if l < best_loss:
            best_loss, best = l, dict(p)
            log(f"  [{name}] eval {n_eval:>5}  loss {l:9.3f}")

    # 座標方向の局所精錬
    if best is not None and np.isfinite(best_loss):
        for _ in range(3):
            improved = False
            for i, k in enumerate(keys):
                span = (hi[i] - lo[i]) * 0.08
                for d in (-span, span):
                    cand = dict(best)
                    v = np.clip(best[k] + d, lo[i], hi[i])
                    cand[k] = int(round(v)) if isint[i] else float(v)
                    l = loss_of(stats_of(build, cand, seeds), target, scale)
                    n_eval += 1
                    if l < best_loss - 1e-9:
                        best_loss, best, improved = l, cand, True
            if not improved:
                break
        log(f"  [{name}] 精錬後 loss {best_loss:.3f}  ({n_eval} evals, "
            f"{time.time()-t0:.0f}s)")
    return best, best_loss, build, n_eval, time.time() - t0


def main():
    t0 = time.time()
    logf = open(os.path.join(HERE, "calib.log"), "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log("実データを読み込み、目標と尺度を作る…")
    win, target, scale, n_win = target_and_scale()
    log(f"S&P500 直近{WINDOW}本を目標／尺度は実データ窓{n_win}本から")
    log("較正する量: " + ", ".join(FITTED))
    for k in FITTED:
        log(f"   目標 {k:<18} {target[k]:+.4f}   （実窓のばらつき {scale[k]:.4f}）")

    out = {"config": {"window": WINDOW, "stride": STRIDE, "seed": SEED,
                      "fitted": list(FITTED), "n_real_windows": n_win},
           "target": target, "scale": scale, "models": {}}
    np.save(os.path.join(HERE, "target_window.npy"), win)

    for name in MODELS:
        log(f"\n=== {name} ===")
        best, bl, build, n_eval, secs = calibrate(name, target, scale, log)
        if best is None:
            log(f"  [{name}] 較正失敗（有効なパラメータが見つからない）")
            out["models"][name] = {"status": "failed"}
            continue
        # 較正後の評価: 100本走らせ、全13統計量の分布を出す
        paths, rows = [], []
        for s in range(1000, 1100):
            try:
                r = build(best, s)
                if r.size < WINDOW * 0.9 or not np.isfinite(r).all() or r.std() == 0:
                    continue
                rn = r / r.std()
                paths.append(rn)
                rows.append(evaluate(rn))
            except Exception:
                continue
        log(f"  [{name}] 評価用パス {len(paths)} 本")
        if paths:
            np.save(os.path.join(HERE, f"paths_{name}.npy"),
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
        out["models"][name] = {"status": "ok", "params": best, "loss": bl,
                               "n_eval": n_eval, "secs": secs,
                               "n_paths": len(paths), "stats": summ}
        json.dump(out, open(os.path.join(HERE, "calibration.json"), "w"),
                  ensure_ascii=False, indent=1)

    log(f"\n完了 {time.time()-t0:.0f}s")
    logf.close()


if __name__ == "__main__":
    main()
