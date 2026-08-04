"""論文通りの Chiarella-Iori-Perelló を S&P500 に較正する。

**なぜ論文の数値をそのまま使えないか。**注文の抽選区間の対数幅は恒等式

    ln(p_M / p_m) = αᵢ · Vᵢ · (p_m·Sᵢ + Cᵢ)

なので、板の締まり具合は **A = α · N_S · pᶠ** という群だけで決まる。論文値の
α=0.1, N_S=50, pᶠ=300 では A=1500 となり、V→板幅→スプレッド→V の帰還利得が
2.05 と 1 を超える。**論文の Figure 7（ファンダ支配下で指値は中値のごく近く）は
論文自身のパラメータでは出ない。**A を 450 付近まで下げると利得が 0.96 になり、
価格がファンダに密着し指値も中値の 0.15% に収まる（`models/book_stats.py`）。

**ノイズの積み方も1つ決める必要がある。**式(1) の ε は1ステップの率だが
式(3) が τᵢ 倍するので、字義通りだと σ_ε=1e-4 が τᵢ/Σg≈115 倍されて 1.15% に
なる。ここでは ε を1ステップのショックと読み、horizon では √τᵢ で積む
（`noise_horizon="sqrt"`）。

較正する量（いずれもスケール不変な統計量で識別できるもの）:

    alpha       A = α·N_S·pᶠ を通して板の幅を決める
    sigma_1     ファンダメンタリスト重みの平均
    sigma_2     チャーティスト重みの平均
    tau         参照投資期間
    sigma_eps   ノイズ成分
    sigma_fund  ファンダメンタルのボラティリティ
    agg         **1営業日に相当するモデルステップ数。**論文の1ステップは
                「1人が注文を出す」であって1日ではないので、日次の S&P500 と
                比べるには集約が要る。これも較正対象にする。

目標は S&P500 の直近1000営業日。統計量のばらつきは6指数124窓から取る
（`calibrate.py` と同じ）。`FITTED` の3つだけを合わせ、残りは検証に使う。

**結果（PRICE_GATE = 0.80〜1.25）:**

    論文パラメータ            loss = inf（価格がファンダから外れて棄却される）
    較正後                    loss = 0.223
    A = α·N_S·pᶠ = 50        α は箱の下限に張り付く（＝板は締まるほど良い）
    σ₁=15.8, σ₂=6.77, τ=274, σ_ε=6.1e-4, σ_fund=8.5e-4, agg=2.5

この点は**板と価格の両方で論文通り**になる（指値の距離が中央値 0.68%、99.1% が
Figure 7 の横軸 0〜0.07 に収まり、価格はファンダの 0.941）。

**ゲートを外すと loss は 0.063 まで下がるが、板が壊れる。**σ₂ が 9.86 まで上がり、
価格がファンダの 0.313、指値の距離が中央値 41% になる。**リターンの定型的事実に
合わせることと板を正気に保つことは逆方向に引く。**しかもリターンの統計量だけを
見ていてもこの破綻は見えない。

**どう較正しても直らないもの:**

    leverage      z = 5.18   符号ごと逆（+0.020 対 −0.109）
    multiscaling  z = 3.35
    drift         z = −2.20  論文がドリフト0のファンダを規定しているので構造上一致しない

leverage は separation.py で識別力が最大（garch_norm に 0.966）かつ ABM 論文が
ほとんど報告しない統計量である。**よく報告される3つを 0.4σ 以内に合わせても、
報告されない1つは符号すら合わない。**

    python3 calibration/calibrate_cip.py     # 約14分（4並列）
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.dirname(HERE)
sys.path.insert(0, SB)
sys.path.insert(0, HERE)

import calibrate as C  # noqa: E402
from facts import evaluate  # noqa: E402
from models.chiarella_iori import ChiarellaIoriPerello, CIParams  # noqa: E402

WINDOW = C.WINDOW
SEED = C.SEED
FITTED = C.FITTED
N_DRAWS, N_SEEDS, N_PROC = 512, 3, 4

# 価格がファンダのこの範囲に収まる走行だけを採用する（論文の主張を制約に課す）
PRICE_GATE = (0.80, 1.25)

# (下限, 上限, 対数で振るか)
# σ₂ の上限は最初 3.0（論文 Figure 4 右の描画範囲）にしていたが、最適点が
# そこに張り付いたので広げた。論文は σ₁,σ₂ を 0〜30 で振ったと書いている。
BOX = {
    "alpha":      (0.002, 0.20, 1),
    "sigma_1":    (0.0, 40.0, 0),
    "sigma_2":    (0.0, 12.0, 0),
    "tau":        (20, 500, 0),
    "sigma_eps":  (1e-5, 1e-3, 1),
    "sigma_fund": (1e-4, 1e-2, 1),
    "agg":        (2, 25, 0),
}
KEYS = sorted(BOX)

# 論文の値（比較の基準）
PAPER = {"alpha": 0.1, "sigma_1": 10.0, "sigma_2": 1.20, "tau": 200,
         "sigma_eps": 1e-4, "sigma_fund": 1e-3, "agg": 10}


def build(p, seed):
    """パラメータと seed から、長さ WINDOW の「日次」リターンを返す。"""
    agg = max(2, int(round(p["agg"])))
    pr = CIParams(
        alpha=float(p["alpha"]), sigma_1=float(p["sigma_1"]),
        sigma_2=float(p["sigma_2"]), tau=int(round(p["tau"])),
        sigma_eps=float(p["sigma_eps"]), sigma_fund=float(p["sigma_fund"]),
        tau_cap=2000, noise_horizon="sqrt",
    )
    n_steps = agg * WINDOW
    o = ChiarellaIoriPerello(n_steps=n_steps, warmup=agg * 50, params=pr).run(seed=seed)

    # **価格がファンダに追随している走行しか採らない。**論文の主張そのもの
    # （"the market arrives at a price that follows closely the fundamental"）を
    # 制約として課す。最初は 0.2〜5.0 と緩くしていたが、それだと σ₂≈10 の
    # 「リターンの統計量は合うが価格がファンダの 31% に落ちている」点を
    # 拾ってしまった。板を壊さずにどこまで S&P500 に寄れるかを測るための門。
    ratio = float(np.median(o["prices"])) / pr.p_f0
    if not (PRICE_GATE[0] < ratio < PRICE_GATE[1]):
        raise ValueError(f"価格がファンダから離れている ({ratio:.3f})")

    lr = np.asarray(o["returns"], dtype=float)
    lr = lr[np.isfinite(lr)]
    k = (lr.size // agg) * agg
    if k < agg * WINDOW * 0.9:
        raise ValueError("集約後が短い")
    daily = lr[:k].reshape(-1, agg).sum(axis=1)   # 対数リターンは足し算で集約
    return daily[-WINDOW:]


def stats_of(p, seeds):
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


_T = {}


def _init(target, scale):
    _T["target"], _T["scale"] = target, scale


def _eval(p):
    s = stats_of(p, list(range(N_SEEDS)))
    if s is None:
        return np.inf, None
    loss = float(sum(((s[k] - _T["target"][k]) / _T["scale"][k]) ** 2 for k in FITTED))
    return loss, s


def main():
    t0 = time.time()
    _, target, scale, n_win = C.target_and_scale()
    print(f"目標: S&P500 直近{WINDOW}営業日、ばらつきは6指数{n_win}窓から")
    print(f"合わせる統計量: {', '.join(FITTED)}（残りは検証用）\n")

    lo = np.array([BOX[k][0] for k in KEYS], float)
    hi = np.array([BOX[k][1] for k in KEYS], float)
    islog = [BOX[k][2] for k in KEYS]

    u = qmc.Sobol(d=len(KEYS), scramble=True, seed=SEED).random(N_DRAWS)
    draws = []
    for row in u:
        p = {}
        for i, k in enumerate(KEYS):
            if islog[i]:
                p[k] = float(np.exp(np.log(max(lo[i], 1e-12)) + row[i] *
                                    (np.log(hi[i]) - np.log(max(lo[i], 1e-12)))))
            else:
                p[k] = float(lo[i] + row[i] * (hi[i] - lo[i]))
        draws.append(p)

    with Pool(N_PROC, initializer=_init, initargs=(target, scale)) as pool:
        pl, ps = _eval_wrap(pool, [PAPER])[0]
        print(f"論文パラメータの loss = {pl:.3f}" +
              ("" if np.isfinite(pl) else "  ← 走行が棄却された（価格が壊れる）"))

        res = _eval_wrap(pool, draws)
        best_i = int(np.argmin([r[0] for r in res]))
        bl, bs = res[best_i]
        best = draws[best_i]
        print(f"探索後 loss = {bl:.3f}   ({time.time()-t0:.0f}s, {N_DRAWS}点)")

        # 座標ごとの局所精錬
        for _ in range(3):
            cands, meta = [], []
            for i, k in enumerate(KEYS):
                span = (hi[i] - lo[i]) * 0.08
                for d in (-span, span):
                    c = dict(best)
                    c[k] = float(np.clip(best[k] + d, lo[i], hi[i]))
                    cands.append(c)
                    meta.append(k)
            out = _eval_wrap(pool, cands)
            j = int(np.argmin([o[0] for o in out]))
            if out[j][0] < bl - 1e-9:
                bl, bs, best = out[j][0], out[j][1], cands[j]
            else:
                break
        print(f"精錬後 loss = {bl:.3f}   ({time.time()-t0:.0f}s)\n")

    A = best["alpha"] * 50.0 * 300.0
    at_bound = []
    for i, k in enumerate(KEYS):
        f = (float(best[k]) - lo[i]) / (hi[i] - lo[i])
        if f <= 0.02 or f >= 0.98:
            at_bound.append(k)
    print("較正結果")
    for k in KEYS:
        mark = "  ← 境界" if k in at_bound else ""
        print(f"  {k:12} {best[k]:12.5g}   (論文 {PAPER[k]}){mark}")
    print(f"  A = α·N_S·pᶠ {A:11.1f}   (論文 1500)")
    print(f"  境界: {at_bound if at_bound else 'なし'}\n")

    print(f"  {'統計量':<26} {'S&P500':>9} {'モデル':>9} {'z':>7}  合わせた")
    summ = {}
    for k in bs:
        z = (bs[k] - target[k]) / scale[k]
        summ[k] = {"model": bs[k], "target": target[k], "z": float(z),
                   "fitted": k in FITTED}
        print(f"  {k:<26} {target[k]:9.4f} {bs[k]:9.4f} {z:7.2f}  "
              f"{'○' if k in FITTED else ''}")

    json.dump({"params": best, "A": A, "loss": bl, "paper_loss": pl, "at_bound": at_bound,
               "paper_params": PAPER, "fitted": list(FITTED), "stats": summ},
              open(os.path.join(HERE, "calibration_cip.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n書き出し: {os.path.join(HERE, 'calibration_cip.json')}"
          f"   全体 {time.time()-t0:.0f}s")


def _eval_wrap(pool, ps):
    return pool.map(_eval, ps)


if __name__ == "__main__":
    main()
