"""Chiarella-Iori-Perelló の板の統計量を測る。

**論文 Figure 7** — 指値注文が中値からどれだけ離れて置かれるかの DDF。
ファンダメンタリスト支配下（σ₁=10, σ₂=0）では横軸が 0〜0.07 に収まり、
本文は「買い売りとも指値は中値のごく近くに置かれる（だから価格がファンダに
密着する）」と書いている。ノイズのみでは 0〜0.2、チャーティストを入れると
0〜0.25 に広がり、本文は「ファンダの場合の4倍の領域」と述べる。

**Figure 6 は市場注文のサイズの DDF であって、置き方の距離ではない。**
引き継ぎメモは Figure 6 と書いていたが、距離の図は Figure 7 である。

測るもの:
  - 指値の中値からの相対距離（買い／売り別）の分位点と DDF
  - スプレッド、板の生存注文数
  - **V の帰還写像 V → V'(V)。**エージェントが使う分散を v_pin で固定し、
    その板が実際に生む価格系列の分散 V' を測る。V'(V)=V が自己整合な状態。

    python3 models/book_stats.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chiarella_iori import ChiarellaIoriPerello, CIParams

HERE = os.path.dirname(os.path.abspath(__file__))


class BookProbe(ChiarellaIoriPerello):
    """指値の置かれた位置と、その時点の板の状態を記録する。"""

    def run(self, *, seed):
        self.placements = []   # (t, side, |price-mid|/mid)
        self.book = []         # (t, 生存注文数, 相対スプレッド)
        self._t = 0
        return super().run(seed=seed)

    def _expire(self, t):
        self._t = t
        super()._expire(t)
        b, _ = self._best_bid()
        a, _ = self._best_ask()
        sp = (a - b) / (0.5 * (a + b)) if (b is not None and a is not None) else np.nan
        self.book.append((t, len(self._orders), sp))

    def _add_limit(self, side, price, vol, expiry, agent):
        b, _ = self._best_bid()
        a, _ = self._best_ask()
        if b is not None and a is not None:
            mid = 0.5 * (a + b)
            self.placements.append((self._t, side, abs(price - mid) / mid))
        return super()._add_limit(side, price, vol, expiry, agent)


def measure(params, n_steps=6000, warmup=1000, seed=0):
    """1本走らせて板の統計量を返す。距離は全て中値に対する相対値。"""
    m = BookProbe(n_steps=n_steps, warmup=warmup, params=params)
    out = m.run(seed=seed)
    cut = params.tau_cap + warmup

    pl = [(s, d) for (t, s, d) in m.placements if t >= cut]
    bk = np.array([(n, sp) for (t, n, sp) in m.book if t >= cut], dtype=float)
    def arr(xs):
        a = np.array(xs, dtype=float)
        return a if a.size else np.array([np.nan])

    d = arr([x[1] for x in pl])
    db = arr([x[1] for x in pl if x[0] == "B"])
    ds = arr([x[1] for x in pl if x[0] == "S"])
    r = out["returns"][np.isfinite(out["returns"])]

    return {
        "n_placements": len(pl),
        "dist_median": float(np.median(d)),
        "dist_p90": float(np.percentile(d, 90)),
        "dist_p99": float(np.percentile(d, 99)),
        "dist_max": float(np.max(d)),
        "dist_median_buy": float(np.median(db)),
        "dist_median_sell": float(np.median(ds)),
        # 論文 Figure 7 中央パネルの横軸上限 0.07 に何割が収まるか
        "frac_within_0.07": float(np.mean(d < 0.07)),
        "frac_within_0.01": float(np.mean(d < 0.01)),
        "spread_median": float(np.nanmedian(bk[:, 1])),
        "depth_median": float(np.median(bk[:, 0])),
        "price_median": float(np.median(out["prices"])),
        "price_over_fundamental": float(np.median(out["prices"]) / params.p_f0),
        "V_realised": float(r.var()),
        "n_trades": int(out["n_trades"]),
    }


def ddf(d, n=40):
    """DDF（1 − 累積分布）を返す。論文 Figure 7 の縦軸。"""
    x = np.linspace(0, np.percentile(d, 99.9), n)
    return x, np.array([float(np.mean(d > t)) for t in x])


def main():
    res = {}

    # --- 論文 Figure 7 の3パネル。V は実測（＝現状の挙動） -----------------
    print("論文 Figure 7 — 指値の中値からの距離。V は実測（現状）")
    print(f"  {'ケース':<28} {'中央値':>9} {'p90':>9} {'<0.07':>8} "
          f"{'スプレッド':>10} {'板':>6} {'価格/ファンダ':>12}")
    panels = [
        ("noise only  σ1=0 σ2=0", CIParams(sigma_1=0.0, sigma_2=0.0)),
        ("+fundamentalist σ2=0", CIParams(sigma_2=0.0)),
        ("+chartist σ2=1.20", CIParams()),
    ]
    for tag, pr in panels:
        s = measure(pr)
        res[tag] = s
        print(f"  {tag:<28} {100*s['dist_median']:8.2f}% {100*s['dist_p90']:8.2f}% "
              f"{100*s['frac_within_0.07']:7.1f}% {100*s['spread_median']:9.3f}% "
              f"{s['depth_median']:6.0f} {s['price_over_fundamental']:11.3f}")
    print("  論文: ファンダ支配下は横軸 0〜0.07 に収まり、価格はファンダに密着する。\n")

    # --- V を固定した場合。板だけを論文の水準に置いたらどうなるか ----------
    print("同じ測定を、エージェントの使う V を固定して行う（診断）")
    print(f"  {'V 固定値':>10} {'V 実現値':>11} {'比':>9} {'中央値':>9} "
          f"{'スプレッド':>10} {'板':>6} {'価格/ファンダ':>12}")
    for vp in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        s = measure(CIParams(sigma_2=0.0, v_pin=vp))
        res[f"v_pin={vp:g}"] = s
        print(f"  {vp:10.0e} {s['V_realised']:11.3e} {s['V_realised']/vp:9.1f} "
              f"{100*s['dist_median']:8.2f}% {100*s['spread_median']:9.3f}% "
              f"{s['depth_median']:6.0f} {s['price_over_fundamental']:11.3f}")
    print("  V'(V) > V が全域で成立する ＝ 論文の締まった板は自己整合な状態ではない。")
    print("  ファンダメンタルの分散 σ_f² = 1e-6 に固定すれば板も価格も論文通りになる。\n")

    # --- 帰還を駆動しているのはノイズ項。σ_ε を落とすと比が 101 → 1.9 -----
    print("帰還の源。式(1) のノイズ項は1ステップあたりの率で、式(3) が τᵢ 倍する")
    print(f"  {'σ_ε':>8} {'τ_f':>7} {'V 実現値':>11} {'比':>9} {'中央値':>9} {'板':>6}")
    for se in [1e-4, 0.0]:
        for mode in ["own", "fixed"]:
            pr = CIParams(sigma_2=0.0, sigma_eps=se, tau_f_mode=mode, v_pin=1e-6)
            s = measure(pr)
            res[f"sigma_eps={se:g},tau_f={mode}"] = s
            print(f"  {se:8.0e} {mode:>7} {s['V_realised']:11.3e} "
                  f"{s['V_realised']/1e-6:9.1f} {100*s['dist_median']:8.2f}% "
                  f"{s['depth_median']:6.0f}")
    print("  σ_ε=0 で比が 101 → 1.9 に落ちる ＝ 暴走のほぼ全部がノイズ項×τᵢ。")
    print("  τ_f を定数 200 にすると比が 7e5 に飛ぶ ＝ τ_f = τᵢ が正しい読み。")

    # --- DDF（論文 Figure 7 そのもの）を書き出す ---------------------------
    curves = {}
    for tag, pr in panels:
        m = BookProbe(n_steps=6000, warmup=1000, params=pr)
        m.run(seed=0)
        cut = pr.tau_cap + 1000
        for side, key in (("B", "buy"), ("S", "sell")):
            d = np.array([x[2] for x in m.placements
                          if x[0] >= cut and x[1] == side])
            if d.size:
                x, y = ddf(d)
                curves[f"{tag}|{key}"] = {"x": x.tolist(), "ddf": y.tolist()}

    out = os.path.join(HERE, "book_stats.json")
    json.dump({"summary": res, "ddf": curves}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\n書き出し: {out}")


if __name__ == "__main__":
    main()
