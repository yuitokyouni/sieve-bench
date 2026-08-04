"""Chiarella-Iori-Perelló の板の統計量を測る。

**論文 Figure 7** — 指値注文が中値からどれだけ離れて置かれるかの DDF。
ファンダメンタリスト支配下（σ₁=10, σ₂=0）では横軸が 0〜0.07 に収まり、
本文は「買い売りとも指値は中値のごく近くに置かれる（だから価格がファンダに
密着する）」と書いている。ノイズのみでは 0〜0.2、チャーティストを入れると
0〜0.25 に広がり、本文は「ファンダの場合の4倍の領域」と述べる。

**Figure 6 は成行注文のサイズの DDF であって、置き方の距離ではない。**

測るもの:
  1. 論文 Figure 7 の3パネル（ノイズのみ／＋ファンダ／＋チャーティスト）
  2. **V の帰還写像 V → V'(V)。**エージェントが使う分散を v_pin で固定し、
     その板が実際に生む価格系列の分散 V' を測る。V'(V)=V が自己整合な状態。
  3. **利得 A = α·N_S·pᶠ 依存性。**注文の抽選区間の対数幅は恒等式
     ln(p_M/p_m) = αᵢ·Vᵢ·(p_m·Sᵢ + Cᵢ) なので、板の締まり具合はこの群だけで
     決まる。論文値 A=1500 では利得が 1 を超え、論文自身の Figure 7 が出ない。
  4. **注文が板から消える経路の内訳。**「相手の資力不足で削除」は論文が
     規定していない機構だが、実際には支配的な経路になっている。

    python3 models/book_stats.py        # 約40秒
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chiarella_iori import ChiarellaIoriPerello, CIParams

HERE = os.path.dirname(os.path.abspath(__file__))


class BookProbe(ChiarellaIoriPerello):
    """指値の置かれた位置、板の状態、注文が消える経路を記録する。"""

    def run(self, *, seed):
        self.placements = []   # (t, side, |price-mid|/mid)
        self.book = []         # (t, 生存注文数, 相対スプレッド)
        self.filled = self.dropped = self.expired = 0
        self._t = 0
        return super().run(seed=seed)

    def _expire(self, t):
        self._t = t
        n0 = len(self._orders)
        super()._expire(t)
        self.expired += n0 - len(self._orders)
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

    def _market(self, i, side, vol, S, C, t, ti, draw, p_hat, aV):
        """親と同一の論理に、注文が消える経路の計数だけを足したもの。"""
        if vol <= 0:
            return None
        last, remaining = None, vol
        while remaining > 1e-12:
            if side == "B":
                pr, oid = self._best_ask()
                if pr is None or pr > draw:
                    break
            else:
                pr, oid = self._best_bid()
                if pr is None or pr < draw:
                    break
            o = self._orders[oid]
            cp = o[3]
            q = min(remaining, o[1])
            if side == "B":
                q = min(q, C[i] / pr if pr > 0 else 0.0, max(S[cp], 0.0))
            else:
                q = min(q, max(S[i], 0.0), C[cp] / pr if pr > 0 else 0.0)
            if q <= 1e-12:
                # ← 論文が規定していない機構。相手に在庫／現金が無い気配を消す
                self._orders.pop(oid, None)
                self.dropped += 1
                continue
            cost = q * pr
            if side == "B":
                S[i] += q; C[i] -= cost; S[cp] -= q; C[cp] += cost
            else:
                S[i] -= q; C[i] += cost; S[cp] += q; C[cp] -= cost
            o[1] -= q
            remaining -= q
            last = pr
            if o[1] <= 1e-12:
                self._orders.pop(oid, None)
                self.filled += 1
        if remaining > 1e-12:
            self._add_limit(side, self._round(draw), remaining, t + ti, i)
        return last


def measure(params, n_steps=6000, warmup=1000, seed=0):
    """1本走らせて板の統計量を返す。距離は全て中値に対する相対値。"""
    m = BookProbe(n_steps=n_steps, warmup=warmup, params=params)
    out = m.run(seed=seed)
    cut = params.tau_cap + warmup

    def arr(xs):
        a = np.array(xs, dtype=float)
        return a if a.size else np.array([np.nan])

    pl = [(s, d) for (t, s, d) in m.placements if t >= cut]
    bk = np.array([(n, sp) for (t, n, sp) in m.book if t >= cut], dtype=float)
    d = arr([x[1] for x in pl])
    r = out["returns"][np.isfinite(out["returns"])]
    gone = m.filled + m.dropped + m.expired

    return {
        "n_placements": len(pl),
        "dist_median": float(np.median(d)),
        "dist_p90": float(np.percentile(d, 90)),
        "dist_median_buy": float(np.median(arr([x[1] for x in pl if x[0] == "B"]))),
        "dist_median_sell": float(np.median(arr([x[1] for x in pl if x[0] == "S"]))),
        # 論文 Figure 7 中央パネルの横軸上限 0.07 に何割が収まるか
        "frac_within_0.07": float(np.mean(d < 0.07)),
        "spread_median": float(np.nanmedian(bk[:, 1])),
        "depth_median": float(np.median(bk[:, 0])),
        "price_over_fundamental": float(np.median(out["prices"]) / params.p_f0),
        "V_realised": float(r.var()),
        "n_trades": int(out["n_trades"]),
        "gone_filled": m.filled, "gone_dropped": m.dropped,
        "gone_expired": m.expired,
        "drop_share": float(m.dropped / max(gone, 1)),
    }


def ddf(d, n=40):
    """DDF（1 − 累積分布）。論文 Figure 7 の縦軸。"""
    x = np.linspace(0, np.percentile(d, 99.9), n)
    return x, np.array([float(np.mean(d > t)) for t in x])


def main():
    res = {}

    print("1) 論文 Figure 7 の3パネル。ノイズの積み方 linear（論文の字義）と sqrt")
    print(f"   {'ケース':<26} {'ノイズ':>7} {'中央値':>9} {'<0.07':>7} "
          f"{'スプレッド':>10} {'板':>6} {'価格/ファンダ':>12}")
    panels = [
        ("noise only  σ1=0 σ2=0", dict(sigma_1=0.0, sigma_2=0.0)),
        ("+fundamentalist σ2=0", dict(sigma_2=0.0)),
        ("+chartist σ2=1.20", dict()),
    ]
    for tag, kw in panels:
        for nh in ["linear", "sqrt"]:
            s = measure(CIParams(noise_horizon=nh, **kw))
            res[f"{tag}|{nh}"] = s
            print(f"   {tag:<26} {nh:>7} {100*s['dist_median']:8.2f}% "
                  f"{100*s['frac_within_0.07']:6.1f}% {100*s['spread_median']:9.3f}% "
                  f"{s['depth_median']:6.0f} {s['price_over_fundamental']:11.3f}")
    print("   論文: ファンダ支配下は横軸 0〜0.07 に収まり、価格はファンダに密着。\n")

    print("2) 帰還写像 V → V'(V)。自己整合なら比が 1")
    print(f"   {'V 固定値':>10} {'linear の V2':>14} {'比':>9} "
          f"{'sqrt の V2':>13} {'比':>9}   （V2 = V′）")
    for vp in [1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        a = measure(CIParams(sigma_2=0.0, v_pin=vp, noise_horizon="linear"))
        b = measure(CIParams(sigma_2=0.0, v_pin=vp, noise_horizon="sqrt"))
        res[f"v_pin={vp:g}|linear"] = a
        res[f"v_pin={vp:g}|sqrt"] = b
        print(f"   {vp:10.0e} {a['V_realised']:14.3e} {a['V_realised']/vp:9.1f} "
              f"{b['V_realised']:13.3e} {b['V_realised']/vp:9.2f}")
    print("   linear は全域で V' > V ＝ 論文の締まった板が自己整合な状態にならない。")
    print("   式(1) の ε は1ステップの率なのに式(3) が τᵢ 倍するため、σ_ε=1e-4 が")
    print("   τᵢ/Σg≈115 倍されて 1.15% の床を作る。sqrt で積むと床は 8.5e-7。\n")

    print("3) 利得は A = α·N_S·pᶠ だけで決まる（ln(p_M/p_m)=αᵢ·Vᵢ·(p_m·Sᵢ+Cᵢ) の帰結）")
    print(f"   {'α':>7} {'A':>8} {'利得(V=1e-6)':>13} | 自由走行: "
          f"{'V':>10} {'中央値':>9} {'スプレッド':>10} {'価格/ファンダ':>12}")
    for al in [0.10, 0.05, 0.03, 0.02, 0.01]:
        g = measure(CIParams(sigma_2=0.0, alpha=al, v_pin=1e-6, noise_horizon="sqrt"))
        f = measure(CIParams(sigma_2=0.0, alpha=al, noise_horizon="sqrt"))
        res[f"alpha={al:g}|pinned"] = g
        res[f"alpha={al:g}|free"] = f
        mark = "  ← 利得<1" if g["V_realised"] / 1e-6 < 1.0 else ""
        print(f"   {al:7.3f} {al*50*300:8.0f} {g['V_realised']/1e-6:13.2f} | "
              f"{f['V_realised']:10.3e} {100*f['dist_median']:8.2f}% "
              f"{100*f['spread_median']:9.3f}% {f['price_over_fundamental']:11.3f}{mark}")
    print("   論文値 α=0.1（A=1500）では利得が 1 を超える。**論文の Figure 7 は")
    print("   論文自身のパラメータでは再現しない。**\n")

    print("4) 生存注文が板から消える経路（論文は資力不足の扱いを規定していない）")
    print(f"   {'設定':<28} {'約定':>8} {'資力不足':>9} {'期限切れ':>9} {'削除の割合':>10}")
    for tag, kw in [("linear α=0.1（論文の字義）", dict(noise_horizon="linear")),
                    ("sqrt α=0.1", dict(noise_horizon="sqrt")),
                    ("sqrt α=0.03（利得<1）", dict(noise_horizon="sqrt", alpha=0.03))]:
        s = measure(CIParams(sigma_2=0.0, **kw))
        res[f"removal|{tag}"] = s
        print(f"   {tag:<28} {s['gone_filled']:8d} {s['gone_dropped']:9d} "
              f"{s['gone_expired']:9d} {100*s['drop_share']:9.1f}%")
    print("   資力不足による削除が支配的。ここは論文に規定が無い実装上の選択で、")
    print("   発注時に現金・在庫を拘束する（エスクロー）読み方もありうる。")

    # --- DDF（論文 Figure 7 そのもの）を書き出す ---------------------------
    curves = {}
    for tag, kw in panels:
        for nh in ["linear", "sqrt"]:
            pr = CIParams(noise_horizon=nh, **kw)
            m = BookProbe(n_steps=6000, warmup=1000, params=pr)
            m.run(seed=0)
            cut = pr.tau_cap + 1000
            for side, key in (("B", "buy"), ("S", "sell")):
                d = np.array([x[2] for x in m.placements
                              if x[0] >= cut and x[1] == side])
                if d.size:
                    x, y = ddf(d)
                    curves[f"{tag}|{nh}|{key}"] = {"x": x.tolist(), "ddf": y.tolist()}

    out = os.path.join(HERE, "book_stats.json")
    json.dump({"summary": res, "ddf": curves}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\n書き出し: {out}")


if __name__ == "__main__":
    main()
