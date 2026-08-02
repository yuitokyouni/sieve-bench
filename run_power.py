"""識別力の測定。

問い：**どの統計量が、実データと生成データを実際に分けるか。**

各統計量は1本の系列を1つの数に落とす。実データの窓を多数集めればその数の分布が出る。
生成器の実行を多数集めれば、そちらの分布も出る。**2つの分布がどれだけ重なるか**が
その統計量の識別力である。重なりの測り方には AUC（ROC曲線下面積）を使う。

  AUC = 0.5 → 全く分けられない（その統計量はその生成器に対して無力）
  AUC = 1.0 → 完全に分かれる

表に出すのは power = 2 * |AUC - 0.5| で、0（無力）から 1（完全）まで。

この実験の要点は、**落とすことではなく落とせないことを見つけること**にある。
iid_bootstrap（周辺分布が実データと完全に同一）や garch_t（強い基準線）を
分けられない統計量は、実務で情報を持たない。それを名指しするのが目的である。
"""

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facts import BATTERY, evaluate            # noqa: E402
from generators import GENERATORS, build_context  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

WINDOW = 1000      # 1本の系列の長さ（約4年の営業日）
STRIDE = 250       # 実データの窓をずらす幅
N_RUNS = 200       # 生成器ごとの実行回数
N_BOOT = 1000      # power の信頼区間を作るブートストラップ回数
SEED = 20260802


def load_returns():
    """キャッシュ済みの指数から対数リターンを作り、各指数の標準偏差で割る。

    全統計量はスケール不変なので、この標準化は結果を変えない。
    振幅の違いを「差」として拾わないことを明示するために入れてある。
    """
    if not os.path.isdir(DATA) or not any(
            f.endswith(".json") for f in os.listdir(DATA)):
        sys.exit("data/ が空。先に `python3 fetch.py` を実行すること。\n"
                 "（指数データは再配布しないので同梱されていない）")
    out = {}
    for fn in sorted(os.listdir(DATA)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(DATA, fn)))["chart"]["result"][0]
        c = np.array([x for x in d["indicators"]["quote"][0]["close"]
                      if x is not None], dtype=float)
        r = np.diff(np.log(c))
        r = r[np.isfinite(r)]
        out[fn[:-5]] = r / r.std()
    return out


def real_windows(series, window=WINDOW, stride=STRIDE):
    w = []
    for name, r in series.items():
        for s in range(0, len(r) - window + 1, stride):
            w.append((name, r[s:s + window]))
    return w


def auc(real_vals, synth_vals):
    """P(統計量[生成] > 統計量[実データ])。分けられなければ 0.5。"""
    a = np.asarray(real_vals, dtype=float)
    b = np.asarray(synth_vals, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    r = rankdata(np.concatenate([a, b]))
    u_a = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return float(1.0 - u_a / (len(a) * len(b)))


def power_ci(real_vals, synth_vals, rng, n_boot=N_BOOT):
    """power の 95% ブートストラップ区間。

    実データ側・生成器側をそれぞれ復元抽出し直して power を計算し直す。

    **これは窓の重なりを解決しない。**実データの窓は 250 ずつずらしただけで
    互いに独立ではないので、この区間は「独立標本だったとしたら」の幅である。
    重なりの影響そのものは、窓を重ねない版（stride = window）を別に出して
    比べることで見る。
    """
    a = np.asarray(real_vals, dtype=float)
    b = np.asarray(synth_vals, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    out = np.empty(n_boot)
    for i in range(n_boot):
        x = auc(rng.choice(a, len(a), replace=True),
                rng.choice(b, len(b), replace=True))
        out[i] = np.nan if not np.isfinite(x) else 2 * abs(x - 0.5)
    return (float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)))


def main():
    rng = np.random.default_rng(SEED)
    series = load_returns()
    pool = np.concatenate(list(series.values()))
    print(f"指数 {len(series)} 本、リターン {len(pool)} 点", flush=True)

    print("生成器のパラメータを実データから推定中（ここが「合わせ込んだ量」）…",
          flush=True)
    ctx = build_context(series["gspc"])
    o, a_, b_ = ctx["garch"]
    ctx["pool"] = pool
    print(f"  GARCH(1,1): omega={o:.5f} alpha={a_:.4f} beta={b_:.4f} "
          f"(alpha+beta={a_ + b_:.4f})", flush=True)
    print(f"  Student-t df={ctx['t_df']:.2f}", flush=True)

    wins = real_windows(series)
    print(f"実データの窓 {len(wins)} 本（長さ {WINDOW}）", flush=True)
    real = [evaluate(w) for _, w in wins]

    results = {}
    for gname, gfn in GENERATORS.items():
        vals = [evaluate(gfn(WINDOW, rng, ctx)) for _ in range(N_RUNS)]
        results[gname] = vals
        print(f"  {gname}: {N_RUNS} 回", flush=True)

    stats_names = list(BATTERY.keys())
    gen_names = list(GENERATORS.keys())
    power = {}
    signed = {}
    ci = {}
    for s in stats_names:
        rv = [d[s] for d in real]
        power[s] = {}
        signed[s] = {}
        ci[s] = {}
        for g in gen_names:
            sv = [d[s] for d in results[g]]
            a = auc(rv, sv)
            signed[s][g] = a
            power[s][g] = np.nan if not np.isfinite(a) else 2 * abs(a - 0.5)
            ci[s][g] = power_ci(rv, sv, rng)

    # 窓を重ねない版（stride = window）。実データ側の標本数は減るが、
    # 窓どうしが独立になる。重なりが結論を作っていないかの確認。
    wins_nd = real_windows(series, stride=WINDOW)
    real_nd = [evaluate(w) for _, w in wins_nd]
    print(f"重ならない窓 {len(wins_nd)} 本でも計算する", flush=True)
    power_nd = {}
    for s in stats_names:
        rv = [d[s] for d in real_nd]
        power_nd[s] = {}
        for g in gen_names:
            a = auc(rv, [d[s] for d in results[g]])
            power_nd[s][g] = np.nan if not np.isfinite(a) else 2 * abs(a - 0.5)

    out = {
        "config": {"window": WINDOW, "stride": STRIDE, "n_runs": N_RUNS,
                   "n_boot": N_BOOT, "n_windows_nonoverlap": len(wins_nd),
                   "seed": SEED, "n_real_windows": len(wins),
                   "indices": sorted(series.keys()),
                   "garch": list(ctx["garch"]), "t_df": ctx["t_df"]},
        "power": power, "auc": signed, "power_ci95": ci,
        "power_nonoverlap": power_nd,
        "real_median": {s: float(np.nanmedian([d[s] for d in real]))
                        for s in stats_names},
    }
    with open(os.path.join(HERE, "power.json"), "w") as f:
        json.dump(out, f, indent=1)

    # --- 表示 -------------------------------------------------------------
    w0 = max(len(s) for s in stats_names) + 1
    hdr = "統計量".ljust(w0) + "".join(g[:13].rjust(15) for g in gen_names)
    print("\n識別力  power = 2|AUC-0.5|   0=分けられない  1=完全に分かれる\n")
    print(hdr)
    print("-" * len(hdr))
    for s in stats_names:
        row = s.ljust(w0)
        for g in gen_names:
            v = power[s][g]
            row += ("   nan" if not np.isfinite(v) else f"{v:.3f}").rjust(15)
        print(row)
    print("-" * len(hdr))

    print("\n生成器ごとに、それを最もよく捉えた統計量：")
    for g in gen_names:
        rank = sorted(((power[s][g], s) for s in stats_names
                       if np.isfinite(power[s][g])), reverse=True)
        top = ", ".join(f"{s}({v:.2f})" for v, s in rank[:3])
        print(f"  {g:16s} {top}")

    print(f"\n統計量ごとの最悪ケース（全生成器の中で最も分けられなかった相手）"
          f"／95%区間は {N_BOOT} 回のブートストラップ"
          f"／重ならない窓 {len(wins_nd)} 本での同じ値も併記")
    print(f"  {'統計量':<26}{'最悪 power':>11}{'95%区間':>18}"
          f"{'重ねない窓':>12}   相手")
    print("  " + "-" * 82)
    rank = []
    for s in stats_names:
        vs = [(power[s][g], g) for g in gen_names if np.isfinite(power[s][g])]
        if vs:
            rank.append((min(vs)[0], min(vs)[1], s))
    for v, g, s in sorted(rank, reverse=True):
        lo, hi = ci[s][g]
        nd = power_nd[s][g]
        cis = "     —" if not np.isfinite(lo) else f"[{lo:.3f}, {hi:.3f}]"
        nds = "  —" if not np.isfinite(nd) else f"{nd:.3f}"
        print(f"  {s:<26}{v:>11.3f}{cis:>18}{nds:>12}   {g}")

    print("\n**区間が広い行は、その数字を桁でしか読めない。**\n"
          "区間は実データ側・生成器側の復元抽出によるもので、"
          "窓の重なりは解消していない。\n"
          "重なりの影響は「重ねない窓」の列と比べて判断すること。")


if __name__ == "__main__":
    main()
