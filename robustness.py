"""頑健性の点検 — 指摘3・4・8への対応。

3. **横方向の依存。**6指数は同時期の世界株式で強く相関する。2008年や2020年の
   窓は全指数に同時に現れ、EURO STOXX 50 は DAX 構成銘柄を含むので一部は
   二重計上になる。窓を交換可能として扱うブートストラップは区間を狭く誤る。
   ここでは (a) 窓どうしの相関から実効標本数を推定し、(b) 指数×時期の
   ブロック単位で再抽出した区間を出して、素朴な版と並べる。

4. **混合の交絡。**「実データ」は6指数の混合なので、分離度が低いときに
   「統計量が鈍い」のか「指数間の異質性が大きい」のかを区別できない。
   ここでは指数ごとに分けた分離度を出す。

8. **Hill の k 固定。**k の選択でバイアスと分散が大きく動くのは極値理論で
   よく知られた罠（Hill horror plot）。k を 2.5% / 5% / 10% と振って
   結論が動くかを見る。
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from facts import BATTERY, evaluate, _hill      # noqa: E402
from generators import GENERATORS, build_context  # noqa: E402
from run_power import load_returns, real_windows, WINDOW, N_RUNS, SEED  # noqa: E402
from separation import ks_stat, perm_test        # noqa: E402

FRACS = (0.025, 0.05, 0.10)


def hill_at(r, frac, side="right"):
    x = r[r > 0] if side == "right" else -r[r < 0]
    return _hill(x, frac)


def effective_n(vals_by_window, names, dates):
    """窓どうしの相関から実効標本数を概算する。

    同一時期・別指数の窓は強く相関する。時期をブロックとみなし、
    ブロック内平均相関 rho からデザイン効果 1 + (m-1)rho を出す
    （m はブロックあたりの窓数）。
    """
    by_period = {}
    for v, n, d in zip(vals_by_window, names, dates):
        by_period.setdefault(d, []).append(v)
    groups = [g for g in by_period.values() if len(g) > 1]
    if not groups:
        return len(vals_by_window), 1.0
    allv = np.array(vals_by_window, float)
    tot = np.nanvar(allv)
    if tot <= 0:
        return len(vals_by_window), 1.0
    within = np.nanmean([np.nanvar(np.array(g, float)) for g in groups])
    rho = float(np.clip(1.0 - within / tot, 0.0, 0.99))
    m = float(np.mean([len(g) for g in groups]))
    deff = 1.0 + (m - 1.0) * rho
    return len(vals_by_window) / deff, deff


def main():
    rng = np.random.default_rng(SEED)
    series = load_returns()
    ctx = build_context(series["gspc"])
    ctx["pool"] = np.concatenate(list(series.values()))

    # 窓に指数名と「時期」（開始位置のブロック）を持たせる
    wins, names, periods = [], [], []
    for nm, r in series.items():
        for k, s in enumerate(range(0, len(r) - WINDOW + 1, 250)):
            wins.append(r[s:s + WINDOW])
            names.append(nm)
            periods.append(k)          # 同じ k は概ね同じ時期
    real = [evaluate(w) for w in wins]
    print(f"窓 {len(wins)} 本 / 指数 {len(series)} / 時期ブロック {len(set(periods))}\n")

    sims = {g: [evaluate(f(WINDOW, rng, ctx)) for _ in range(N_RUNS)]
            for g, f in GENERATORS.items()}

    out = {}

    # ---------------------------------------------------------------- 指摘3
    print("=== 3. 実効標本数（時期ブロック内の相関から） ===")
    print(f"{'統計量':<28}{'名目':>7}{'実効':>8}{'デザイン効果':>13}")
    print("-" * 58)
    eff = {}
    for s in BATTERY:
        v = [d[s] for d in real]
        n_eff, deff = effective_n(v, names, periods)
        eff[s] = {"n_eff": n_eff, "deff": deff}
        print(f"{s:<28}{len(v):>7}{n_eff:>8.0f}{deff:>13.2f}")
    out["effective_n"] = eff
    med = np.median([e["n_eff"] for e in eff.values()])
    print(f"\n→ 名目 {len(wins)} 本に対し実効は中央値 {med:.0f} 本。"
          f"ブートストラップ区間は約 sqrt({len(wins)/med:.1f}) = "
          f"{np.sqrt(len(wins)/med):.2f} 倍に広げるべき。")

    # ---------------------------------------------------------------- 指摘4
    print("\n=== 4. 指数ごとの分離度（混合の交絡） ===")
    idx = sorted(series)
    key_stats = ["leverage", "acf_abs_1", "gain_loss_asymmetry", "excess_kurtosis"]
    print(f"{'統計量':<24}{'混合':>8}" + "".join(i[:6].rjust(8) for i in idx))
    print("-" * (32 + 8 * len(idx)))
    per_index = {}
    for s in key_stats:
        rv = [d[s] for d in real]
        row = f"{s:<24}{ks_stat(rv, [d[s] for d in sims['garch_t']]):>8.2f}"
        per_index[s] = {}
        for i in idx:
            sel = [d[s] for d, n in zip(real, names) if n == i]
            k = ks_stat(sel, [d[s] for d in sims["garch_t"]]) if len(sel) >= 5 else np.nan
            per_index[s][i] = k
            row += (f"{k:.2f}" if np.isfinite(k) else "—").rjust(8)
        print(row)
    out["per_index_vs_garch_t"] = per_index
    print("\n→ 指数ごとの値が混合より系統的に大きければ、"
          "低い分離度は『統計量が鈍い』ではなく『指数間の異質性』が原因。")

    # ---------------------------------------------------------------- 指摘8
    print("\n=== 8. Hill 推定量の k 感度 ===")
    print(f"{'':<22}" + "".join(f"k={int(f*100)}%".rjust(10) for f in FRACS))
    print("-" * (22 + 10 * len(FRACS)))
    hs = {}
    for side in ("right", "left"):
        rv = {f: [hill_at(w, f, side) for w in wins] for f in FRACS}
        line = f"{'実データ '+side:<22}"
        for f in FRACS:
            line += f"{np.nanmedian(rv[f]):.2f}".rjust(10)
        print(line)
        hs[side] = {"real": {str(f): float(np.nanmedian(rv[f])) for f in FRACS}, "ks": {}}
        for g in ("garch_t", "block_bootstrap"):
            gv = {f: [] for f in FRACS}
            for _ in range(N_RUNS):
                x = GENERATORS[g](WINDOW, rng, ctx)
                for f in FRACS:
                    gv[f].append(hill_at(x, f, side))
            line = f"{'  vs '+g+' KS':<22}"
            for f in FRACS:
                k = ks_stat(rv[f], gv[f])
                hs[side]["ks"].setdefault(g, {})[str(f)] = float(k)
                line += f"{k:.2f}".rjust(10)
            print(line)
    out["hill_sensitivity"] = hs
    print("\n→ k で KS が大きく動くなら、hill_* の行は k の選択に依存している。")

    json.dump(out, open(os.path.join(HERE, "robustness.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n→ robustness.json")


if __name__ == "__main__":
    main()
