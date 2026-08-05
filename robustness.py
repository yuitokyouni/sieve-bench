"""頑健性の点検。

    python3 robustness.py     # → robustness.json（1分ほど）

## v0.2 で直したこと —— 「同じ時期」の定義が間違っていた

v0.1 はここで**窓の通し番号 k を「同じ時期」の代理**として使い、
`periods.append(k)` で実効標本数を出していた。これは成り立たない：

  - 営業日数が指数ごとに違う（FTSE 6316、日経 6121）ので、同じ k でも
    暦の上では最大1年近くずれる
  - EURO STOXX 50 は 2007-04 開始なので、その k=0 は他の指数の k=0 から
    **5年半ずれる**

つまり「同時期だから相関しているはず」と束ねていた集合が、実際には
同時期ではなかった。ここでは `windows.py` の暦ブロックで測り直し、
**旧版の値と並べて**どれだけ違ったかを出す。

そのうえで v0.1 から引き継ぐ点検を続ける：

  - 指数ごとに分けた分離度（「実データ」が6指数の混合であることの交絡）
  - Hill の k 感度（極値理論でよく知られた罠）
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from facts import BATTERY, evaluate, _hill                     # noqa: E402
from generators import GENERATORS, build_context               # noqa: E402
from resampling import intraclass_rho, ks_stat                 # noqa: E402
from windows import (BLOCK_WIDTHS, WINDOW, STRIDE, calendar_blocks,  # noqa: E402
                     load_series, real_windows)

FRACS = (0.025, 0.05, 0.10)
N_RUNS = 200
SEED = 20260802


def hill_at(r, frac, side="right"):
    x = r[r > 0] if side == "right" else -r[r < 0]
    return _hill(x, frac)


def design_effect(values, labels):
    """デザイン効果 1 + (m-1)rho と実効標本数。m はブロックあたりの平均本数。"""
    rho = intraclass_rho(values, labels)
    if not np.isfinite(rho):
        return len(values), 1.0, np.nan
    sizes = [int(np.sum(labels == u)) for u in sorted(set(np.asarray(labels).tolist()))]
    m = float(np.mean([s for s in sizes if s > 1] or [1.0]))
    deff = 1.0 + (m - 1.0) * rho
    return len(values) / deff, deff, rho


def main():
    rng = np.random.default_rng(SEED)
    series = load_series()
    ctx = build_context(series["gspc"][0], verbose=True)
    ctx["pool"] = np.concatenate([r for r, _ in series.values()])

    wins = real_windows(series)
    real = [evaluate(w.values) for w in wins]
    names = [w.index for w in wins]

    # 旧版の「時期」＝指数ごとの窓の通し番号
    legacy = []
    seen = {}
    for w in wins:
        seen[w.index] = seen.get(w.index, -1) + 1
        legacy.append(seen[w.index])
    legacy = np.array(legacy)

    span = calendar_blocks(wins, BLOCK_WIDTHS["span"])
    year = calendar_blocks(wins, BLOCK_WIDTHS["year"])

    print(f"\n窓 {len(wins)} 本 / 指数 {len(series)}")
    print(f"  旧版の『時期』（窓番号 k）      {len(set(legacy.tolist())):3d} 区分")
    print(f"  暦ブロック 1年幅                {len(set(year.tolist())):3d} 個")
    print(f"  暦ブロック 窓長幅（約4年）      {len(set(span.tolist())):3d} 個")

    out = {}

    # -------------------------------------------------- 1. 実効標本数の測り直し
    print("\n=== 1. 実効標本数 —— 旧版（窓番号）と新版（暦）の比較 ===")
    print(f"{'統計量':<28}{'旧 n_eff':>10}{'新 n_eff(年)':>14}"
          f"{'新 n_eff(窓長)':>15}{'rho(窓長)':>11}")
    print("-" * 80)
    eff = {}
    for s in BATTERY:
        v = [d[s] for d in real]
        n_old, _, _ = design_effect(v, legacy)
        n_yr, _, _ = design_effect(v, year)
        n_sp, deff, rho = design_effect(v, span)
        eff[s] = {"legacy_k": n_old, "calendar_year": n_yr,
                  "calendar_span": n_sp, "deff_span": deff, "rho_span": rho}
        print(f"{s:<28}{n_old:>10.0f}{n_yr:>14.0f}{n_sp:>15.0f}{rho:>11.2f}")
    out["effective_n"] = eff
    m_old = np.median([e["legacy_k"] for e in eff.values()])
    m_sp = np.median([e["calendar_span"] for e in eff.values()])
    print(f"\n→ 中央値: 旧 {m_old:.0f} 本 → 暦（窓長幅） {m_sp:.0f} 本。"
          f"名目 {len(wins)} 本に対する希釈は {len(wins)/m_sp:.1f} 倍。")
    print("  旧版は**そもそも同時期でない窓を同時期として束ねていた**ので、")
    print("  依存を過小評価していた。区間だけでなく p 値も影響を受ける")
    print("  （`separation.py` はブロック復元抽出で作り直した）。")

    # ------------------------------------------------------ 2. 混合の交絡
    print("\n=== 2. 指数ごとの分離度（『実データ』が6指数の混合であることの交絡）===")
    sims = {g: [evaluate(f(WINDOW, rng, ctx)) for _ in range(N_RUNS)]
            for g, f in GENERATORS.items()}
    idx = sorted(series)
    key_stats = ["leverage", "acf_abs_1", "return_skewness",
                 "gain_loss_asymmetry", "excess_kurtosis"]
    for target in ("garch_t", "gjr_t"):
        print(f"\n  対 {target}")
        print(f"  {'統計量':<26}{'混合':>7}" + "".join(i[:6].rjust(8) for i in idx))
        print("  " + "-" * (33 + 8 * len(idx)))
        per_index = {}
        for s in key_stats:
            rv = [d[s] for d in real]
            row = f"  {s:<26}{ks_stat(rv, [d[s] for d in sims[target]]):>7.2f}"
            per_index[s] = {}
            for i in idx:
                sel = [d[s] for d, n in zip(real, names) if n == i]
                k = (ks_stat(sel, [d[s] for d in sims[target]])
                     if len(sel) >= 5 else np.nan)
                per_index[s][i] = k
                row += (f"{k:.2f}" if np.isfinite(k) else "—").rjust(8)
            print(row)
        out[f"per_index_vs_{target}"] = per_index
    print("\n→ 指数ごとの値が混合より系統的に大きければ、低い分離度は")
    print("  『統計量が鈍い』ではなく『指数間の異質性による希釈』が原因。")

    # ------------------------------------------------------ 3. Hill の k 感度
    print("\n=== 3. Hill 推定量の k 感度 ===")
    print(f"{'':<24}" + "".join(f"k={int(f*100)}%".rjust(10) for f in FRACS))
    print("-" * (24 + 10 * len(FRACS)))
    hs = {}
    for side in ("right", "left"):
        rv = {f: [hill_at(w.values, f, side) for w in wins] for f in FRACS}
        line = f"{'実データ ' + side:<24}"
        for f in FRACS:
            line += f"{np.nanmedian(rv[f]):.2f}".rjust(10)
        print(line)
        hs[side] = {"real": {str(f): float(np.nanmedian(rv[f])) for f in FRACS},
                    "ks": {}}
        for g in ("garch_t", "gjr_t", "block_bootstrap"):
            gv = {f: [] for f in FRACS}
            for _ in range(N_RUNS):
                x = GENERATORS[g](WINDOW, rng, ctx)
                for f in FRACS:
                    gv[f].append(hill_at(x, f, side))
            line = f"{'  vs ' + g + ' KS':<24}"
            for f in FRACS:
                k = ks_stat(rv[f], gv[f])
                hs[side]["ks"].setdefault(g, {})[str(f)] = float(k)
                line += f"{k:.2f}".rjust(10)
            print(line)
    out["hill_sensitivity"] = hs
    print("\n→ k で KS が大きく動くなら、hill_* の行は k の選択に依存している。")

    out["config"] = {"window": WINDOW, "stride": STRIDE, "n_runs": N_RUNS,
                     "seed": SEED, "n_windows": len(wins),
                     "n_blocks_span": len(set(span.tolist())),
                     "n_blocks_year": len(set(year.tolist()))}
    json.dump(out, open(os.path.join(HERE, "robustness.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n→ robustness.json")


if __name__ == "__main__":
    main()
