"""識別力の測定 —— どの統計量が、どの生成器を見破れるか。

各統計量は1本のリターン系列を1つの数に落とす。実データの窓を多数集めれば
その数の分布が出る。生成器の実行を多数集めれば、そちらの分布も出る。
**2つの分布がどれだけ重なるか**が、その統計量の識別力である。
重なりは2標本 Kolmogorov-Smirnov 統計量（分布関数の最大乖離）で測る。

    python3 separation.py     # → separation.json（2分ほど）

## v0.2 で直したこと

**1. p 値の出し方が間違っていた。**v0.1 は実データ 124 本と生成 200 本を
1つのプールに入れてラベルを完全にランダムに入れ替えていた。これは 324 個が
交換可能という仮定だが、実データの窓は
（a）ストライド 250・窓長 1000 で隣と 75% 重なり、
（b）同時期の各指数が相関し、
（c）EURO STOXX 50 は DAX 構成銘柄を含む。
`selftest.py` で測ったところ、**帰無仮説が真でもブロック内相関 0.6 なら
56% 棄却していた**（名目 5%）。ここでは交換の単位を暦のブロックに上げる
（`resampling.block_boot_test`）。

**2. それでも名目水準には届かない。**ただし**依存を扱う手法が無いわけではない。**
暦ブロック単位の randomization inference、block bootstrap、wild cluster bootstrap、
HAC など、依存を保つ方法は存在する。ここで測ったのは2つだけである。
消せないのは手法ではなく**独立な情報単位の不足**で、25年を4年窓で見れば
独立な暦は6個しか無い。**そこで閾値を較正した。**名目 p < 0.01 で真の大きさが
3〜5% に収まる（`selftest.py`）。以下ではこれを 5% 水準の判定線として使う。

**2b. 正しく補正すると、独立情報が無いことがそのまま出る。**実データ同士の対照は
同じ暦ブロックを両側が共有する**対応のある設計**なので、ブロックごとのラベル
入れ替えで検定する（`paired_block_test`）。割り当ては 2^6 = 64 通りしかなく、
p の刻みは 0.016。**上の判定線 0.01 には構造上到達できない。**
これは検定の欠陥ではなく、6つ分の独立な暦しか無いことの直接の帰結である。

**3. 天井という呼び方をやめた。**米欧 vs アジアの差は「越えるべき天井」ではなく、
**現実どうしのばらつきの一例**にすぎない。制度・通貨・産業構成・期間が違うので、
それより差が小さいことは対象市場を再現している証拠にならない。同じ市場の
別時代どうし、似た2市場どうしなど、参照を複数並べる形に変えた。

**4. 周辺分布を15個並べるだけでは足りない。**個別には全部合っていて同時分布が
壊れている生成器を通してしまう。統計量ベクトル全体の energy 検定を足した。
個別の KS は「何が壊れているか」の診断として残す。

**5. 多重性。**16統計量 × 8生成器 = 128 セル。Benjamini-Hochberg の q 値を出す。
"""

import json
import os
import sys
from collections import Counter
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facts import BATTERY, evaluate                                # noqa: E402
from generators import GENERATORS, HELD_OUT, build_contexts        # noqa: E402
from resampling import (benjamini_hochberg, block_boot_test,       # noqa: E402
                        block_perm_test, energy_block_test,
                        intraclass_rho, iid_perm_test, ks_stat,
                        paired_block_test)
from windows import (BLOCK_WIDTHS, WINDOW, STRIDE, calendar_blocks,  # noqa: E402
                     describe_blocks, load_series, real_windows)

N_DRAW = 2000
N_RUNS = 200
SEED = 20260802

# `selftest.py` の較正表から。名目 p < この値で真の大きさが 3〜5%。
ALPHA = 0.01

# 参照対照（現実どうしのばらつき）。天井ではない。
GROUP_A = {"gspc", "ftse", "gdaxi", "sx5e"}   # 米欧
GROUP_B = {"n225", "hsi"}                     # アジア
ERA_SPLIT = 2014                              # 前半／後半の境目


def contrasts(wins):
    """実データどうしの対照を作る。返り値 {名前: (左の添字, 右の添字, 説明)}。"""
    idx = range(len(wins))
    out = {}
    out["region"] = ([i for i in idx if wins[i].index in GROUP_A],
                     [i for i in idx if wins[i].index in GROUP_B],
                     "米欧 vs アジア（制度・通貨・産業構成が違う）")
    out["era"] = ([i for i in idx if wins[i].start.year < ERA_SPLIT],
                  [i for i in idx if wins[i].start.year >= ERA_SPLIT],
                  f"同じ6指数の {ERA_SPLIT} 年より前 vs 後（同じ市場の別時代）")
    out["gspc_vs_ftse"] = ([i for i in idx if wins[i].index == "gspc"],
                           [i for i in idx if wins[i].index == "ftse"],
                           "S&P500 vs FTSE100（最も似た2市場）")
    same = [i for i in idx if wins[i].index == "gspc"]
    out["gspc_split"] = (same[0::2], same[1::2],
                         "S&P500 の窓を1本おきに二分（同一市場・同一時代）")
    return out


def main():
    rng = np.random.default_rng(SEED)
    series = load_series()
    pool = np.concatenate([r for r, _ in series.values()])
    print(f"指数 {len(series)} 本、リターン {len(pool)} 点", flush=True)

    wins = real_windows(series)
    blocks = calendar_blocks(wins, BLOCK_WIDTHS["span"])
    binfo = describe_blocks(wins, blocks)
    real = [evaluate(w.values) for w in wins]
    print(f"\n実データの窓 {len(wins)} 本 → 暦ブロック {len(binfo)} 個", flush=True)
    for b in binfo:
        print(f"    ブロック{b['block']}  {b['start']} 〜 {b['end']}  "
              f"{b['n']:2d}本  {len(b['indices'])}指数")

    # **指数ごとに当てる。**v0.2 までは S&P500 だけに当てたパラメータを6指数
    # すべての基準線に使っていたため、KS の中に「機構の欠落」と「市場間の
    # 異質性（ボラ水準・裾・持続性）」が混ざり、前者に帰属できなかった。
    ctxs = build_contexts(series, verbose=True)
    counts = Counter(w.index for w in wins)
    alloc = {n: max(1, round(N_RUNS * counts[n] / len(wins))) for n in sorted(counts)}
    n_total = sum(alloc.values())
    print(f"\n指数ごとのパラメータと割り当て（実データの窓の構成に合わせる）")
    print(f"  {'指数':<8}{'窓':>4}{'生成':>6}{'alpha':>8}{'gamma':>8}{'beta':>8}"
          f"{'nu':>7}")
    for n in sorted(counts):
        o, a, g, b, nu = ctxs[n]["gjr_t"]
        print(f"  {n:<8}{counts[n]:>4}{alloc[n]:>6}{a:>8.4f}{g:>8.4f}{b:>8.4f}"
              f"{nu:>7.2f}")

    results = {}
    for gname, gfn in GENERATORS.items():
        runs = []
        for n, k in alloc.items():
            runs += [evaluate(gfn(WINDOW, rng, ctxs[n])) for _ in range(k)]
        results[gname] = runs
        print(f"  {gname}: {n_total} 回", flush=True)

    stats_names = list(BATTERY.keys())
    gen_names = list(GENERATORS.keys())

    out = {"config": {"window": WINDOW, "stride": STRIDE, "n_runs": N_RUNS,
                      "n_draw": N_DRAW, "seed": SEED, "alpha": ALPHA,
                      "n_real_windows": len(wins), "n_blocks": len(binfo),
                      "alloc": alloc, "n_gen_runs": n_total,
                      "fit_per_index": True,
                      "block_width_days": BLOCK_WIDTHS["span"],
                      "held_out_generators": list(HELD_OUT)},
           "blocks": binfo, "rho": {}, "ks": {},
           "p": {}, "p_blockperm": {}, "p_iid": {}, "q": {},
           "joint_energy": {}, "reference_contrasts": {}}

    # ------------------------------------------------------------- 個別診断層
    print("\n個別の統計量ごとに測る（帰無分布は3通り）…", flush=True)
    for s in stats_names:
        rv = [d[s] for d in real]
        out["rho"][s] = intraclass_rho(rv, blocks)
        out["ks"][s], out["p"][s] = {}, {}
        out["p_blockperm"][s], out["p_iid"][s] = {}, {}
        for g in gen_names:
            gv = [d[s] for d in results[g]]
            k, p, _, _ = block_boot_test(rv, blocks, gv, ks_stat, rng, N_DRAW)
            _, p2, _, _ = block_perm_test(rv, blocks, gv, ks_stat, rng, N_DRAW)
            _, p3, _ = iid_perm_test(rv, gv, ks_stat, rng, N_DRAW)
            out["ks"][s][g], out["p"][s][g] = k, p
            out["p_blockperm"][s][g], out["p_iid"][s][g] = p2, p3
        print(f"  {s}", flush=True)

    flat = [(s, g) for s in stats_names for g in gen_names]
    q = benjamini_hochberg([out["p"][s][g] for s, g in flat])
    for (s, g), qq in zip(flat, q):
        out["q"].setdefault(s, {})[g] = float(qq) if np.isfinite(qq) else None

    # --------------------------------------------------------------- 総合層
    print("\n統計量ベクトル全体の energy 検定…", flush=True)
    A = np.array([[d[s] for s in stats_names] for d in real])
    for g in gen_names:
        B = np.array([[d[s] for s in stats_names] for d in results[g]])
        e, p, n95 = energy_block_test(A, blocks, B, rng, N_DRAW)
        out["joint_energy"][g] = {"energy": e, "pvalue": p, "null95": n95}
        print(f"  {g:16s} E = {e:8.2f}  p = {p:.4f}", flush=True)

    # ----------------------------------------------------- 現実どうしのばらつき
    print("\n参照対照（現実どうしのばらつき）…", flush=True)
    for cname, (ia, ib, desc) in contrasts(wins).items():
        ba, bb = set(blocks[ia].tolist()), set(blocks[ib].tolist())
        # 両側が同じ暦ブロックを共有しているなら、それは対応のある設計である。
        # 同時期の米欧とアジアは共通ショックを受けるので独立な単位ではない。
        paired = ba == bb
        row = {"desc": desc, "n_a": len(ia), "n_b": len(ib),
               "design": "paired_block" if paired else "unpaired_block",
               "ks": {}, "pvalue": {}}
        for s in stats_names:
            va = [real[i][s] for i in ia]
            vb = [real[i][s] for i in ib]
            if paired:
                k, p, nb, tot = paired_block_test(va, blocks[ia], vb, blocks[ib],
                                                  ks_stat, N_DRAW, rng)
            else:
                k, p, _, _ = block_perm_test(va, blocks[ia], vb, ks_stat, rng,
                                             N_DRAW, b_blocks=blocks[ib])
                nb = len(ba) + len(bb)
                tot = comb(len(ba) + len(bb), len(ba))
            row["ks"][s], row["pvalue"][s] = k, p
        row["n_blocks"], row["n_assignments"] = nb, int(tot)
        row["p_resolution"] = 1.0 / tot if tot else None
        out["reference_contrasts"][cname] = row
        print(f"  {cname:16s} {len(ia):3d} vs {len(ib):3d}  "
              f"{row['design']}  割り当て {int(tot):,} 通り "
              f"（p の刻み {1.0/tot:.3f}）", flush=True)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "separation.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    report(out, stats_names, gen_names)


def report(out, stats_names, gen_names):
    def cell(v, p):
        if not np.isfinite(v):
            return "—"
        mark = "*" if p < ALPHA else " "
        return f"{v:.2f}({p:.3f}){mark}"

    w0 = max(len(s) for s in stats_names) + 1
    print("\n\nKS 統計量（分布関数の最大乖離）。括弧内はブロック復元抽出の p 値。")
    print(f"* は較正済みの判定線 p < {ALPHA}（真の大きさ 3〜5%）を下回ったもの。")
    print("右2列 gjr_t / egarch_t は**統計量を選んだ後に足した held-out**。\n")
    hdr = "統計量".ljust(w0) + "".join(g[:11].rjust(16) for g in gen_names)
    print(hdr)
    print("-" * len(hdr))
    for s in stats_names:
        row = s.ljust(w0)
        for g in gen_names:
            row += cell(out["ks"][s][g], out["p"][s][g]).rjust(16)
        print(row)
    print("-" * len(hdr))

    print("\n帰無分布の作り方でどれだけ変わるか（判定線 p < "
          f"{ALPHA} を下回るセル数 / {len(stats_names) * len(gen_names)}）")
    for key, label in (("p_iid", "完全置換（v0.1・無効）"),
                       ("p_blockperm", "ブロック置換"),
                       ("p", "ブロック復元抽出（採用）")):
        n = sum(1 for s in stats_names for g in gen_names
                if np.isfinite(out[key][s][g]) and out[key][s][g] < ALPHA)
        print(f"  {label:<26} {n:3d}")
    nq = sum(1 for s in stats_names for g in gen_names
             if out["q"][s][g] is not None and out["q"][s][g] < 0.05)
    print(f"  {'BH の q < 0.05':<26} {nq:3d}   （多重性を通した後）")

    print("\nブロック内相関（較正表のどの行で読むか）")
    rr = sorted(((v, s) for s, v in out["rho"].items() if np.isfinite(v)),
                reverse=True)
    print("  高い: " + ", ".join(f"{s}={v:.2f}" for v, s in rr[:4]))
    print("  低い: " + ", ".join(f"{s}={v:.2f}" for v, s in rr[-4:]))

    print("\n統計量ベクトル全体（energy 検定）")
    for g in gen_names:
        j = out["joint_energy"][g]
        mark = "*" if j["pvalue"] < ALPHA else " "
        print(f"  {g:16s} E = {j['energy']:8.2f}  p = {j['pvalue']:.4f}{mark}")

    print("\n参照対照 —— **現実どうしでもこれだけ違う。天井ではない。**")
    for cname, row in out["reference_contrasts"].items():
        big = sorted(((v, s) for s, v in row["ks"].items() if np.isfinite(v)),
                     reverse=True)[:3]
        print(f"  {cname:14s} ({row['n_a']:3d} vs {row['n_b']:3d}) {row['desc']}")
        print(f"      {row['design']}／割り当て {row['n_assignments']:,} 通り"
              f"／p の刻み {row['p_resolution']:.3f}")
        print("      最大: " + ", ".join(f"{s} {v:.2f}" for v, s in big))
    print("\n  **対応のある対照は 2^K 通りしか割り当てが無い。**独立な暦が6つなら")
    print("  64 通りで、p の刻みは 0.016。手法の欠陥ではなく、独立情報の量が")
    print("  そのまま表に出ている。")

    print(f"\n判定線 p < {ALPHA} で**全生成器**を落とせた統計量：")
    any_hit = False
    for s in stats_names:
        ps = [out["p"][s][g] for g in gen_names if np.isfinite(out["p"][s][g])]
        if ps and max(ps) < ALPHA:
            print(f"  {s:<28} 最悪 p = {max(ps):.4f}")
            any_hit = True
    if not any_hit:
        print("  なし。")
    print("\n（この「全生成器を落とせた」は交差-合併検定なので、多重性の補正は")
    print("  要らない。全部が有意であることを要求する側は、もともと水準を超えない。")
    print("  補正が要るのは「どこかに有意差がある」を探す側で、そちらは BH の q。）")


if __name__ == "__main__":
    main()
