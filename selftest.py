"""検定そのものを検定する。

このリポジトリの主張は「どの統計量が生成器を見破れるか」だが、
**その判定に使う検定が正しい大きさ（size）を持っているか**は別の問題である。
ここでは正解が分かっている人工データで第一種過誤を直接測る。

実データ側と同じ形の依存（ブロック内の共通成分、6ブロック・サイズ 22/23/24/24/24/7）を
持つ標本を、**同じ周辺分布**から作った独立な 200 本と比べる。帰無仮説は真なので
5% 水準の棄却率は 5% であるべき。

測って分かったこと（下の表が実測値）:

  1. **旧版の完全置換は反保守的。**ブロック内相関 0.3 で 35%、0.6 で 59% 棄却する。
     名目 5% に対して一桁近い。`separation.py` v0.1 の p 値はこれで出ていた。
  2. **ブロック置換・ブロック復元抽出は大きく改善するが、名目には届かない。**
     どちらも 10〜18%。原因は手法ではなく**ブロックが6個しかないこと**で、
     クラスタ数が 20 を切ると過剰棄却するのは計量経済で広く知られている
     （Cameron, Gelbach & Miller 2008）。25年・6指数・4年窓というデータの
     形が、そもそも 5% 水準の主張を支えるだけの独立な暦を持っていない。
  3. **だから閾値を較正して使う。**名目 p < 0.01 で真の大きさが 4〜5% に収まる
     （ブロック内相関 0.2〜0.6 のいずれでも）。`separation.py` はこの
     **p < 0.01 を 5% 水準の判定線**として使う。

実行:  python3 selftest.py     （5分ほど）
"""

import sys

import numpy as np

from resampling import (benjamini_hochberg, block_boot_test, block_perm_test,
                        energy_block_test, iid_perm_test, ks_stat)

N_REP = 300
N_PERM = 300
N_REP_CAL = 500
N_PERM_CAL = 400

# 実データの構成に合わせる：124本 = 6ブロック、生成器 200 本
BLOCK_SIZES = [22, 23, 24, 24, 24, 7]
N_B = 200

ALPHAS = (0.05, 0.02, 0.01, 0.005)

# `separation.py` が使う判定線。下の較正表から決めてある。
CALIBRATED_ALPHA = 0.01


def clustered(rng, rho):
    """ブロック内で相関 rho を持つ標本。周辺分布は N(0,1) のまま。"""
    vals, labels = [], []
    for b, s in enumerate(BLOCK_SIZES):
        mu = rng.normal(0.0, np.sqrt(rho))
        vals.append(mu + rng.normal(0.0, np.sqrt(1.0 - rho), s))
        labels.append(np.full(s, b))
    return np.concatenate(vals), np.concatenate(labels)


def pvals(rho, rng, which, n_rep, n_perm, shift=0.0, scale=1.0):
    """指定した帰無分布の作り方で p 値を n_rep 個集める。"""
    out = []
    for _ in range(n_rep):
        a, lab = clustered(rng, rho)
        b = shift + scale * rng.normal(0.0, 1.0, N_B)
        if which == "iid":
            out.append(iid_perm_test(a, b, ks_stat, rng, n_perm)[1])
        elif which == "perm":
            out.append(block_perm_test(a, lab, b, ks_stat, rng, n_perm)[1])
        else:
            out.append(block_boot_test(a, lab, b, ks_stat, rng, n_perm)[1])
    return np.array(out)


def main():
    rng = np.random.default_rng(7)
    bad = []

    print("=== 1. 帰無が真のときの棄却率（名目 5%）===")
    print("実データ側だけがブロック内で相関し、生成器側は独立。周辺分布は同一。\n")
    print(f"{'ブロック内相関':<16}{'完全置換(旧)':>14}{'ブロック置換':>16}"
          f"{'ブロック復元抽出':>18}")
    print("-" * 66)
    for rho in (0.0, 0.3, 0.6):
        r = [np.mean(pvals(rho, rng, w, N_REP, N_PERM) < 0.05)
             for w in ("iid", "perm", "boot")]
        print(f"rho = {rho:<11.1f}{r[0]:>13.1%}{r[1]:>15.1%}{r[2]:>17.1%}")
        if rho >= 0.3:
            if r[0] < 0.10:
                bad.append(f"rho={rho} で旧版の膨張が再現しない（{r[0]:.1%}）")
            if r[2] > r[0]:
                bad.append(f"rho={rho} で新版が旧版より悪い")
    print("\n→ 完全置換は名目を一桁近く超える。**v0.1 の p 値はこれだった。**")
    print("  ブロック法は大きく改善するが名目には届かない。手法の限界ではなく、")
    print("  **ブロックが6個しかない**ことによる（クラスタ数 < 20 の過剰棄却）。")

    print(f"\n=== 2. 閾値の較正（ブロック復元抽出、{N_REP_CAL}反復）===")
    print("名目 alpha を下げたとき、真の大きさがどこで 5% に収まるか。\n")
    print(f"{'ブロック内相関':<16}" + "".join(f"alpha={a}".rjust(13) for a in ALPHAS))
    print("-" * (16 + 13 * len(ALPHAS)))
    at_cal = []
    for rho in (0.2, 0.4, 0.6):
        p = pvals(rho, rng, "boot", N_REP_CAL, N_PERM_CAL)
        sizes = [np.mean(p < a) for a in ALPHAS]
        at_cal.append(sizes[ALPHAS.index(CALIBRATED_ALPHA)])
        print(f"rho = {rho:<11.1f}" + "".join(f"{s:.1%}".rjust(13) for s in sizes))
    print(f"\n→ 名目 p < {CALIBRATED_ALPHA} で真の大きさは "
          f"{min(at_cal):.1%}〜{max(at_cal):.1%}。**これを 5% 水準の判定線に使う。**")
    if max(at_cal) > 0.08:
        bad.append(f"較正した閾値でも大きさが {max(at_cal):.1%} ある")

    print(f"\n=== 3. 検出力（判定線 p < {CALIBRATED_ALPHA} のもとで）===")
    print(f"{'相手':<26}{'完全置換(旧)':>14}{'ブロック復元抽出':>18}")
    print("-" * 60)
    for label, kw in (("平均 +0.5", {"shift": 0.5}),
                      ("標準偏差 ×1.5", {"scale": 1.5})):
        p_i = pvals(0.3, rng, "iid", N_REP, N_PERM, **kw)
        p_b = pvals(0.3, rng, "boot", N_REP, N_PERM, **kw)
        r_i = np.mean(p_i < CALIBRATED_ALPHA)
        r_b = np.mean(p_b < CALIBRATED_ALPHA)
        print(f"{label:<26}{r_i:>13.1%}{r_b:>17.1%}")
        if r_b <= max(at_cal):
            bad.append(f"{label} の検出力が帰無時の棄却率を超えない（{r_b:.1%}）")
    print("\n→ **ここは弱点として読むべき数字である。**独立な暦ブロックが6個しか")
    print("  ないので、標準偏差が1.5倍ずれている相手すら較正済みの判定線では")
    print("  1割しか落とせない。**分離が出ないことは統計量が鈍い証拠にならない。**")
    print("  実データの表で p が小さく出ている行は、それだけ大きな差だということ。")

    print("\n=== 4a. energy 検定（統計量ベクトル全体）===")
    print("列ごとの周辺分布は同一で、列どうしの相関だけが違う相手を作る。\n")
    n_rep_e = 150
    p_null, p_alt = [], []
    for _ in range(n_rep_e):
        a2 = np.column_stack([clustered(rng, 0.4)[0] for _ in range(4)])
        lab = clustered(rng, 0.4)[1]
        p_null.append(energy_block_test(a2, lab, rng.normal(0, 1, (N_B, 4)),
                                        rng, 300)[1])
        z = rng.normal(0, 1, (N_B, 4))
        dep = np.column_stack([z[:, 0], z[:, 0] * 0.9 + z[:, 1] * 0.44,
                               z[:, 2], z[:, 2] * 0.9 + z[:, 3] * 0.44])
        p_alt.append(energy_block_test(a2, lab, dep, rng, 300)[1])
    p_null, p_alt = np.array(p_null), np.array(p_alt)
    print(f"{'':<22}{'alpha=0.05':>13}{'alpha=0.01':>13}")
    print("-" * 48)
    print(f"{'同分布（大きさ）':<22}{np.mean(p_null < 0.05):>12.1%}"
          f"{np.mean(p_null < 0.01):>13.1%}")
    print(f"{'相関だけ違う（検出力）':<22}{np.mean(p_alt < 0.05):>12.1%}"
          f"{np.mean(p_alt < 0.01):>13.1%}")
    if np.mean(p_null < 0.01) > 0.08:
        bad.append(f"energy 検定の大きさが過大（{np.mean(p_null < 0.01):.1%}）")
    if np.mean(p_alt < 0.05) <= np.mean(p_null < 0.05):
        bad.append("energy 検定が同時分布の違いに反応しない")
    print("\n→ 列ごとの周辺分布を個別に見れば差が出ない相手に反応する。"
          "\n  ただし検出力はここでも高くない（同じ6ブロックの制約）。")

    print("\n=== 4b. Benjamini-Hochberg ===")
    q = benjamini_hochberg([0.001, 0.008, 0.04, 0.2, 0.7, np.nan])
    print("  p = [0.001, 0.008, 0.04, 0.2, 0.7, nan]")
    print("  q =", np.round(q, 4).tolist())
    fin = q[np.isfinite(q)]
    if not np.all(np.diff(fin) >= -1e-12):
        bad.append("BH が単調でない")
    if not np.all(fin >= np.array([0.001, 0.008, 0.04, 0.2, 0.7]) - 1e-12):
        bad.append("BH の q が p を下回った")

    print()
    if bad:
        for m in bad:
            print("NG:", m)
        return 1
    print("全て通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
