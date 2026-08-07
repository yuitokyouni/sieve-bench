"""不変性の監査 —— 走らせる前に、定義から「絶対に見えないもの」を出す。

    python3 invariance.py     # → invariance.json（10秒）

## なぜ第一層がこれなのか

統計量は系列を1つの数に落とす写像 T_j: X → R である。**この瞬間に必ず情報が捨てられる。**
重要なのは、何が捨てられたかを実験の後に推測することではなく、**定義から調べること**である。

変換 g に対して

    T_j(gx) = T_j(x)   （すべての x について）

が成り立つなら、T_j は g に対して不変である。バッテリー全体 B = {T_1..T_k} の
共通不変変換は

    G_B = ∩_j G_j,   G_j = {g : T_j(gx) = T_j(x) ∀x}

で、g ∈ G_B なら x と gx はバッテリーから見て**同一**である。

**これは「検出力が低い」とは質的に違う。**共通不変変換に入っているなら、
標本をいくら増やしても、検定をいくら改善しても、Monte Carlo をいくら回しても
差は出ない。**検出力が 0 になるようにバッテリーを定義してしまっている。**

加法的な変換 x → x + h だけを考えるなら、見えない h の集合

    N_B = {h : T_j(x+h) = T_j(x) ∀x, ∀j}

は線形部分空間になるので、そこは blind subspace と呼べる。
一般の変換群に対して「空間」と呼ぶのは危ないので、以下では
**盲目な変換 / 不変集合**という語を使う。

## v0.2 の「平均を見落とした」の一般形

一定のドリフト c を足す変換は h = c·1 なので、旧バッテリーに span{1} ⊆ N_B が
成り立っていたなら、ドリフトは**原理的に**見えなかったことになる。
累積リターンで見ると c·1 は Σc = ct、つまり**直線のランプ**として現れる。
だから return 空間では見えず、cumulative 空間の図では人間の目に一発で入った。

**ただしこれは主張であって、確かめるべきものである。**平均を引く6個は定義から
厳密に不変だが、`|r|` を経由する残りは |x+c| ≠ |x|+c なので厳密には不変ではない。
ここでは各セルを**実際に走らせて**分類する：

    不変      … 相対変化が計算機精度（定義から従う。理由を注記した）
    ほぼ不変  … 窓間のばらつきの 5% 未満（原理的には見えるが、この標本では無理）
    反応      … それ以上

**厳密な不変とほぼ不変は、扱いが違う。**前者はデータを増やしても消えない。
後者は標本を増やせば見えるようになりうる。
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from facts import BATTERY                      # noqa: E402
from windows import load_series, real_windows   # noqa: E402

SEED = 20260802
EXACT = 1e-9      # 相対変化がこれ未満なら「定義から不変」とみなす
SMALL = 0.05      # 窓間ばらつきに対してこれ未満なら「ほぼ不変」

# 検証したい性質に対応する変換を**先に宣言する。**網羅が目的ではない。
# 生成器の梯子と対応させてある。
TRANSFORMS = {
    "location":  ("x + c",        "位置／ドリフト。c = 0.15·σ（ABM が出した規模）"),
    "scale":     ("a·x",          "尺度。a = 2.5"),
    "sign":      ("−x",           "符号／非対称性。上げと下げを入れ替える"),
    "reversal":  ("x を逆順に",    "時間の向き。因果の順序だけを反転する"),
    "shuffle":   ("x をランダム並べ替え", "時間の並び。周辺分布は保つ"),
    "tail":      ("sign(x)|x|^1.3", "裾の重さ。単調・符号保存・順序保存"),
}


def apply_transform(x, name, rng):
    s = float(np.std(x))
    if name == "location":
        return x + 0.15 * s
    if name == "scale":
        return 2.5 * x
    if name == "sign":
        return -x
    if name == "reversal":
        return x[::-1].copy()
    if name == "shuffle":
        return rng.permutation(x)
    if name == "tail":
        return np.sign(x) * s * np.abs(x / s) ** 1.3
    raise ValueError(name)


def classify(rel_exact, rel_spread):
    if rel_exact < EXACT:
        return "不変"
    if rel_spread < SMALL:
        return "ほぼ不変"
    return "反応"


def main():
    rng = np.random.default_rng(SEED)
    series = load_series()
    wins = [w.values for w in real_windows(series)]
    print(f"実データの窓 {len(wins)} 本で、各変換の前後を比べる\n")

    stats = list(BATTERY.keys())
    base = {s: np.array([BATTERY[s](w) for w in wins], dtype=float) for s in stats}
    spread = {s: float(np.nanpercentile(base[s], 75) - np.nanpercentile(base[s], 25))
              for s in stats}

    out = {"config": {"seed": SEED, "n_windows": len(wins),
                      "exact_tol": EXACT, "small_frac": SMALL},
           "transforms": {k: {"expr": v[0], "desc": v[1]}
                          for k, v in TRANSFORMS.items()},
           "verdict": {}, "effect": {}, "spread": spread}

    for s in stats:
        out["verdict"][s], out["effect"][s] = {}, {}

    for tname in TRANSFORMS:
        moved = [apply_transform(w, tname, rng) for w in wins]
        for s in stats:
            after = np.array([BATTERY[s](w) for w in moved], dtype=float)
            b = base[s]
            m = np.isfinite(b) & np.isfinite(after)
            if m.sum() < 5:
                out["verdict"][s][tname], out["effect"][s][tname] = "—", None
                continue
            d = np.abs(after[m] - b[m])
            scale = np.maximum(np.abs(b[m]), 1e-12)
            rel_exact = float(np.max(d / scale))
            rel_spread = (float(np.median(d) / spread[s]) if spread[s] > 0
                          else float("inf"))
            out["verdict"][s][tname] = classify(rel_exact, rel_spread)
            out["effect"][s][tname] = rel_spread

    # --------------------------------------------------- バッテリー全体の不変集合
    common = [t for t in TRANSFORMS
              if all(out["verdict"][s][t] == "不変" for s in stats)]
    # ドリフトを見る2つ（v0.2 で足したもの）を除いた「旧バッテリー」でも見る
    legacy = [s for s in stats if s not in ("drift",)]
    common_legacy = [t for t in TRANSFORMS
                     if all(out["verdict"][s][t] == "不変" for s in legacy)]
    out["common_invariance"] = common
    out["common_invariance_without_drift"] = common_legacy

    json.dump(out, open(os.path.join(HERE, "invariance.json"), "w"),
              ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------ 表示
    w0 = max(len(s) for s in stats) + 1
    hdr = "統計量".ljust(w0) + "".join(t.rjust(12) for t in TRANSFORMS)
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        row = s.ljust(w0)
        for t in TRANSFORMS:
            row += out["verdict"][s][t].rjust(12)
        print(row)
    print("-" * len(hdr))

    print("\n「ほぼ不変」の実際の大きさ（窓間ばらつきに対する比。1.0 なら四分位幅と同程度）")
    for s in stats:
        near = [(t, out["effect"][s][t]) for t in TRANSFORMS
                if out["verdict"][s][t] == "ほぼ不変"]
        if near:
            print(f"  {s:<26}" + ", ".join(f"{t} {v:.4f}" for t, v in near))

    print("\n" + "=" * 70)
    print("バッテリー全体が不変な変換（＝どれだけ標本を増やしても見えないもの）")
    print("=" * 70)
    print(f"  現在の16個          : {common or 'なし'}")
    print(f"  drift を抜いた15個  : {common_legacy or 'なし'}")

    print("\n変換ごとに、**厳密に不変な**（＝原理的に見えない）統計量")
    print("-" * 70)
    for t in TRANSFORMS:
        blind = [s for s in stats if out["verdict"][s][t] == "不変"]
        out.setdefault("blind_to", {})[t] = blind
        print(f"  {t:<10} {len(blind):>2}/{len(stats)}  " +
              (", ".join(blind) if blind else "なし"))

    print("\n→ **この列に載っている統計量だけを報告する論文は、その性質を**")
    print("  **原理的に検出できない。**標本を増やしても検定を改善しても変わらない。")

    print("\n→ ただし**厳密な不変と『ほぼ不変』は別物である。**前者はデータを")
    print("  増やしても消えない（検出力が定義上 0）。後者は標本次第で見えうる。")
    print("  実測した反応の大きさ（location, 四分位幅比）:")
    rank = sorted(((out["effect"][s]["location"], s) for s in stats), reverse=True)
    print("    " + ", ".join(f"{s} {v:.2f}" for v, s in rank[:5]))

    literature_closure(out, stats)

    json.dump(out, open(os.path.join(HERE, "invariance.json"), "w"),
              ensure_ascii=False, indent=1)


# 文献監査のカテゴリ → 本ベンチの統計量。volume-volatility-corr と
# regime-switching は出来高・レジームを要するのでバッテリーに無い（除外）。
CATEGORY_TO_STATS = {
    "fat-tails": ["excess_kurtosis", "hill_right", "hill_left"],
    "vol-clustering": ["acf_abs_1", "ljung_box_sq"],
    "long-memory": ["acf_abs_20", "acf_abs_decay"],
    "leverage": ["leverage"],
    "gain-loss-asymmetry": ["gain_loss_asymmetry", "return_skewness"],
    "absence-of-autocorr": ["acf_return_1"],
    "aggregational-gaussianity": ["aggregational_gaussianity"],
}


def literature_closure(out, stats):
    """**実際の論文が報告している組み合わせは、何に対して盲目か。**

    不変性の監査はシミュレーション不要なので、文献にそのまま適用できる。
    各論文が挙げているスタイライズドファクツを統計量に写し、その共通不変集合を取る。
    **そこに入っている性質は、その論文がどれだけデータを増やしても検出できない。**

    監査は「言及」を数えているので「報告」より広い（`audit/README.md`）。
    したがってここでの盲目の判定は**保守側**である ── 実際に報告している
    統計量はより少ないはずなので、盲目な変換はもっと多い可能性が高い。
    """
    path = os.path.join(HERE, "audit", "relabeled.json")
    if not os.path.exists(path):
        return
    papers = [p for p in json.load(open(path))
              if p.get("sf_mention") and p.get("is_sim")]
    if not papers:
        return

    res = {}
    for thresh in ("new_ge1", "new_ge3"):
        rows = []
        for p in papers:
            got = []
            for c in p.get(thresh, []):
                got += CATEGORY_TO_STATS.get(c, [])
            got = sorted(set(got))
            if not got:
                continue
            blind = [t for t in TRANSFORMS
                     if all(out["verdict"][s][t] == "不変" for s in got)]
            rows.append({"id": p["arxiv_id"], "stats": got, "blind": blind})
        n = len(rows)
        per = {t: sum(1 for r in rows if t in r["blind"]) for t in TRANSFORMS}
        res[thresh] = {"n_papers": n, "blind_counts": per,
                       "median_n_stats": float(np.median([len(r["stats"])
                                                          for r in rows]))}
        print(f"\n{'=' * 70}")
        print(f"文献 {n} 本が挙げている組み合わせは、何に対して盲目か"
              f"（しきい値 {thresh}）")
        print("=" * 70)
        for t in TRANSFORMS:
            k = per[t]
            print(f"  {t:<10} {k:>3}/{n}  ({k / n:.0%}) が原理的に検出できない")
        print(f"  （挙げている統計量の数の中央値 {res[thresh]['median_n_stats']:.0f}）")

    out["literature_blindness"] = res
    print("\n→ **これはシミュレーション不要の解析的な主張である。**")
    print("  その変換に対して不変な統計量しか挙げていない論文は、")
    print("  標本を増やしても検定を改善してもその性質を検出できない。")
    print("  監査は『言及』を数えているので、この判定は保守側に寄っている。")


if __name__ == "__main__":
    main()
