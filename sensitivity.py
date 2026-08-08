"""実装の自由度に、答えがどれだけ引きずられるか。

    python3 sensitivity.py     # → sensitivity.json（1分ほど）

## なぜ要るか

統計量には「窓幅」「ラグ数」「閾値」「地平」といった、**現象ではなく実装の都合で
決めた数**がある。これを振ったときに答えが動くなら、報告している値は現象ではなく
その選択を測っている。

v0.3 で `gain_loss_asymmetry` がまさにそうだった。地平 `horizon` を 250 → 2000 と
延ばすだけで、実データの値が **+0.398 → +0.043** に落ちた。打ち切られる起点の割合が
55% → 14% に減るからで、**現象ではなく捨て方が値を作っていた。**
誰も `horizon` を振っていなかったので、気づくのに v0.3 までかかった。

`robustness.py` には Hill の `k` を振る検査が前からあった。**あれを1つの統計量の
ための特別扱いにせず、`facts.SPEC` に宣言された全ての自由度に自動で掛ける。**

## 読み方

「振り幅」は、パラメータを動かしたときの中央値の変動を、**窓どうしのばらつき
（四分位幅）で割ったもの**である。

    0.1 未満  … 実装の都合はほぼ効いていない
    0.5 以上  … データを変えるのと同じくらい、つまみが効いている（警告）
    1.0 以上  … つまみの方が支配的（バグを疑う）

つまり **「その数字は現象か、それとも君の選択か」**を1つの数にしている。
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from facts import BATTERY, SPEC                    # noqa: E402
from windows import load_series, real_windows      # noqa: E402

WARN = 0.5
FAIL = 1.0


def main():
    series = load_series()
    wins = [w.values for w in real_windows(series)]
    print(f"実データの窓 {len(wins)} 本で、実装の自由度を振る\n")

    out = {"config": {"n_windows": len(wins), "warn": WARN, "fail": FAIL},
           "swing": {}}
    rows, worst = [], []

    for s, fn in BATTERY.items():
        params = SPEC.get(s, {}).get("params", {})
        base = np.array([fn(w) for w in wins], dtype=float)
        iqr = float(np.nanpercentile(base, 75) - np.nanpercentile(base, 25))
        if not params:
            continue
        out["swing"][s] = {}
        for pname, vals in params.items():
            meds = []
            for v in vals:
                x = np.array([fn(w, **{pname: v}) for w in wins], dtype=float)
                meds.append(float(np.nanmedian(x)))
            swing = (max(meds) - min(meds)) / iqr if iqr > 0 else float("inf")
            out["swing"][s][pname] = {"values": list(vals), "medians": meds,
                                      "swing_over_iqr": swing}
            flag = "失敗" if swing >= FAIL else ("警告" if swing >= WARN else "")
            rows.append((s, pname, vals, meds, swing, flag))
            if flag:
                worst.append((s, pname, swing, flag))

    w0 = max(len(r[0]) for r in rows) + 1
    print(f"{'統計量'.ljust(w0)}{'自由度':<10}{'振り幅/四分位幅':>16}   中央値の動き")
    print("-" * (w0 + 60))
    for s, pname, vals, meds, swing, flag in rows:
        mv = " → ".join(f"{m:+.3f}" for m in meds)
        mark = f"  {flag}" if flag else ""
        print(f"{s.ljust(w0)}{pname:<10}{swing:>16.3f}   {mv}{mark}")
    print("-" * (w0 + 60))

    json.dump(out, open(os.path.join(HERE, "sensitivity.json"), "w"),
              ensure_ascii=False, indent=1)

    if worst:
        print("\n実装の都合が効きすぎている：")
        for s, pname, swing, flag in sorted(worst, key=lambda x: -x[2]):
            print(f"  {flag}  {s} の {pname}（振り幅 {swing:.2f} × 四分位幅）")
        print("\n**その統計量の値は、現象ではなくその選択を測っている疑いがある。**")
    else:
        print("\n全ての自由度が振り幅 0.5 未満。実装の都合は効いていない。")

    return 1 if any(f == "失敗" for *_, f in worst) else 0


if __name__ == "__main__":
    sys.exit(main())
