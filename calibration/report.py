"""較正結果の図表。

figure: 各モデルにつき1枚、100本の世界線（累積対数リターン）を重ね、
        目標である S&P500 直近1000営業日を上に重ねる。
table:  合わせ込んだ3つと、合わせ込んでいない10個を分けて出す。
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "calibration.json")))
FITTED = C["config"]["fitted"]
real = np.load(os.path.join(HERE, "target_window.npy"))
real = real / real.std()

JP = [f for f in ("Noto Sans CJK JP", "Noto Sans JP", "IPAGothic", "TakaoGothic",
                  "DejaVu Sans") if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams["font.family"] = JP[0] if JP else "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LABEL = {"chiarella_iori": "Chiarella-Iori", "franke_westerhoff": "Franke-Westerhoff",
         "zero_intelligence": "Zero Intelligence", "lux_marchesi": "Lux-Marchesi",
         "cont_bouchaud": "Cont-Bouchaud"}

ok = [m for m in C["models"] if C["models"][m].get("status") == "ok"
      and os.path.exists(os.path.join(HERE, f"paths_{m}.npy"))]

# ---------------------------------------------------------------- 世界線
n = len(ok) + 1
cols = 3
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows),
                         sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()

real_cum = np.cumsum(real)
lim = 0

panels = [("__real__", real)] + [(m, None) for m in ok]
for ax, (name, _) in zip(axes, panels):
    if name == "__real__":
        ax.plot(real_cum, color="#B83A24", lw=1.6, zorder=3)
        ax.set_title("S&P500  直近1000営業日（目標）", fontsize=11)
        lim = max(lim, np.abs(real_cum).max())
        continue
    P = np.load(os.path.join(HERE, f"paths_{name}.npy"))
    cum = np.cumsum(P, axis=1)
    for row in cum:
        ax.plot(row, color="#27547F", lw=0.45, alpha=0.16, zorder=1)
    ax.plot(np.median(cum, axis=0), color="#16324F", lw=1.3, zorder=2)
    ax.plot(real_cum, color="#B83A24", lw=1.4, zorder=3)
    st = C["models"][name]["stats"]
    ax.set_title(f"{LABEL.get(name,name)}   ({cum.shape[0]}本)", fontsize=11)
    ax.text(0.02, 0.03,
            f"尖度 {st['excess_kurtosis']['median']:.1f}"
            f" / acf|r| {st['acf_abs_1']['median']:.3f}"
            f" / lev {st['leverage']['median']:+.3f}",
            transform=ax.transAxes, fontsize=8, color="#44607A")
    lim = max(lim, np.abs(np.percentile(cum, [1, 99])).max())

for ax in axes[len(panels):]:
    ax.axis("off")
for ax in axes[:len(panels)]:
    ax.axhline(0, color="#B9C7D4", lw=0.6, zorder=0)
    ax.set_ylim(-lim * 1.05, lim * 1.05)
    ax.grid(alpha=0.15, lw=0.5)

fig.suptitle("較正後の100本の世界線（累積対数リターン・標準化）— 赤が S&P500",
             fontsize=13, y=0.995)
fig.supxlabel("営業日", fontsize=10)
fig.supylabel("累積リターン（標準偏差単位）", fontsize=10)
fig.tight_layout(rect=[0.012, 0.012, 1, 0.985])
fig.savefig(os.path.join(HERE, "worldlines.png"), dpi=145)
print("→ worldlines.png")

# ---------------------------------------------------------------- 表
lines = []
w = max(len(LABEL.get(m, m)) for m in ok)


def block(keys, title):
    lines.append(f"\n{title}")
    lines.append("統計量".ljust(26) + "S&P500".rjust(9) +
                 "".join(LABEL.get(m, m)[:13].rjust(16) for m in ok))
    lines.append("-" * (35 + 16 * len(ok)))
    for k in keys:
        row = k.ljust(26) + f"{C['target'][k]:+9.3f}"
        for m in ok:
            s = C["models"][m]["stats"][k]
            row += f"{s['median']:+8.3f}[{s['z']:+5.1f}]".rjust(16)
        lines.append(row)


BAT = list(C["target"].keys())
block(FITTED, "■ 合わせ込んだ量（3つ）  ※括弧内は実データ窓のばらつきで測った z")
block([k for k in BAT if k not in FITTED], "■ 合わせ込んでいない量（10個）— 判定はこちらでやる")

lines.append("\n■ 較正の内訳")
for m in ok:
    d = C["models"][m]
    lines.append(f"{LABEL.get(m,m):<20} loss={d['loss']:7.3f}  "
                 f"{d['n_eval']:>5}回評価  {d['secs']:6.0f}s  パス{d['n_paths']}本")
    lines.append("   " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in sorted(d["params"].items())))

txt = "\n".join(lines)
open(os.path.join(HERE, "report.txt"), "w").write(txt)
print(txt)
