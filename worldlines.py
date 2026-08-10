"""世界線図 — 実指数6本と、無較正のABM各100本の累積リターン。

v5 の較正済み世界線図（累積の図を人間が見て3つの欠陥を見つけたもの）の後継。
今回は**較正を一切しない**。各モデルを論文の公表パラメータ（無いものは実装既定値）
のまま走らせ、素の姿を並べる。較正していないので実データを重ねる意味はなく、
実データは1枚目に6指数をまとめて描くだけにする。

パネル構成:
    1枚目      6指数の直近1000営業日（各窓の標準偏差で標準化した累積対数リターン）
    2〜6枚目   ABM 100本（標準化・累積）。無較正。

モデルの出どころは2系統ある:
    models/            論文から起こし直した再実装（FW は Table 2 検証済み、CI は未完成）
    financial-abm-lab  旧実装。LM・CB は論文と突き合わせて忠実と確認済み、
                       ZI は Gode-Sunder 型の自作ヌルモデル（原論文の推定値は無い）

financial-abm-lab の場所は環境変数 ABM_LAB で指定する（既定は
/workspace/yuitokyouni/financial-abm-lab）。見つからなければ lab 系の
3モデルは飛ばして、その旨をパネルに書く。

    python3 fetch.py        # 先にデータを取得しておく
    python3 worldlines.py   # → worldlines.png（10分ほど。LM が支配的）
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAB = os.environ.get("ABM_LAB", "/workspace/yuitokyouni/financial-abm-lab")
for p in glob.glob(os.path.join(LAB, "packages", "*")):
    sys.path.insert(0, p)

from facts import evaluate  # noqa: E402

WINDOW = 1000
N_PATHS = 100
SEEDS = range(1000, 1000 + N_PATHS)

INDICES = {"gspc": "S&P500", "n225": "日経225", "ftse": "FTSE100",
           "gdaxi": "DAX", "hsi": "ハンセン", "sx5e": "EURO STOXX 50"}


def load_index(name):
    d = json.load(open(os.path.join(HERE, "data", f"{name}.json")))
    d = d["chart"]["result"][0]
    c = np.array([x for x in d["indicators"]["quote"][0]["close"]
                  if x is not None], dtype=float)
    r = np.diff(np.log(c))
    r = r[np.isfinite(r)][-WINDOW:]
    return r / r.std()


# ---------------------------------------------------------------- モデル
# 各エントリ: (タイトル, 出どころの注記, seed -> リターン系列)

def model_specs():
    specs = []

    from models.franke_westerhoff import FrankeWesterhoffTPA
    specs.append(("Franke-Westerhoff", "再実装・論文 Table 1 の推定値",
                  lambda s: FrankeWesterhoffTPA(n_steps=WINDOW, burn_in=1000)
                  .run(seed=s)["returns"]))

    from models.chiarella_iori import ChiarellaIoriPerello
    specs.append(("Chiarella-Iori", "再実装・未完成・論文パラメータ",
                  lambda s: ChiarellaIoriPerello(n_steps=WINDOW, warmup=1000)
                  .run(seed=s)["returns"]))

    try:
        from abm_models.lux_marchesi import LuxMarchesi, Params
        from abm_models.cont_bouchaud import ContBouchaud
        from abm_models.zero_intelligence import ZeroIntelligence
    except ImportError:
        specs.append(("Lux-Marchesi / Cont-Bouchaud / Zero Intelligence",
                      f"ABM_LAB が見つからない（{LAB}）", None))
        return specs

    specs.append(("Lux-Marchesi", "lab 実装（忠実と確認済み）・論文 Set I",
                  lambda s: LuxMarchesi(n_integer_steps=int(WINDOW * 1.25),
                                        steps_per_unit=100, n_c_init=50,
                                        params=Params()).run(seed=s)["returns"]))
    specs.append(("Cont-Bouchaud", "lab 実装（忠実と確認済み）・実装既定値",
                  lambda s: ContBouchaud(N=10000, c=0.9, a=0.01, lam=1.0,
                                         T=int(WINDOW * 1.25),
                                         report_every=10**9).run(seed=s)["returns"]))
    specs.append(("Zero Intelligence", "lab 実装・自作ヌルモデル・既定値（原論文なし）",
                  lambda s: ZeroIntelligence(n_agents=100, n_steps=int(WINDOW * 1.25),
                                             noise_scale=0.01, price_impact=0.01,
                                             tick_size=0.01).run(seed=s)["returns"]))
    return specs


def paths_of(build):
    """有効な（有限・分散非ゼロの）パスだけ標準化して返す。妥当性フィルタは掛けない。"""
    out, rows = [], []
    for s in SEEDS:
        try:
            r = np.asarray(build(s), dtype=float)
        except Exception:
            continue
        r = r[np.isfinite(r)][-WINDOW:]
        if r.size < WINDOW or r.std() == 0:
            continue
        rn = r / r.std()
        out.append(rn)
        rows.append(evaluate(rn))
    med = {k: float(np.nanmedian([x[k] for x in rows])) for k in rows[0]} if rows else {}
    return np.array(out), med


def annotate(med, n):
    a = (f"尖度 {med['excess_kurtosis']:.1f} / acf|r| {med['acf_abs_1']:.3f}"
         f" / lev {med['leverage']:+.3f}\n"
         f"VR20 {med['variance_ratio_20']:.2f} / drift {med['drift']:+.4f}")
    if n < N_PATHS:
        a += f"  （有効 {n}/{N_PATHS} 本）"
    return a


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["IPAGothic", "Noto Sans CJK JP", "DejaVu Sans"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    # -------- 1枚目: 実指数6本
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for i, (name, label) in enumerate(INDICES.items()):
        ax.plot(load_index(name).cumsum(), lw=1.3, color=cmap(i), label=label)
    ax.set_title("実指数6本 直近1000営業日", fontsize=13)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.6)
    ax.axhline(0, color="#999", lw=0.5, zorder=0)

    # -------- 2〜6枚目: ABM 100本
    for ax, (title, note, build) in zip(axes[1:], model_specs()):
        if build is None:
            ax.text(0.5, 0.5, note, transform=ax.transAxes,
                    ha="center", fontsize=11)
            ax.set_title(title, fontsize=13)
            continue
        print(f"{title} …", flush=True)
        paths, med = paths_of(build)
        c = paths.cumsum(axis=1)
        for row in c:
            ax.plot(row, color="#3d6a96", alpha=0.06, lw=0.7)
        ax.plot(np.median(c, axis=0), color="#1b3a5c", lw=1.6)
        ax.set_title(f"{title}（{len(paths)}本）", fontsize=13)
        ax.text(0.5, 1.0, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="#666")
        ax.text(0.02, 0.03, annotate(med, len(paths)), transform=ax.transAxes,
                fontsize=9.5, color="#3d6a96")
        ax.axhline(0, color="#999", lw=0.5, zorder=0)

    for ax in axes[3:]:
        ax.set_xlabel("営業日")
    for ax in axes[::3]:
        ax.set_ylabel("累積リターン（標準偏差単位）")
    fig.suptitle("無較正の世界線 — 論文の公表パラメータのまま各100本（標準化・累積対数リターン）",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "worldlines.png")
    fig.savefig(out, dpi=110)
    print("→", out)


if __name__ == "__main__":
    main()
