"""Kronos 駆動の自己組織化板から、sieve-bench に通せるリターン系列を作る。

YH007-8 の P3-F で substrate 合格が確認されている構成をそのまま使う。
1パスごとに .npy へ保存し、時間予算を使い切ったら終了する。
バックグラウンドが刈られても、再実行すれば続きから走る。

    python3 run_kronos.py --budget 540 --n-paths 30

基盤モデル（Kronos-small, 24.7M）が全エージェントの評価値を bar 単位で共有する。
YH007 が候補 finding として保存した「同一基盤モデル共有 → directional 同期 →
集合 over-reaction」が、識別力としてどう出るかを見るのが目的。
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kronos_paths")
os.makedirs(OUT, exist_ok=True)

WINDOW = 1000
BAR = 10
MAIN_STEPS = (WINDOW + 2) * BAR   # バー数 ≒ main_steps / bar_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=540.0, help="秒。使い切ったら終了")
    ap.add_argument("--n-paths", type=int, default=30)
    ap.add_argument("--n-kronos", type=int, default=8)
    ap.add_argument("--n-zi", type=int, default=50)
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()

    from abm_models.self_organized_book import SelfOrganizedBookMarket

    t0 = time.time()
    done = 0
    for i in range(a.n_paths):
        fp = os.path.join(OUT, f"path_{i:03d}.npy")
        if os.path.exists(fp):
            done += 1
            continue
        if time.time() - t0 > a.budget:
            break
        t1 = time.time()
        m = SelfOrganizedBookMarket(
            warmup_steps=200, main_steps=MAIN_STEPS, n_zi=a.n_zi,
            n_kronos=a.n_kronos, kronos_n_samples=a.samples,
            kronos_eval_mode="chase", bar_size=BAR)
        r = m.run(seed=1000 + i)
        x = np.asarray(r["returns_main_market"], dtype=float)
        x = x[np.isfinite(x)]
        if x.size >= WINDOW and x.std() > 0:
            np.save(fp, x[-WINDOW:].astype(np.float32))
            done += 1
            print(f"  path {i:03d}  n={x.size}  std={x.std():.6f}  "
                  f"{time.time()-t1:.0f}s", flush=True)
        else:
            print(f"  path {i:03d}  棄却 (n={x.size}, std={x.std():.2e})", flush=True)

    have = len([f for f in os.listdir(OUT) if f.endswith(".npy")])
    print(f"保存済み {have}/{a.n_paths} 本  経過 {time.time()-t0:.0f}s")
    if have >= a.n_paths:
        print("完了")


if __name__ == "__main__":
    main()
