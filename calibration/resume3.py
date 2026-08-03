"""v3 の続き — lux_marchesi と cont_bouchaud だけ走らせて結果を統合する。

前回は LM の局所精錬中にプロセスが落ち、CI/FW/ZI までしか保存されていない。
探索段階では LM は loss 1.129（既定値 35.523）まで来ていたので、
同じ Sobol 列を使えば同じ点に到達する。精錬の巡回数を 3 → 2 に減らし、
確実に完走させる。
"""

import json
import os
import time

import numpy as np

import calibrate as C
import calibrate3 as V3

HERE = os.path.dirname(os.path.abspath(__file__))
V3.BUDGET["lux_marchesi"] = (300, 3)
V3.BUDGET["cont_bouchaud"] = (280, 3)


def main():
    t0 = time.time()
    lf = open(os.path.join(HERE, "resume3.log"), "w")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    out = json.load(open(os.path.join(HERE, "calibration_v3.json")))
    done = [k for k, v in out["models"].items() if v.get("status") == "ok"]
    log(f"保存済み: {done}")
    target, scale = out["target"], out["scale"]

    for name in ("lux_marchesi", "cont_bouchaud"):
        log(f"\n=== {name} ===")
        r = V3.run_model(name, target, scale, log)
        # frontier は巨大なので保存前に間引く
        if r.get("frontier"):
            fr = sorted(r["frontier"], key=lambda d: d["loss"])[:400]
            r["frontier"] = fr
        out["models"][name] = r
        json.dump(out, open(os.path.join(HERE, "calibration_v3.json"), "w"),
                  ensure_ascii=False, indent=1)
        log(f"  [{name}] 保存した")

    log(f"\n完了 {time.time()-t0:.0f}s")
    lf.close()


if __name__ == "__main__":
    main()
