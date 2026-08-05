"""実装の状態を宣言し、**実行可能な不変量は実際に走らせて**確かめる。

    python3 models/status.py

## なぜ分けるのか

Franke-Westerhoff と Chiarella-Iori を原論文から起こし直したとき、
既存実装が論文のモデルと別物だったことが分かった。そのとき効いたのは
「パラメータ名が一致していること」ではなく、**論文の式と突き合わせたこと**である。

一方 Franke-Westerhoff は、論文 Table 2 の報告値8つを再現しているのに、
実データと比べると drift が 9.5 倍あって `separation.py` で落ちる。

この2つは別の問いである。混ぜると「検証済み」という言葉が何も意味しなくなる。
シミュレーションの V&V（NIST や ASME V&V 10/20 の系統）でも、
software verification（式を正しく解けているか）と
model validation（現実と合うか）は分けて扱われる。ここでもそうする。

## 語彙

specification_conformance —— 原論文の式・パラメータどおりに実装されているか
    verified   … 全ての式と表を突き合わせ済み
    partial    … 論文に規定の無い箇所があり、こちらで補った
    unverified … 突き合わせていない

invariant_tests —— 走らせて確かめられる保存則・恒等式が成立するか
    pass / fail / not_applicable

paper_replication —— 原論文が報告した数値を再現するか
    pass / partial / fail

empirical_validation —— 実データと区別がつかないか（`separation.py`）
    pass / partial / fail / not_run

**上3つは verification、最後だけが validation である。**
そして paper_replication が fail のモデルで empirical_validation を測っても
意味が無い（何を測っているのか分からない）ので not_run にしてある。
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import chiarella_iori as ci            # noqa: E402
import franke_westerhoff as fw         # noqa: E402

MODELS = [fw.IMPLEMENTATION_STATUS, ci.IMPLEMENTATION_STATUS]

FIELDS = ["specification_conformance", "invariant_tests",
          "paper_replication", "empirical_validation"]


def run_invariants():
    """宣言ではなく実測。返り値 {モデル: [(検査名, 通ったか, 実測値)]}。"""
    out = {}

    r = fw.FrankeWesterhoffTPA(n_steps=20000, burn_in=2000).run(seed=0)["returns"]
    out["franke_westerhoff_tpa"] = [
        ("リターンが有限", bool(np.all(np.isfinite(r))), f"n={len(r)}"),
        ("価格が発散しない", bool(np.max(np.abs(r)) < 1e3),
         f"max|r|={np.max(np.abs(r)):.2f}"),
        ("ボラティリティが縮退しない", bool(np.std(r) > 1e-6),
         f"std={np.std(r):.3f}"),
    ]

    res = ci.ChiarellaIoriPerello(n_steps=3000, warmup=500).run(seed=0)
    c = res["conservation"]
    out["chiarella_iori_perello"] = [
        ("株式の保存", c["shares_rel"] < 1e-9, f"相対誤差 {c['shares_rel']:.2e}"),
        ("現金の保存", c["cash_rel"] < 1e-9, f"相対誤差 {c['cash_rel']:.2e}"),
        ("約定が起きている", res["n_trades"] > 0, f"{res['n_trades']} 約定"),
        ("価格が正", bool(np.all(res["prices"] > 0)),
         f"min={np.min(res['prices']):.3f}"),
    ]
    return out


def main():
    print("実装の状態（宣言）\n")
    w = max(len(f) for f in FIELDS) + 2
    hdr = "".ljust(28) + "".join(m["model"][:22].rjust(26) for m in MODELS)
    print(hdr)
    print("-" * len(hdr))
    for f in FIELDS:
        print(f.ljust(28) + "".join(m[f].rjust(26) for m in MODELS))
    print("-" * len(hdr))
    print("\n上3行が verification（論文どおりか）、"
          "最終行だけが validation（現実と合うか）。")

    print("\n\n不変量の実測（走らせて確かめる）")
    res = run_invariants()
    ok = True
    for model, checks in res.items():
        print(f"\n  {model}")
        for name, passed, detail in checks:
            print(f"    {'通過' if passed else '失敗'}  {name:<24} {detail}")
            ok = ok and passed

    for m in MODELS:
        got = res.get(m["model"])
        if got is None:
            continue
        expect = "pass" if all(p for _, p, _ in got) else "fail"
        if m["invariant_tests"] != expect:
            print(f"\nNG: {m['model']} の宣言 invariant_tests="
                  f"{m['invariant_tests']} が実測 {expect} と食い違う")
            ok = False

    print("\n\n注記")
    for m in MODELS:
        print(f"\n  {m['model']}  ({m['source']})")
        print(f"    {m['notes']}")

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
