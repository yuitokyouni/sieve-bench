"""「言及」を「報告」に格上げする。

    python3 audit/reported.py     # → audit/reported.json（54本の再取得で数分）

## なぜ必要か

`coding.py` は本文中の**言及回数**を数えている。これは
「序論で Cont (2001) を引きながらスタイライズドファクツを列挙した」だけの論文と、
「実際にその統計量を測って図表に載せた」論文を区別しない。

不変性監査（`invariance.py`）はこの列を使って
「その論文は時間の向きを原理的に検出できない」と主張するので、
**言及ベースだと主張が弱い。**「言及していないだけで、実は測っているかもしれない」
という反論に対して何も言えない。

## どう格上げするか

**報告の証拠として2つの、より厳しい signal を取る。**

1. **caption** —— 図表のキャプションに名前が出る。
   LaTeX なら `\\caption{...}`、arXiv HTML なら `ltx_caption` / `figcaption`。
   キャプションに書くのは、ふつう実際に載せた図表についてである。

2. **with_number** —— 言及の前後 120 文字以内に数値がある。
   「leverage effect is well documented」ではなく
   「the leverage correlation is −0.09」の側を拾う。

`reported = caption ∪ with_number` とする。

## この操作化の限界（両方向に外す）

- **見落とす方向**：「Table 3: Simulation results」のように、キャプションが
  統計量名を含まない図表で報告している場合。本文で数値なしに述べている場合
- **拾いすぎる方向**：関連研究で他人の測定値を引用している場合
  （「Cont (2001) reports a tail index of 3.0」）

したがって `mention ⊇ reported` であることは保証されるが、
**`reported` が真の報告集合と一致する保証は無い。**
3つの定義（mention / with_number / caption）を全部出して感度を見る。
"""

import gzip
import json
import os
import re
import subprocess
import sys
import tarfile
import time

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "fulltext")
CACHE = os.path.join(D, "reported.json")

sys.path.insert(0, D)
from coding import CATS, strip_refs, strip_html   # noqa: E402

# 言及の周りに数値があるか見る幅（文字）
NEAR = 120
NUM = re.compile(r"[-−+]?\d+\.\d+|\b\d+\s*%|=\s*[-−+]?\d")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True)


def fetch_raw(pid):
    """本文を**加工前の状態で**取る。キャプションを取るために構造が要る。"""
    r = sh(f'curl -sS --max-time 55 -w "%{{http_code}}" -o /tmp/r.html '
           f'"https://arxiv.org/html/{pid}"')
    if r.stdout.decode().strip().endswith("200"):
        raw = open("/tmp/r.html", encoding="utf-8", errors="ignore").read()
        if len(raw) > 20000:
            return raw, "html"
    time.sleep(0.8)
    r = sh(f'curl -sSL --max-time 85 -o /tmp/r.tgz -w "%{{http_code}}" '
           f'"https://arxiv.org/e-print/{pid}"')
    if r.stdout.decode().strip().endswith("200") and os.path.getsize("/tmp/r.tgz") > 1000:
        buf = []
        try:
            with tarfile.open("/tmp/r.tgz") as t:
                for m in t.getmembers():
                    if m.name.lower().endswith(".tex") and m.size < 3_000_000:
                        buf.append(t.extractfile(m).read().decode("utf-8", errors="ignore"))
            if buf:
                return "\n".join(buf), "eprint"
        except tarfile.ReadError:
            try:
                return gzip.open("/tmp/r.tgz", "rb").read().decode(
                    "utf-8", errors="ignore"), "eprint-gz"
            except Exception:
                pass
    return None, "fail"


def latex_captions(src):
    """`\\caption{...}` の中身を、波括弧の対応を取って抜く。"""
    out = []
    for m in re.finditer(r"\\caption\s*(\[[^\]]*\])?\s*\{", src):
        i, depth = m.end(), 1
        while i < len(src) and depth:
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        out.append(src[m.end():i - 1])
    return out


def html_captions(src):
    """arXiv HTML のキャプション。`ltx_caption` と `<figcaption>` の両方。"""
    out = []
    for m in re.finditer(r'<figcaption[^>]*>(.*?)</figcaption>', src, re.S | re.I):
        out.append(strip_html(m.group(1)))
    for m in re.finditer(r'<[^>]+class="[^"]*ltx_caption[^"]*"[^>]*>(.*?)</', src, re.S | re.I):
        out.append(strip_html(m.group(1)))
    return out


def plain_body(src, how):
    return strip_refs(strip_html(src) if how == "html"
                      else re.sub(r"\s+", " ", src))


def analyse(src, how):
    caps = " \n ".join(html_captions(src) if how == "html" else latex_captions(src))
    body = plain_body(src, how)
    low, capl = body.lower(), caps.lower()

    res = {"n_captions": len(html_captions(src) if how == "html"
                             else latex_captions(src)),
           "nchar": len(body), "mention": [], "with_number": [], "caption": []}
    for c, pat in CATS.items():
        hits = list(re.finditer(pat, low))
        if hits:
            res["mention"].append(c)
        if re.search(pat, capl):
            res["caption"].append(c)
        for h in hits:
            around = low[max(0, h.start() - NEAR):h.end() + NEAR]
            if NUM.search(around):
                res["with_number"].append(c)
                break
    res["reported"] = sorted(set(res["caption"]) | set(res["with_number"]))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    papers = [p for p in json.load(open(os.path.join(D, "relabeled.json")))
              if p.get("sf_mention") and p.get("is_sim")]
    print(f"対象 {len(papers)} 本\n")

    done = {}
    if os.path.exists(CACHE):
        done = {r["arxiv_id"]: r for r in json.load(open(CACHE))}

    rows, t0 = [], time.time()
    for i, p in enumerate(papers):
        pid = p["arxiv_id"]
        if pid in done and done[pid].get("how") != "fail":
            rows.append(done[pid])
            continue
        safe = pid.replace("/", "_")
        fp = os.path.join(OUT, safe + ".src")
        how_fp = fp + ".how"
        if os.path.exists(fp) and os.path.getsize(fp) > 2000:
            src = open(fp, encoding="utf-8", errors="ignore").read()
            how = open(how_fp).read().strip() if os.path.exists(how_fp) else "html"
        else:
            src, how = fetch_raw(pid)
            if src:
                open(fp, "w", encoding="utf-8").write(src)
                open(how_fp, "w").write(how)
            time.sleep(1.0)
        row = {"arxiv_id": pid, "how": how, "mention_old": p.get("new_ge1", [])}
        if src:
            row.update(analyse(src, how))
        rows.append(row)
        json.dump(rows, open(CACHE, "w"), ensure_ascii=False, indent=1)
        if (i + 1) % 10 == 0:
            ok = sum(1 for r in rows if r.get("mention") is not None)
            print(f"  {i+1}/{len(papers)}  取得 {ok}  {time.time()-t0:.0f}s", flush=True)

    json.dump(rows, open(CACHE, "w"), ensure_ascii=False, indent=1)

    got = [r for r in rows if r.get("nchar")]
    print(f"\n本文取得 {len(got)}/{len(papers)}  "
          f"（キャプション抽出できた論文 {sum(1 for r in got if r['n_captions'])} 本）")

    from collections import Counter
    for key in ("mention", "with_number", "caption", "reported"):
        c = Counter(x for r in got for x in r.get(key, []))
        print(f"\n{key}")
        for cat, n in c.most_common():
            print(f"  {cat:<28}{n:>4}/{len(got)}  ({n/len(got):.0%})")

    print(f"\n1本あたりのカテゴリ数の中央値")
    import statistics as st
    for key in ("mention", "with_number", "caption", "reported"):
        v = [len(r.get(key, [])) for r in got]
        print(f"  {key:<14}{st.median(v):.1f}")


if __name__ == "__main__":
    main()
