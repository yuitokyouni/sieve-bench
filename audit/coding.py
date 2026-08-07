"""文献ラベルを本文から取り直す。

元の調査は 430本中 222本が要旨のみ・gpt-4o-mini による抽出だった。
本文で取り直し、機械的に再現可能な形にする。

判定は正規表現。参考文献欄は必ず除去する（前回 \\bibitem{bouchaud2001leverage}
のような引用を「本文で leverage を扱っている」と誤検出した）。
出現回数を記録し、しきい値1回以上／3回以上の両方を出す。

**未実装:** arXiv から取れなかった 109 本のフォールバック（Semantic Scholar の
openAccessPdf / Crossref）はまだ入れていない。到達性は確認済みだが、
現状の `fetch()` は arXiv HTML → arXiv e-print までしか試していない。
"""

import gzip
import json
import os
import re
import subprocess
import tarfile
import time

D = os.path.dirname(os.path.abspath(__file__))
os.chdir(D)
OUT = os.path.join(D, "fulltext")
os.makedirs(OUT, exist_ok=True)

LIT = os.environ.get("ABM_LIT", "/workspace/financial-abm-lab/data/literature_methods.json")

CATS = {
    "fat-tails": r"(excess kurtosis|kurtosis|leptokurt|fat[\s\-]?tail|heavy[\s\-]?tail|"
                 r"tail index|hill estimator|power[\s\-]?law tail|tail exponent)",
    "vol-clustering": r"(volatility clustering|clustering of volatility|"
                      r"clustered volatilit|volatility cluster)",
    "long-memory": r"(long[\s\-]?memory|long[\s\-]?range dependence|"
                   r"slowly decaying autocorrelat|hurst exponent)",
    "leverage": r"(leverage effect|return[\s\-]volatility (asymmetry|correlation)|"
                r"asymmetric volatilit)",
    "gain-loss-asymmetry": r"(gain[/\s\-]?loss asymmetry)",
    "absence-of-autocorr": r"(absence of (linear |significant )?autocorrelat|"
                           r"lack of autocorrelat|no (significant )?autocorrelat|"
                           r"uncorrelated returns|linear unpredictability)",
    "aggregational-gaussianity": r"(aggregational (gaussianity|normality)|"
                                 r"aggregate gaussianity)",
    "volume-volatility-corr": r"(volume[\s\-]volatility (correlation|relation)|"
                              r"trading volume and volatility)",
    "regime-switching": r"(regime[\s\-]switching|regime shift)",
}
INC_SF = re.compile(r"stylized fact|stylised fact", re.I)
INC_SIM = re.compile(r"(agent[\s\-]based (model|simulation)|artificial (stock )?market|"
                     r"market simulat|simulated market|multi[\s\-]agent simulat)", re.I)


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True)


def strip_html(h):
    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", h, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))


def strip_refs(t):
    for p in (r"\\begin\{thebibliography\}", r"\\bibliography\{", r"\\printbibliography"):
        m = re.search(p, t)
        if m:
            t = t[:m.start()]
    t = re.sub(r"\\bibitem.*?(?=\\bibitem|$)", " ", t, flags=re.S)
    ms = list(re.finditer(r"\b(References|REFERENCES|Bibliography)\b", t))
    if ms and ms[-1].start() > len(t) * 0.4:
        t = t[:ms[-1].start()]
    return t


def fetch(pid):
    r = sh(f'curl -sS --max-time 55 -w "%{{http_code}}" -o /tmp/f.html '
           f'"https://arxiv.org/html/{pid}"')
    if r.stdout.decode().strip().endswith("200"):
        raw = open("/tmp/f.html", encoding="utf-8", errors="ignore").read()
        if len(raw) > 20000:
            return strip_html(raw), "html"
    time.sleep(0.8)
    r = sh(f'curl -sSL --max-time 85 -o /tmp/f.tgz -w "%{{http_code}}" '
           f'"https://arxiv.org/e-print/{pid}"')
    if r.stdout.decode().strip().endswith("200") and os.path.getsize("/tmp/f.tgz") > 1000:
        buf = []
        try:
            with tarfile.open("/tmp/f.tgz") as t:
                for m in t.getmembers():
                    if m.name.lower().endswith((".tex", ".bbl", ".txt")) and m.size < 3_000_000:
                        buf.append(t.extractfile(m).read().decode("utf-8", errors="ignore"))
            if buf:
                return re.sub(r"\s+", " ", " ".join(buf)), "eprint"
        except tarfile.ReadError:
            try:
                return re.sub(r"\s+", " ", gzip.open("/tmp/f.tgz", "rb")
                              .read().decode("utf-8", errors="ignore")), "eprint-gz"
            except Exception:
                pass
    return None, "fail"


def main():
    lit = json.load(open(LIT))
    rows = []
    t0 = time.time()
    for i, rec in enumerate(lit):
        pid = rec["arxiv_id"]
        safe = pid.replace("/", "_")
        fp = os.path.join(OUT, safe + ".txt")
        if os.path.exists(fp) and os.path.getsize(fp) > 2000:
            txt, how = open(fp, encoding="utf-8", errors="ignore").read(), "cached"
        else:
            txt, how = fetch(pid)
            if txt:
                open(fp, "w", encoding="utf-8").write(txt)
            time.sleep(0.8)
        old = rec.get("stylized_facts_targeted") or ""
        out = {"arxiv_id": pid, "year": rec.get("year"), "title": rec.get("title", "")[:110],
               "how": how, "old_labels": sorted(
                   x.strip().lower() for x in old.split(",") if x.strip())}
        if txt:
            body = strip_refs(txt)
            low = body.lower()
            out["nchar"] = len(body)
            out["sf_mention"] = bool(INC_SF.search(body))
            out["is_sim"] = bool(INC_SIM.search(body))
            out["counts"] = {c: len(re.findall(p, low)) for c, p in CATS.items()}
            out["new_ge1"] = sorted(c for c, n in out["counts"].items() if n >= 1)
            out["new_ge3"] = sorted(c for c, n in out["counts"].items() if n >= 3)
        rows.append(out)
        if (i + 1) % 20 == 0 or i == len(lit) - 1:
            json.dump(rows, open("relabeled.json", "w"), ensure_ascii=False, indent=1)
            ok = sum(1 for r in rows if r.get("counts"))
            print(f"{i+1}/{len(lit)}  本文取得 {ok}  経過 {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open("relabeled.json", "w"), ensure_ascii=False, indent=1)
    print(f"完了 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
