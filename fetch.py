"""実データの取得。

このリポジトリは指数データを同梱しない。Yahoo Finance の生レスポンスの再配布は
利用規約上グレーであり、ベンチマークがそれで消えるのは避けたいからである。
代わりに各自がここで取る。

**取得期間は絶対日付で固定してある。**`range=25y` のような相対指定だと
実行した日によって窓がずれ、数字が再現しなくなる。period1/period2 を
epoch 秒で打ち込んであるので、いつ実行しても同じ系列が返る。

返ってきたデータが論文の表と同じものかは、ハッシュで確認できる：

    python3 fetch.py            # 取得して検証まで
    python3 fetch.py --verify   # 取得済みのものを検証するだけ

ハッシュは (timestamp, close) の対を小数6桁で並べたものに対して取っている。
meta の regularMarketTime などは取得時刻で変わるので、ハッシュには含めない。
"""

import argparse
import calendar
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 取得期間（絶対固定）。ここを変えると数字が変わる。
PERIOD1 = calendar.timegm(dt.date(2001, 7, 30).timetuple())   # 996451200
PERIOD2 = calendar.timegm(dt.date(2026, 8, 1).timetuple())    # 1785542400

SYMBOLS = {
    "ftse":  "^FTSE",       # FTSE 100
    "gdaxi": "^GDAXI",      # DAX
    "gspc":  "^GSPC",       # S&P 500
    "hsi":   "^HSI",        # ハンセン
    "n225":  "^N225",       # 日経225
    "sx5e":  "^STOXX50E",   # EURO STOXX 50
}

# 論文の表を出したときの系列。fetch 後に --verify で照合される。
# sx5e だけ開始が遅いのは、Yahoo にそれ以前が無いため。
EXPECTED = {
    "ftse":  ("8b7acc0fe42e6557756606c3e1a739919c6f86048d08c947d8816d1469ccfbe9", 6317),
    "gdaxi": ("24be7ac0c8a97737a3530ff93894425725fc575bd8b70164a9be2643a39ec4d9", 6349),
    "gspc":  ("83c09f5271ff01a2119aaf161491f3798c1c6df8dfd765287b969ebd66a78e09", 6288),
    "hsi":   ("48423d72e252a862e10bcd5ab29a1a18db9dc57501874ab19c3f05c4915fc1b4", 6163),
    "n225":  ("ef4207e7981f389bd1c887411402200d167727e5d8f6350581dcc67fb87f2581", 6122),
    "sx5e":  ("f2fae44bd503a72e3a0c0600bd8379bf60a3d40157f89d1fb867b55b8318b99c", 4846),
}

BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
UA = "Mozilla/5.0 (compatible; sieve-bench/0.1)"


def digest(payload):
    """(timestamp, close) の対から安定なハッシュを作る。

    close が None の日（休場など）は落とす。run_power.py の読み取りと同じ扱い。
    """
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    close = res["indicators"]["quote"][0]["close"]
    good = [(t, c) for t, c in zip(ts, close) if c is not None]
    body = "\n".join(f"{t},{c:.6f}" for t, c in good)
    return hashlib.sha256(body.encode()).hexdigest(), good


def fetch_one(symbol, tries=4):
    url = (BASE + urllib.parse.quote(symbol)
           + f"?period1={PERIOD1}&period2={PERIOD2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = 2
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == tries - 1:
                raise
            print(f"    失敗（{type(e).__name__}）、{delay}秒後に再試行", flush=True)
            time.sleep(delay)
            delay *= 2


def span(good):
    a = dt.datetime.utcfromtimestamp(good[0][0]).date()
    b = dt.datetime.utcfromtimestamp(good[-1][0]).date()
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="取得済みの data/ を照合するだけ（通信しない）")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    print(f"期間 {dt.datetime.utcfromtimestamp(PERIOD1).date()} 〜 "
          f"{dt.datetime.utcfromtimestamp(PERIOD2).date()}（絶対固定）\n")

    bad = []
    for i, (name, symbol) in enumerate(sorted(SYMBOLS.items())):
        path = os.path.join(DATA, f"{name}.json")

        if args.verify:
            if not os.path.exists(path):
                print(f"  {name:6s} 未取得")
                bad.append(name)
                continue
            payload = json.load(open(path))
        else:
            print(f"  {name:6s} {symbol} を取得中…", flush=True)
            payload = fetch_one(symbol)
            with open(path, "w") as f:
                json.dump(payload, f)
            if i < len(SYMBOLS) - 1:
                time.sleep(1)     # 連打しない

        h, good = digest(payload)
        want_h, want_n = EXPECTED[name]
        a, b = span(good)
        ok = (h == want_h and len(good) == want_n)
        mark = "一致" if ok else "不一致"
        print(f"  {name:6s} {symbol:11s} n={len(good):5d}  {a} 〜 {b}  "
              f"{h[:16]}  {mark}")
        if not ok:
            bad.append(name)
            if len(good) != want_n:
                print(f"         期待 n={want_n}、実際 n={len(good)}")

    print()
    if bad:
        print(f"照合できなかった指数: {', '.join(bad)}")
        print("Yahoo 側の系列が改定された可能性がある。数字は論文の表と")
        print("ずれる。power.json を作り直した上で、差分を明記して使うこと。")
        return 1
    print("全指数が論文の表と同一。`python3 run_power.py` で再現できる。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
