"""再標本化と検定。**ここが v0.1 で一番壊れていた場所である。**

## 何が壊れていたか

`separation.py` の置換検定は、実データ 124 本と生成 200 本を1つのプールに入れて
**ラベルを完全にランダムに入れ替えて**いた。これは「プールされた 324 個が交換可能」
という仮定であり、実データ側では成り立たない：

  - 窓長 1000・ストライド 250 なので、隣り合う窓は 75% を共有する
  - 2008年や2020年の窓は6指数すべてに同時に現れる
  - EURO STOXX 50 は DAX 構成銘柄を含むので一部は二重計上

完全置換の帰無分布では、擬似「実データ」群の内部相関が本物より弱くなる。
標本平均・分位点のばらつきが過小評価され、**帰無分布が細くなり、p 値が
小さく出る（反保守的）。**`robustness.py` は実効標本数を出して「区間は 1.59 倍に
広げるべき」と述べていたが、**p 値そのものは直していなかった。**

## 何に置き換えたか

交換の単位を、窓1本から**暦のブロック**に上げる（`windows.calendar_blocks`）。
ブロックごと丸ごと入れ替えるので、ブロック内の相関は帰無分布の中でも保たれる。

生成器側は独立な実行なので本来ブロックを持たない。実データ側のブロックサイズの
分布をそのまま写して**擬似ブロック**に分ける。両群のブロックサイズが揃っていないと
置換で標本数が動いてしまうためである。

**これは近似である。**擬似ブロックには内部相関が無いので、帰無分布の中で
「相関のあるブロックが生成器側に移り、相関の無いブロックが実データ側に移る」
入れ替えが起きる。厳密な検定ではない。だからブロック幅を変えた3段を並べて、
結論が仮定にどれだけ依存するかを出す（`separation.py`）。

## 追加したもの

- `energy_test` — 15個の周辺分布を別々に見るのではなく、**統計量ベクトル全体**を
  1つの多変量2標本検定にかける。個別 KS が全部通っても同時分布が壊れている
  生成器を落とすため。個別の診断は残す（何が壊れているかは個別でしか分からない）。
- `benjamini_hochberg` — 統計量 × 生成器のセルは 100 を超える。5% で見れば
  どこかに偶然の有意差が出る。
"""

import numpy as np


# --------------------------------------------------------------------- 統計量


def ks_stat(a, b):
    """2標本 Kolmogorov-Smirnov 統計量。分布関数の最大乖離。"""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    if len(a) < 5 or len(b) < 5:
        return np.nan
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _pairwise(x):
    """行どうしのユークリッド距離行列。"""
    d2 = np.maximum(
        (x * x).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(d2)


def energy_from_D(D, ia, ib):
    """距離行列から energy statistic を作る。

        E = nm/(n+m) [ 2·mean|a-b| - mean|a-a'| - mean|b-b'| ]

    分布が一致するときのみ 0。位置・尺度・依存構造のいずれのずれにも反応する。
    """
    n, m = len(ia), len(ib)
    saa = D[np.ix_(ia, ia)].sum()
    sbb = D[np.ix_(ib, ib)].sum()
    sab = D[np.ix_(ia, ib)].sum()
    e = 2.0 * sab / (n * m) - saa / (n * n) - sbb / (m * m)
    return float(n * m / (n + m) * e)


# --------------------------------------------------------------- ブロック置換


def pseudo_blocks(n, size_profile, rng):
    """独立な n 個を、与えられたサイズ分布を写した擬似ブロックに分ける。

    サイズ分布を巡回させながら使い、余りは最後のブロックに入れる。
    どの実行がどのブロックに入るかはランダム（実行は交換可能なので任意）。
    """
    order = rng.permutation(n)
    sizes, i = [], 0
    while sum(sizes) < n:
        sizes.append(int(size_profile[i % len(size_profile)]))
        i += 1
    over = sum(sizes) - n
    if over:
        sizes[-1] -= over
        if sizes[-1] <= 0:
            over = -sizes.pop()
            if over and sizes:
                sizes[-1] -= over
    out, at = [], 0
    for s in sizes:
        if s <= 0:
            continue
        out.append(order[at:at + s])
        at += s
    return out


def block_perm_test(a_vals, a_blocks, b_vals, stat, rng, n_perm=2000,
                    b_blocks=None):
    """ブロック単位の置換検定。返り値 (統計量, p値, 帰無の95%点, ブロック数)。

    `a_blocks` は `a_vals` と同じ長さのブロック番号。`b_blocks` を与えなければ
    `a_blocks` のサイズ分布を写した擬似ブロックを作る（生成器側は独立なので）。

    置換は「全ブロックのうち K_A 個を A 群に割り当てる」。ブロックサイズが
    完全に揃うとは限らないので、標本数は置換ごとに多少揺れる。この揺れは
    帰無分布をわずかに広げる方向、つまり保守側に働く。
    """
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    ma, mb = np.isfinite(a), np.isfinite(b)
    a, ab = a[ma], np.asarray(a_blocks)[ma]
    b = b[mb]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan, 0

    groups_a = [np.where(ab == u)[0] for u in sorted(set(ab.tolist()))]
    if b_blocks is None:
        sizes = sorted((len(g) for g in groups_a), reverse=True)
        groups_b = pseudo_blocks(len(b), sizes, rng)
    else:
        bb = np.asarray(b_blocks)[mb]
        groups_b = [np.where(bb == u)[0] for u in sorted(set(bb.tolist()))]

    obs = stat(a, b)
    pool = np.concatenate([a, b])
    off = len(a)
    blocks = [g for g in groups_a] + [g + off for g in groups_b]
    ka, kt = len(groups_a), len(groups_a) + len(groups_b)

    null = np.empty(n_perm)
    for i in range(n_perm):
        pick = rng.permutation(kt)[:ka]
        sel = np.concatenate([blocks[j] for j in pick])
        mask = np.zeros(len(pool), bool)
        mask[sel] = True
        null[i] = stat(pool[mask], pool[~mask])
    good = np.isfinite(null)
    pval = float((1.0 + np.sum(null[good] >= obs)) / (good.sum() + 1.0))
    return float(obs), pval, float(np.percentile(null[good], 95)), kt


def block_boot_test(a_vals, a_blocks, b_vals, stat, rng, n_boot=2000):
    """**設計を保つ帰無分布。**ブロック置換より正しく、これを主に使う。

    ブロック置換にはまだ歪みが残る。実データ側は相関のあるブロックの集まり、
    生成器側は独立な 200 本、という**非対称な設計**なのに、置換は相関ブロックを
    生成器側へ、相関のない擬似ブロックを実データ側へ移してしまうからである。
    実測でも、ブロック内相関 0.6 のとき名目 5% に対して棄却率 15.8% だった
    （`selftest.py`）。

    ここでは置換をやめ、帰無分布を**設計そのままに**作る：

      - A* … 実データのブロックを**復元抽出**する（相関もブロック間のばらつきも保つ）
      - B* … 同じ経験分布から独立に n_b 個引く（生成器側は独立なので）

    両者は構成上まったく同じ周辺分布を持つので帰無仮説は真、かつ
    A* にだけ依存が入るという実際の設計が保たれる。

    **これでも名目水準には届かない。**実測（`selftest.py`）で名目 5% に対し
    真の大きさは 10〜13% ある。原因は手法ではなく**ブロックが6個しかないこと**で、
    クラスタ数が 20 を切ると過剰棄却するのは広く知られている。平滑化を入れても
    改善しないことは確認済み（ tie の副作用ではない）。

    **したがって p 値は較正して読む。**名目 p < 0.01 で真の大きさが 4〜5% に
    収まるので、`separation.py` はそこを 5% 水準の判定線として使う。
    """
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    ma, mb = np.isfinite(a), np.isfinite(b)
    a, ab = a[ma], np.asarray(a_blocks)[ma]
    b = b[mb]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan, 0

    groups = [np.where(ab == u)[0] for u in sorted(set(ab.tolist()))]
    k, nb = len(groups), len(b)
    obs = stat(a, b)
    null = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        astar = np.concatenate([a[groups[j]] for j in pick])
        bstar = rng.choice(a, nb, replace=True)
        null[i] = stat(astar, bstar)
    good = np.isfinite(null)
    pval = float((1.0 + np.sum(null[good] >= obs)) / (good.sum() + 1.0))
    return float(obs), pval, float(np.percentile(null[good], 95)), k


def iid_perm_test(a_vals, b_vals, stat, rng, n_perm=2000):
    """**旧版**の完全置換。比較のためだけに残す。窓の依存を無視している。"""
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan
    obs = stat(a, b)
    pool = np.concatenate([a, b])
    na = len(a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pool)
        null[i] = stat(p[:na], p[na:])
    good = np.isfinite(null)
    pval = float((1.0 + np.sum(null[good] >= obs)) / (good.sum() + 1.0))
    return float(obs), pval, float(np.percentile(null[good], 95))


def energy_block_test(A, a_blocks, B, rng, n_boot=2000):
    """統計量ベクトル**全体**の2標本検定（energy distance）。

    15個の周辺分布を別々に見ていると、どれも個別には合っているのに同時分布が
    現実と全く違う生成器を通してしまう。逆に、統計量どうしが相関していれば
    同じ失敗を15回数えていることにもなる。ここは総合層である
    （何が壊れているかは個別の KS でしか分からないので、そちらは残す）。

    A: (n_a, d) 実データ側、B: (n_b, d) 生成器側。列ごとに中央値と MAD で
    標準化してから距離を測る（統計量ごとに尺度が3桁違うため）。

    帰無分布は一変量と同じ**ブロック復元抽出**で作る。置換だと実測で
    名目 5% に対し 21% 棄却した（生成器側に相関ブロックの相手がいないため）。
    距離行列は一度だけ作り、以降は添字を選び直すだけにしてある。
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    keep = np.all(np.isfinite(A), 1)
    A, ab = A[keep], np.asarray(a_blocks)[keep]
    B = B[np.all(np.isfinite(B), 1)]
    if len(A) < 5 or len(B) < 5:
        return np.nan, np.nan, np.nan

    pool = np.vstack([A, B])
    med = np.median(pool, 0)
    mad = np.median(np.abs(pool - med), 0)
    mad[mad <= 0] = 1.0
    D = _pairwise((pool - med) / (1.4826 * mad))

    na, nb = len(A), len(B)
    obs = energy_from_D(D, np.arange(na), np.arange(na, na + nb))

    groups = [np.where(ab == u)[0] for u in sorted(set(ab.tolist()))]
    k = len(groups)
    null = np.empty(n_boot)
    for i in range(n_boot):
        ia = np.concatenate([groups[j] for j in rng.integers(0, k, k)])
        ib = rng.integers(0, na, nb)
        null[i] = energy_from_D(D, ia, ib)
    p = float((1.0 + np.sum(null >= obs)) / (n_boot + 1.0))
    return obs, p, float(np.percentile(null, 95))


def intraclass_rho(values, labels):
    """ブロック内相関（級内相関）。`selftest.py` の較正表のどの行を見るかを決める。

    rho = 1 - ブロック内分散 / 全体分散。ブロック間で系統的に違えば 1 に近づく。
    """
    v = np.asarray(values, float)
    lab = np.asarray(labels)
    m = np.isfinite(v)
    v, lab = v[m], lab[m]
    if len(v) < 5:
        return np.nan
    groups = [v[lab == u] for u in sorted(set(lab.tolist()))]
    groups = [g for g in groups if len(g) > 1]
    tot = np.var(v)
    if not groups or tot <= 0:
        return np.nan
    within = np.mean([np.var(g) for g in groups])
    return float(np.clip(1.0 - within / tot, 0.0, 0.99))


# ------------------------------------------------------------------- 多重検定


def benjamini_hochberg(pvals):
    """BH の q 値（FDR）。入力と同じ並びで返す。NaN はそのまま通す。"""
    p = np.asarray(pvals, float)
    ok = np.where(np.isfinite(p))[0]
    q = np.full(len(p), np.nan)
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1.0))
        q[i] = prev
    return q
