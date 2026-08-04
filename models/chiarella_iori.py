"""Chiarella-Iori-Perelló の指値注文板モデルを論文通りに実装する。

出典
    Chiarella, C., Iori, G. & Perelló, J.
    "The Impact of Heterogeneous Trading Rules on the Limit Order Book
     and Order Flows." arXiv:0711.3581  (JEDC 33(3), 2009)

**リポジトリの `abm_models/chiarella_iori` はこのモデルではない。**
板が存在せず、best_bid/ask と depth がスカラー4つで、集約需要に線形の
価格インパクトを掛けているだけだった。指値注文もキューもマッチングも無い。

論文の式:

  期待リターン        r̂ᵢ = 1/(g₁+g₂+n) [ g₁(1/τ_f)ln(pᶠ/p) + g₂ r̄ᵢ + n ε ]
  過去平均            r̄ᵢ = (1/τᵢ) Σ_{j=1..τᵢ} r_{t-j},  r_t = ln(p_t/p_{t-1})
  期待価格            p̂ᵢ = p exp(r̂ᵢ τᵢ)
  投資期間            τᵢ = ⌈τ (1+g₁)/(1+g₂)⌉
  リスク回避          αᵢ = α (1+g₁)/(1+g₂)
  分散推定            Vᵢ = (1/τᵢ) Σ (r_{t-j} − r̄ᵢ)²
  CARA 最適保有       πᵢ(p) = ln(p̂ᵢ/p) / (αᵢ Vᵢ p)
  満足価格            πᵢ(p*) = Sᵢ
  予算下限            p_m (πᵢ(p_m) − Sᵢ) = Cᵢ
  上限                p_M = p̂ᵢ
  重み                g₁~Exp(σ₁), g₂~Exp(σ₂), n~Exp(σ_n)   ※平均が σ

  注文（p を [p_m, p_M] から一様に引く）:
      p_m < p < aᵍ     買い指値   s = πᵢ(p) − Sᵢ
      aᵍ ≤ p < p*      買い成行   s = πᵢ(aᵍ) − Sᵢ
      p = p*           発注なし
      p* < p ≤ bᵍ      売り成行   s = Sᵢ − πᵢ(bᵍ)
      bᵍ < p ≤ p_M     売り指値   s = Sᵢ − πᵢ(p)

  未約定の指値は t+τᵢ で板から除かれる。価格は約定価格、無ければ mid。

パラメータ（論文 §Simulation Analysis）:
    N_A=5000, τ=200, Δ=0.0005, σ_ε=1e-4, σ_n=1, α=0.1
    N_S=50, W=N_S·pᶠ(0), S₀~U[0,N_S], C₀~U[0,W]
    pᶠ は幾何ブラウン運動、pᶠ(0)=300、ドリフト0、σ=1e-3
    代表ケース: σ₁=10.00, σ₂=1.20

**明示的な仮定（論文に規定が無い箇所）:**
  1. τ_f の数値が本文に無い。**τ_f = τᵢ（各エージェントの投資期間）が正しい。**
     定数 200 にすると式(3) が τᵢ/τ_f ≈ 7.5 倍に増幅し、V を固定した比較で
     帰還比が 1e2 → 7e5 に飛ぶ（`book_stats.py`）。τ_f = τᵢ なら式(1) の
     1/τ_f が式(3) の τᵢ と相殺し、ファンダ項の寄与が g₁/Σg × ln(pᶠ/p) に収まる。
  2. **初期化。**Vᵢ は過去 τᵢ 本のリターンの分散だが、開始時には履歴が無い。
     Vᵢ→0 だと πᵢ(p)=ln(p̂/p)/(αᵢVᵢp) が発散し、板が壊れる（実測：価格が
     300→0.25 に崩壊）。ここでは **価格がファンダメンタルに従う助走期間**を
     tau_cap ステップ置いて履歴を作ってから、エージェントの取引を始める。
  3. τᵢ = ⌈τ(1+g₁)/(1+g₂)⌉ は裾が重く、σ₁=10 では 40% が 2000 を超える。
     計算量のため tau_cap で打ち切る（論文には上限が無い）。**この打ち切りは
     結果に影響しない。**未約定の指値の寿命は τᵢ だが、板は寿命ではなく
     成行の消費で決まっている。tau_cap を 2000→60000 に上げても板の厚みは
     26→23、実現分散は 1.0e-4→1.3e-4 で動かない。

**論文の Figure 7 は、論文自身のパラメータでは出ない。方程式ではなく数値の側が
合わない。**再現には2つ決める必要があった（下の「決着したこと」）。

**まず図番号の訂正。**距離の図は **Figure 7**（横軸 0〜0.07）であって
Figure 6 ではない。Figure 6 は成行注文のサイズの DDF。

**Vᵢ の次元は論文が自分で決着させている。**付録 A に「本モデルはリターン分散の
期待形成を規定しない。そのため Vt[ρ_{t+τ}] を式(9) の履歴分散で置く」とある。
1ステップあたりの分散をそのまま horizon の分散の代用にするのは論文の明示的な
選択であって、実装の取り違えではない。`v_horizon=False` が論文通り。

**注文が散らばる幅は式から一意に決まる（測定で比 1.0000）。**予算式(11)
p_m(π(p_m) − S) = C に π = ln(p̂/p)/(αVp) を入れると

    ln(p_M / p_m) = αᵢ · Vᵢ · (p_m·Sᵢ + Cᵢ)        ← 抽選区間の対数幅

つまり**幅は「リスク回避度 × 分散 × 資産額」ちょうど**。注文はこの区間の一様
抽選なので、これが中値からの距離をそのまま決める。W = N_S·pᶠ = 15000、α = 0.1
なので、板が締まる条件は V ≪ 1/(αW) = 6.7e-4、すなわち1ステップ std ≪ 2.6%。

この幅が V を上げ、V が幅を上げる。エージェントの使う分散を v_pin で固定して
その板が実際に生む分散 V' を測ると（`book_stats.py`）、**帰還利得**が出る。

**決着したこと 1 — ノイズは √τᵢ で積む（`noise_horizon`）。**式(1) の ε を
1ステップの率と読むと式(3) が τᵢ 倍するので、σ_ε=1e-4 が τᵢ/Σg≈115 倍されて
1.15% になる。これが V→0 でも消えない床（1.3e-4）の正体で、床が σ_f²=1e-6 の
130倍あるから板は決して締まらない。ε を1ステップのショックと読んで √τᵢ で
積むと床は 8.5e-7 と σ_f² を下回る。

    ノイズの積み方       V の床      V=1e-6 での利得
    linear（字義通り）   1.27e-4     101
    sqrt                8.46e-7     2.05

**決着したこと 2 — 板は A = α·N_S·pᶠ だけで決まる。**上の恒等式の帰結として、
締まり具合はこの群のみに依存する。論文値 α=0.1 では A=1500 で利得 2.05、
つまり**論文自身のパラメータでは論文の Figure 7 が出ない。**

    α       A      利得   自由走行の V   指値の距離   スプレッド   価格/ファンダ
    0.10   1500    2.05    2.54e-4      11.11%     1.522%      0.798
    0.05    750    1.42    2.68e-6       0.38%     0.112%      0.959
    0.03    450    0.96    1.08e-6       0.15%     0.056%      0.964   ← 論文の姿

A=450 で V が σ_f²=1e-6 に落ち着き、価格がファンダに密着し、指値が中値の
0.15% に入る。**期待形成・CARA 需要・注文配置表・マッチング・期限切れは
正しかった。合わなかったのは α の値とノイズの積み方の2つだけ。**

**残っている食い違い:**
  1. **資力不足による注文削除。**発注済みの指値の持ち主が現金／在庫を
     失ったときの扱いを論文は規定していない。ここでは板から消しているが、
     利得<1 の領域では**注文が消える経路の 65.7%** がこれで（約定 31%、
     期限切れ 3%）、支配的な機構になっている。発注時に拘束する
     （エスクロー）読み方もあり、板の形が変わる。
  2. **3パネルの順序。**論文は指値の広がりを ファンダ < ノイズ < チャート
     の順に描くが、ここでは ノイズ < ファンダ < チャート になる。

外れた仮説（記録）:
  - **板の薄さ・注文の寿命ではない。**tau_cap を 2000→60000（打ち切り
    40%→0%）にしても板の厚み 26→23、V' 1.0e-4→1.3e-4 で動かない。板は
    寿命ではなく成行の消費で決まる。
  - τ_f の値 → τ_f = τᵢ が正しい（仮定1に記載）
  - Vᵢ に τᵢ を掛ける → 悪化（付録 A の通り論文は掛けていない）
  - 総需要＝発行済株数 となる価格から始める → 変わらず
  - r̂ と p̂ のクリップ → **一度も発動しない**（全設定で 0.00%）。無害。

    python3 models/book_stats.py              # 板の統計量（約40秒）
    python3 calibration/calibrate_cip.py      # S&P500 への較正
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CIParams:
    n_agents: int = 5000
    tau: int = 200            # 参照投資期間
    tau_f: int = 200          # tau_f_mode="fixed" のときだけ使う（比較用）
    tick: float = 0.0005      # Δ
    sigma_eps: float = 1e-4   # ノイズ成分の標準偏差
    sigma_1: float = 10.0     # ファンダメンタリスト重みの平均
    sigma_2: float = 1.20     # チャーティスト重みの平均
    sigma_n: float = 1.0      # ノイズ重みの平均
    alpha: float = 0.1        # 参照リスク回避度
    p_f0: float = 300.0       # ファンダメンタル初期値
    sigma_fund: float = 1e-3  # ファンダメンタルの GBM ボラティリティ
    n_stock: float = 50.0     # N_S
    tau_cap: int = 2000       # τᵢ の上限（計算量のため。論文には無い）
    tau_f_mode: str = "own"   # "fixed"=τ_f を定数、"own"=各エージェントの τᵢ を使う
    v_horizon: bool = False   # Vᵢ を投資期間全体の分散にする（付録Aより False が論文通り）
    start_at_equilibrium: bool = True  # 総需要＝発行済株数 となる価格から始める
    v_pin: float | None = None  # 診断用。Vᵢ を実測でなくこの値に固定する
    # ノイズ成分を horizon にどう積むか。論文の式(1)+(3) を字義通り読むと ε は
    # 1ステップの率で、τᵢ 倍されて ln(p̂/p) に入る（"linear"）。これだと
    # σ_ε=1e-4 が τᵢ/Σg ≈ 115 倍されて 1.15% になり、板が締まらない。
    # "sqrt" は ε を1ステップのショックと読み、horizon では √τᵢ で積む。
    noise_horizon: str = "sqrt"


class ChiarellaIoriPerello:
    """連続ダブルオークション。seed → 約定価格系列とリターン。"""

    name = "chiarella_iori_perello"

    def __init__(self, n_steps: int = 20000, warmup: int = 2000,
                 params: CIParams | None = None):
        self.n_steps = int(n_steps)
        self.warmup = int(warmup)
        self.p = params or CIParams()

    # ---------------------------------------------------------------- 板
    def _best_ask(self):
        while self._asks:
            pr, oid = self._asks[0]
            o = self._orders.get(oid)
            if o is None or o[1] <= 1e-12:
                heapq.heappop(self._asks)
                continue
            return pr, oid
        return None, None

    def _best_bid(self):
        while self._bids:
            npr, oid = self._bids[0]
            o = self._orders.get(oid)
            if o is None or o[1] <= 1e-12:
                heapq.heappop(self._bids)
                continue
            return -npr, oid
        return None, None

    def _expire(self, t):
        while self._exp and self._exp[0][0] <= t:
            _, oid = heapq.heappop(self._exp)
            self._orders.pop(oid, None)

    def _add_limit(self, side, price, vol, expiry, agent):
        oid = self._next_id
        self._next_id += 1
        self._orders[oid] = [price, vol, side, agent]
        heapq.heappush(self._exp, (expiry, oid))
        if side == "B":
            heapq.heappush(self._bids, (-price, oid))
        else:
            heapq.heappush(self._asks, (price, oid))

    # ---------------------------------------------------------------- 需要
    @staticmethod
    def _pi(p, p_hat, aV):
        """πᵢ(p) = ln(p̂/p) / (αᵢ Vᵢ p)"""
        return math.log(p_hat / p) / (aV * p)

    def _solve_p_star(self, p_hat, aV, S):
        """πᵢ(p*) = S を (0, p̂] で解く。π は p について単調減少。"""
        if S <= 0:
            return p_hat
        lo, hi = p_hat * 1e-8, p_hat
        if self._pi(hi, p_hat, aV) >= S:
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._pi(mid, p_hat, aV) > S:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _solve_p_m(self, p_hat, aV, S, C, p_star):
        """p_m(πᵢ(p_m) − S) = C を (0, p*] で解く。左辺は p について減少。"""
        f = lambda p: math.log(p_hat / p) / aV - p * S - C
        lo, hi = p_hat * 1e-8, p_star
        if f(hi) > 0:
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ---------------------------------------------------------------- 本体
    def run(self, *, seed: int) -> dict:
        p = self.p
        rng = np.random.default_rng(seed)
        T = self.warmup + self.n_steps

        g1 = rng.exponential(p.sigma_1, p.n_agents) if p.sigma_1 > 0 else np.zeros(p.n_agents)
        g2 = rng.exponential(p.sigma_2, p.n_agents) if p.sigma_2 > 0 else np.zeros(p.n_agents)
        nn = rng.exponential(p.sigma_n, p.n_agents) if p.sigma_n > 0 else np.zeros(p.n_agents)
        gsum = g1 + g2 + nn
        bad = gsum <= 1e-12
        nn[bad] = 1e-6
        gsum = g1 + g2 + nn

        ratio = (1.0 + g1) / (1.0 + g2)
        tau_i = np.clip(np.ceil(p.tau * ratio).astype(int), 2, p.tau_cap)
        alpha_i = p.alpha * ratio

        W = p.n_stock * p.p_f0
        S = rng.uniform(0.0, p.n_stock, p.n_agents)
        C = rng.uniform(0.0, W, p.n_agents)

        self._orders, self._bids, self._asks, self._exp = {}, [], [], []
        self._next_id = 0

        # --- 助走：価格をファンダメンタルに追随させて履歴を作る ---
        H = p.tau_cap
        price = np.empty(H + T)
        fund = np.empty(H + T)          # ファンダメンタル系列（比較・作図用）
        logret = np.zeros(H + T)
        trades = np.zeros(H + T, dtype=bool)
        pf = p.p_f0
        # **開始水準。**価格＝ファンダだと期待リターンがゼロで CARA 需要もゼロに
        # なり、S>0 を持つ全エージェントが同時に売り手になる。実測ではここから
        # 売り崩れが始まり、高ボラ・低価格の第二均衡に落ちていた（ファンダの17.8%）。
        # 総需要が発行済株数に等しくなる水準から始める。
        p0 = p.p_f0
        if p.start_at_equilibrium:
            V0 = p.sigma_fund ** 2
            supply = p.n_stock / 2.0            # S₀~U[0,N_S] の平均
            lo_, hi_ = p.p_f0 * 0.5, p.p_f0
            for _ in range(60):
                mid_ = 0.5 * (lo_ + hi_)
                lr = math.log(p.p_f0 / mid_)
                dem = 0.0
                for j in range(0, p.n_agents, 25):   # 間引いて概算
                    lph = (g1[j] * lr) / gsum[j]
                    dem += max(lph / (alpha_i[j] * V0 * mid_), 0.0)
                dem /= len(range(0, p.n_agents, 25))
                if dem > supply:
                    lo_ = mid_
                else:
                    hi_ = mid_
            p0 = 0.5 * (lo_ + hi_)
        price[0] = p0
        for t in range(1, H):
            pf *= math.exp(-0.5 * p.sigma_fund ** 2 + p.sigma_fund * rng.normal())
            price[t] = p0 * (pf / p.p_f0)
            fund[t] = pf
            logret[t] = math.log(price[t] / price[t - 1])

        for t in range(H, H + T):
            self._expire(t)
            pf *= math.exp(-0.5 * p.sigma_fund ** 2 + p.sigma_fund * rng.normal())
            fund[t] = pf

            i = int(rng.integers(p.n_agents))
            ti = int(tau_i[i])
            hist = logret[max(0, t - ti):t]
            if hist.size < 2:
                price[t] = price[t - 1]
                continue
            rbar = float(hist.mean())
            Vi = float(hist.var()) if p.v_pin is None else p.v_pin
            if not np.isfinite(Vi) or Vi <= 1e-14:
                Vi = 1e-14

            pt = price[t - 1]
            eps = rng.normal(0.0, p.sigma_eps)
            tf = float(ti) if p.tau_f_mode == "own" else float(p.tau_f)
            # 式(3) が r̂ を τᵢ 倍するので、ε をそのまま入れると horizon 換算で
            # τᵢ 倍になる。ランダムウォークとして積むなら √τᵢ が正しい。
            ne = nn[i] * eps / (math.sqrt(ti) if p.noise_horizon == "sqrt" else 1.0)
            rhat = (g1[i] * math.log(pf / pt) / tf + g2[i] * rbar + ne) / gsum[i]
            rhat = float(np.clip(rhat, -0.5, 0.5))
            p_hat = pt * math.exp(rhat * ti)
            p_hat = float(np.clip(p_hat, p.tick, 1e6))

            # 次元だけを見れば Var は投資期間全体の分散であるべきだが、**論文は
            # 意図的に1ステップの履歴分散で代用している**（付録 A:「本モデルは
            # リターン分散の期待形成を規定しない。そのため Vt[ρ_{t+τ}] を式(9) の
            # 履歴分散で置く」）。よって v_horizon=False が論文通り。掛けると
            # πᵢ が τᵢ（〜1500）倍になり、注文が広範囲に散って板が壊れる。
            aV = alpha_i[i] * (Vi * ti if p.v_horizon else Vi)
            p_star = self._solve_p_star(p_hat, aV, S[i])
            p_m = self._solve_p_m(p_hat, aV, S[i], C[i], p_star)
            p_M = p_hat
            if not (p_m < p_M):
                price[t] = price[t - 1]
                continue

            draw = float(rng.uniform(p_m, p_M))
            aq, _ = self._best_ask()
            bq, _ = self._best_bid()
            last = price[t - 1]

            if draw < p_star:                                   # 買い側
                if aq is not None and draw > aq:                # 成行買い
                    vol = self._pi(aq, p_hat, aV) - S[i]
                    last = self._market(i, "B", vol, S, C, t, ti, draw, p_hat, aV)
                else:                                           # 指値買い
                    vol = self._pi(draw, p_hat, aV) - S[i]
                    if vol > 0:
                        self._add_limit("B", self._round(draw), vol, t + ti, i)
            elif draw > p_star:                                 # 売り側
                if bq is not None and draw < bq:                # 成行売り
                    vol = S[i] - self._pi(bq, p_hat, aV)
                    last = self._market(i, "S", vol, S, C, t, ti, draw, p_hat, aV)
                else:                                           # 指値売り
                    vol = S[i] - self._pi(draw, p_hat, aV)
                    if vol > 0:
                        self._add_limit("S", self._round(draw), min(vol, S[i]), t + ti, i)

            if last is None:
                aq2, _ = self._best_ask()
                bq2, _ = self._best_bid()
                last = 0.5 * (aq2 + bq2) if (aq2 is not None and bq2 is not None) else price[t - 1]
            else:
                trades[t] = True
            price[t] = max(last, p.tick)
            logret[t] = math.log(price[t] / price[t - 1])

        fund[0] = p.p_f0
        cut = H + self.warmup
        return {"prices": price[cut:], "fundamental": fund[cut:],
                "returns": logret[cut:],
                "n_trades": int(trades[cut:].sum())}

    def _round(self, x):
        return max(round(x / self.p.tick) * self.p.tick, self.p.tick)

    def _market(self, i, side, vol, S, C, t, ti, draw, p_hat, aV):
        """成行注文を板に当てる。残余は draw 価格の指値になる（論文の規定）。

        **実行可能量を先に確定してから、両者の勘定を同時に動かす。**
        以前は自分側を更新したあとで相手の資力を見て数量を書き換えていたため、
        1722 約定すべてで保存則が破れ、株式が消滅し現金が湧いていた。
        """
        if vol <= 0:
            return None
        last = None
        remaining = vol
        while remaining > 1e-12:
            if side == "B":
                pr, oid = self._best_ask()
                if pr is None or pr > draw:
                    break
            else:
                pr, oid = self._best_bid()
                if pr is None or pr < draw:
                    break
            o = self._orders[oid]
            cp = o[3]

            # 双方の制約を同時に満たす数量を先に決める
            q = min(remaining, o[1])
            if side == "B":
                q = min(q, C[i] / pr if pr > 0 else 0.0, max(S[cp], 0.0))
            else:
                q = min(q, max(S[i], 0.0), C[cp] / pr if pr > 0 else 0.0)

            if q <= 1e-12:
                # この気配とは約定できない（相手に在庫/現金が無い）。板から除く
                self._orders.pop(oid, None)
                continue

            cost = q * pr
            if side == "B":
                S[i] += q
                C[i] -= cost
                S[cp] -= q
                C[cp] += cost
            else:
                S[i] -= q
                C[i] += cost
                S[cp] += q
                C[cp] -= cost

            o[1] -= q
            remaining -= q
            last = pr
            if o[1] <= 1e-12:
                self._orders.pop(oid, None)

        if remaining > 1e-12:
            self._add_limit(side, self._round(draw), remaining, t + ti, i)
        return last
