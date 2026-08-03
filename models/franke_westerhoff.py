"""Franke & Westerhoff の構造的確率ボラティリティ・モデル（TPA）を論文通りに実装する。

出典
    Franke, R. & Westerhoff, F. (2009).
    "Validation of a Structural Stochastic Volatility Model of Asset Pricing."
    BERG / University of Bamberg working paper.
    https://www.uni-bamberg.de/fileadmin/uni/fakultaeten/sowi_lehrstuehle/
    vwl_wirtschaftspolitik/Team/Westerhoff/Working_Papers/2009/2009_WP_Franke_Westerhoff.pdf

**リポジトリの `abm_models/franke_westerhoff` はこのモデルではない。**
名前とパラメータ記号は重なるが、切り替え指数に herding 項も misalignment の2乗も
無く、価格も対数ではなく水準で更新していた。公表推定値は移植できない。

論文の式（節番号は上記 working paper）:

  需要（式1, 2）      d^f_t = φ (p* − p_t) + ε^f_t,   ε^f ~ N(0, σ_f²)
                     d^c_t = χ (p_t − p_{t−1}) + ε^c_t, ε^c ~ N(0, σ_c²)
  多数派指数（式3）    n^f/2N = (1+x_t)/2,  n^c/2N = (1−x_t)/2
  価格（式4, 5）      p_{t+1} = p_t + (μ/2)[(1+x_t)φ(p*−p_t)
                                          + (1−x_t)χ(p_t−p_{t−1}) + ε_t]
                     ε_t ~ N(0, σ_t²),  σ_t² = [(1+x_t)²σ_f² + (1−x_t)²σ_c²]/2
  多数派の遷移（式6）  x_{t+1} = x_t + (1−x_t)π^cf_t − (1+x_t)π^fc_t
  遷移確率（式7）      π^cf = ν exp(s_t),  π^fc = ν exp(−s_t)
  切り替え指数（式8）  s_t = α_o + α_x x_t + α_d (p_t − p*)²
  リターン（式10）     r_t = 100 (p_t − p_{t−1})     ※パーセントポイント

Table 1（株式市場シナリオの数値パラメータ）をそのまま既定値にしてある。

検証は Table 2 の報告値に対して行う（`verify()`）:
    Hill H = 3.57、V（絶対リターンの平均）= 0.70
    |r| の ACF（3ラグ中心移動平均）lag 1/5/10/25/50/100 = 0.16/0.16/0.15/0.13/0.10/0.05
    シミュレーション長は実データの10倍、68,670 日。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FWParams:
    """Table 1: Numerical parameter set for the stock market scenario."""

    phi: float = 0.18      # ファンダメンタリストの積極性
    chi: float = 2.35      # チャーティストの積極性
    sigma_f: float = 0.79  # ファンダメンタリスト需要のノイズ
    sigma_c: float = 1.91  # チャーティスト需要のノイズ
    mu: float = 0.01       # 需要の市場インパクト係数
    p_star: float = 0.00   # ファンダメンタル価値の対数
    nu: float = 0.57       # population dynamics の柔軟性
    alpha_o: float = -0.15   # predisposition
    alpha_x: float = 1.35    # herding
    alpha_d: float = 11.40   # dispersion


class FrankeWesterhoffTPA:
    """TPA（transition probability approach）版。seed → リターン系列。"""

    name = "franke_westerhoff_tpa"

    def __init__(self, n_steps: int = 1000, burn_in: int = 1000,
                 params: FWParams | None = None):
        self.n_steps = int(n_steps)
        self.burn_in = int(burn_in)
        self.params = params or FWParams()

    def run(self, *, seed: int) -> dict:
        p = self.params
        rng = np.random.default_rng(seed)
        T = self.burn_in + self.n_steps + 1

        price = np.empty(T)
        x = np.empty(T)
        price[0] = price[1] = p.p_star
        x[0] = x[1] = 0.0

        for t in range(1, T - 1):
            # --- 切り替え指数と遷移確率（式7, 8） ---
            s = p.alpha_o + p.alpha_x * x[t] + p.alpha_d * (price[t] - p.p_star) ** 2
            pi_cf = p.nu * np.exp(s)
            pi_fc = p.nu * np.exp(-s)
            # 確率なので [0,1] を超えさせない（論文脚注14：数値実験では上限に達しない）
            pi_cf = min(pi_cf, 1.0)
            pi_fc = min(pi_fc, 1.0)

            # --- 多数派指数の更新（式6）。x は同期間に既定で、翌期に反映される ---
            x_next = x[t] + (1.0 - x[t]) * pi_cf - (1.0 + x[t]) * pi_fc
            x[t + 1] = float(np.clip(x_next, -1.0, 1.0))

            # --- 構造的確率ボラティリティ（式5）---
            #
            # **印字と導出が食い違う箇所。**working paper の式(5)は
            #     σ_t² = [(1+x)²σ_f² + (1−x)²σ_c²] / 2
            # と印字されているが、式(4)の括弧内は「平均需要の2倍」なので
            # （(1±x)/2 ではなく (1±x) が掛かっている）、そこに入るノイズの
            # 分散は平均需要のノイズ分散の4倍になり、分母は消える：
            #     Var[2·((1+x)/2·ε^f + (1−x)/2·ε^c)] = (1+x)²σ_f² + (1−x)²σ_c²
            #
            # 論文 Table 2 の報告値で両方を試したところ、分母なしが一致した：
            #     /2 あり  → V=0.531, Hill=3.77
            #     分母なし → V=0.711, Hill=3.60
            #     論文 sim → V=0.70,  Hill=3.57
            # よって導出側（分母なし）を採る。
            var = ((1.0 + x[t]) ** 2 * p.sigma_f ** 2
                   + (1.0 - x[t]) ** 2 * p.sigma_c ** 2)
            eps = rng.normal(0.0, np.sqrt(var))

            # --- 価格（式4）---
            core = ((1.0 + x[t]) * p.phi * (p.p_star - price[t])
                    + (1.0 - x[t]) * p.chi * (price[t] - price[t - 1])
                    + eps)
            price[t + 1] = price[t] + (p.mu / 2.0) * core

        r = 100.0 * np.diff(price)          # 式10：パーセントポイント
        return {"returns": r[self.burn_in:], "prices": price[self.burn_in:],
                "x": x[self.burn_in:]}


# ---------------------------------------------------------------- 検証

def _acf(x, lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[lag:] * x[:-lag]).sum() / d) if d > 0 else np.nan


def _acf_smoothed(x, lag):
    """論文脚注20：ラグ τ の値は τ−1, τ, τ+1 の平均（τ=1 は最初の2つの平均）。"""
    if lag == 1:
        return float(np.mean([_acf(x, 1), _acf(x, 2)]))
    return float(np.mean([_acf(x, lag - 1), _acf(x, lag), _acf(x, lag + 1)]))


def _hill(x, frac=0.05):
    """論文と同じく、絶対リターンの上位5%に対する Hill 推定量。"""
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    k = max(int(len(a) * frac), 10)
    if len(a) <= k or a[k] <= 0:
        return np.nan
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


PAPER = {"H": 3.57, "V": 0.70,
         "acf": {1: 0.16, 5: 0.16, 10: 0.15, 25: 0.13, 50: 0.10, 100: 0.05}}
EMPIRICAL = {"H": 3.32, "V": 0.71,
             "acf": {1: 0.19, 5: 0.19, 10: 0.16, 25: 0.13, 50: 0.11, 100: 0.07}}


def verify(seed: int = 0, n_steps: int = 68670):
    """論文 Table 2 と突き合わせる。長さも論文と同じ 68,670 日にする。"""
    m = FrankeWesterhoffTPA(n_steps=n_steps, burn_in=2000)
    r = m.run(seed=seed)["returns"]
    a = np.abs(r)
    got = {"H": _hill(r), "V": float(a.mean()),
           "acf": {l: _acf_smoothed(a, l) for l in PAPER["acf"]}}
    print(f"Franke-Westerhoff (2009) Table 2 との照合   n={len(r)}  seed={seed}\n")
    print(f"{'':<10}{'本実装':>10}{'論文 sim':>10}{'差':>9}{'S&P500':>10}")
    print("-" * 50)
    print(f"{'Hill H':<10}{got['H']:>10.2f}{PAPER['H']:>10.2f}"
          f"{got['H']-PAPER['H']:>+9.2f}{EMPIRICAL['H']:>10.2f}")
    print(f"{'V':<10}{got['V']:>10.2f}{PAPER['V']:>10.2f}"
          f"{got['V']-PAPER['V']:>+9.2f}{EMPIRICAL['V']:>10.2f}")
    for l in PAPER["acf"]:
        print(f"{'acf|r| '+str(l):<10}{got['acf'][l]:>10.2f}{PAPER['acf'][l]:>10.2f}"
              f"{got['acf'][l]-PAPER['acf'][l]:>+9.2f}{EMPIRICAL['acf'][l]:>10.2f}")
    return got


if __name__ == "__main__":
    verify()
