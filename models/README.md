# 論文通りの ABM 実装

**既存の再実装が論文のモデルでないことが分かったので、原論文から起こし直したもの。**

## 実装の状態

「完成／未完成」の一列では足りない。**論文どおりに動くこと**と
**現実に似ていること**は別の問いなので、分けて宣言する（`python3 models/status.py`）。

| | `franke_westerhoff` | `chiarella_iori` |
|---|---|---|
| `specification_conformance` | verified | **partial** |
| `invariant_tests` | pass | pass |
| `paper_replication` | pass | **fail** |
| `empirical_validation` | **fail** | not_run |

上3行が verification、最終行だけが validation である。

**Franke-Westerhoff は論文 Table 2 の報告値8つ全てに一致するのに、実データとの
照合では落ちる**（drift が実データの 9.5 倍）。論文を正しく実装できていることは、
現実に似ていることを意味しない。この2つを1つの「検証済み」に潰してはいけない。

**Chiarella-Iori は paper_replication が fail なので empirical_validation を
走らせていない。**論文を再現できていない実装で現実との一致を測っても、
何を測っているのか分からないからである。

`invariant_tests` は宣言ではなく実行可能な検査である。Chiarella-Iori の
株式・現金の保存則は約定ごとに成立する（相対誤差 0.0、`status.py` が毎回確認する）。

語彙の定義と、宣言と実測が食い違ったときの検出は `models/status.py` にある。

## なぜ作り直したか

`financial-abm-lab` の `abm_models/` にある実装を論文と突き合わせたところ：

- **Franke-Westerhoff** — 切り替え指数に herding 項も misalignment の2乗も無く、
  価格も対数ではなく水準で更新していた。パラメータ名だけが一致していて
  方程式が違う。**公表推定値（MSM で株価指数に当てたもの）は移植できない。**
- **Chiarella-Iori** — **板が存在しない。**best_bid/ask と depth がスカラー4つで、
  集約需要に線形の価格インパクトを掛けているだけ。指値もキューもマッチングも無い。
- Lux-Marchesi と Cont-Bouchaud は忠実だった（前者は論文の Hill α 1.92 対 1.93 を再現）。

**パラメータ名の一致は実装の一致を意味しない。**

## franke_westerhoff.py

Franke & Westerhoff (2009) BERG working paper の TPA 版。式(1)〜(8)、Table 1 の
パラメータをそのまま既定値にしてある。`verify()` が Table 2 と照合する。

実装中に **論文の印字と導出の食い違い**を1つ見つけた。式(5)の分母 `/2` を
外した版が Table 2 を再現する（V 0.711 対 0.70、Hill 3.60 対 3.57）。
経緯はソース中のコメントに書いた。

## chiarella_iori.py

Chiarella, Iori & Perelló (arXiv:0711.3581) の連続ダブルオークション。
期待形成・CARA 需要・注文配置表・指値キュー・マッチング・期限切れまで実装済み。
保存則（株式・現金）は成立している。

**だが論文の挙動を再現しない。**ファンダメンタリスト支配下で論文は
「指値は中値のごく近く」と述べるが、本実装は中央値 117% 離れる。
最有力の容疑は Vᵢ の次元（リターン分散か価格分散か）。詳細は
`CALIBRATION.md` と本ファイルの docstring を参照。
