# 較正装置（v1〜v5）

S&P500 に ABM を較正するための装置と、5世代分の結果。

**外部依存:** `financial-abm-lab`（private）の `packages/abm_models` を import する。
この依存がある部分は、そのままでは動かない。論文通りに起こし直した
`models/` の実装は自己完結しているのでそちらは動く。

| ファイル | 内容 |
|---|---|
| `calibrate.py` | v1。目的関数と実データ側の目標・尺度 |
| `calibrate2.py` | v2。妥当性フィルタ（**このフィルタは誤りだった**） |
| `calibrate3.py` | v3。既定値を初期点に／ゼロは連続長で判定／決定論的な帯を棄却 |
| `calibrate4_ci.py` | v4。Chiarella-Iori を単体パラメータ化して境界を解消 |
| `calibrate5.py` | v5。ドリフト制約を妥当性の門として追加 |
| `calibration_v*.json` | 各世代の較正結果（パラメータ・損失・全統計量） |
| `report*.py` | 世界線の図と証拠表 |
| `run_kronos.py` | Kronos（基盤モデル）駆動の板から系列を作る |

**踏んだ地雷と回避手順は `../CALIBRATION.md` に全部書いた。**

## 重要な但し書き

`calibration_v*.json` の `chiarella_iori` と `franke_westerhoff` は、
**論文のモデルではない実装**（`financial-abm-lab` のアダプタ）を較正した結果である。
論文通りの実装に差し替えた後の較正はまだ走らせていない。
忠実性が確認できているのは `lux_marchesi` と `cont_bouchaud` の2つだけ。
