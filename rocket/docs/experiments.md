# 実験記録

VWC 時系列による多収・低収分類の実験を、1実験1項目で新しいものを上に追記する。
ID は `EXP-連番`。結果の数値は CV の各 fold の値と平均±標準偏差を書く。

## 一覧

| ID | 日付 | 概要 | ラベル | 主指標 | 結果 |
|---|---|---|---|---|---|
| EXP-000 | 2026-10 | ROCKET デモ(ArrowHead) | - | Accuracy | 0.977(非公式分割) |

---

<!-- ここから下にコピーして使う
## EXP-XXX: <一言で>

- 日付: YYYY-MM-DD
- スクリプト: `scripts/xxx.py`(実行コマンドも書く)
- 目的・仮説: <何を確かめるか。前の実験から何を変えたか>

### 設定
| 項目 | 値 |
|---|---|
| データ | `field_year_labeled.csv`(N 件、年: ) |
| ラベル | `label_muniyear` / `label_year` |
| 除外 | <除外した圃場×年と基準> |
| 系列の揃え方 | <日付 / dap、期間> |
| 入力の形 | <生 VWC / 圃場ごとに標準化 / 差分> |
| モデル | <例: Rocket(num_kernels=10000) → StandardScaler(with_mean=False) → RidgeClassifierCV> |
| CV | <例: GroupKFold(n_splits=5, groups=muniCd)> |
| 乱数シード | |

### 結果
| fold | テスト地域数 | 件数 | Accuracy | Balanced Acc | AUC | F1(macro) |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 平均±SD | | | | | | |

- 出力: `outputs/<dir>/`

### 考察
-

### 次にやること
-
-->

## EXP-000: ROCKET デモ(ArrowHead)

- 日付: 2026-10
- スクリプト: `scripts/rocket_demo.py`
- 目的: sktime の ROCKET が動くことを確認する

### 設定
| 項目 | 値 |
|---|---|
| データ | sktime `load_arrow_head`(211本、1変量、長さ251、3クラス) |
| モデル | `RocketClassifier(rocket_transform="rocket", num_kernels=10000, random_state=42)` |
| 分割 | 全体を 8:2 に再分割(`train_test_split(test_size=0.2, random_state=42)`、stratify なし) |

### 結果
- Accuracy 0.977

### 考察
- 動作確認としては十分。ただし公式分割(学習36 / テスト175)ではないので、論文値とは比べられない

### 次にやること
- 公式分割(`split="train"` / `split="test"`)に直す
