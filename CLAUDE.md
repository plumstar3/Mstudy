# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

修士研究「大豆収量予測モデルの構築」のリポジトリ。圃場×年（2015〜2018）単位で、気象・土壌・土壌水分（VWC）・NDVI・病害（Harm）・栽培情報から収量を予測/分類する。出発点は CNN-RNN による先行研究（Khaki et al. 2019, Frontiers in Plant Science）。**現在の研究テーマは、土壌水分量(VWC)の時系列から多収・低収圃場を分類し、その精度を上げること。作業は `rocket/` で行う**（詳細は `rocket/CLAUDE.md`）。ユーザーとのやり取り・コメント・出力メッセージは日本語。

テスト・lint・ビルドの仕組みは無い。各スクリプトは単体で実行する解析スクリプト。

## Python 環境（複数あるので注意）

| 環境 | 用途 |
|------|------|
| `C:\Users\amilu\anaconda3\python` | lightgbm / sktime / optuna / sklearn / tensorflow が入っている。ルートの `eval_*.py`・`yield_prediction_v*.py` の実行はこれ |
| `rocket\venv_rocket\` | `rocket/` 専用(sktime 1.2.0 ほか、`rocket/requirements.txt`)。`rocket/scripts/` の実行はこれ |
| `amd\` | README 上のメイン venv（`amd\Scripts\Activate.ps1`）。`requirements.txt` 相当 + mlflow のみで、ML ライブラリは無い |
| `venv_ndvi\` | Google Earth Engine (`ee`) 用。`notebooks/exploration/0x_*ndvi*` 系 |
| `ts2vec\venv_ts2vec\` | PyTorch。TS2Vec 専用 |
| `.venv\` | ほぼ空 |

シェル上の素の `python` は `amd` を指すので、ML 系スクリプトはインタプリタを明示すること。

## よく使うコマンド

```powershell
# 収量回帰（Ridge / LightGBM、CV は kfold か Leave-One-Year-Out）
C:\Users\amilu\anaconda3\python yield_prediction_v3.py --cv-mode loyo

# TS2Vec 2段階パイプライン（ts2vec ディレクトリから実行）
cd ts2vec; .\scripts\run_soybean.ps1

# MLflow 実験（テンプレートをコピーして使う）
Copy-Item -Recurse experiments\template experiments\expXXX_説明
python experiments\expXXX_説明\train.py ; mlflow ui
```

## データとパス

- `data/`・`outputs/`・`*.csv`・`*.db`・`*.png` などは git 管理外（`.gitignore`）。コード以外の成果物はコミットされない。
- 主要データは SQLite:
  - `data/processed/FieldData_fieldid.db` — 圃場データ。テーブル: `Questionaire`（収量・播種日・収穫日・品種・畝間など、圃場×年の静的属性）、`SolidMoisture`（VWC 時系列）、`Solid`（土壌）、`Harm`（病害）、`Nutrition`、`Weather`、`NormalWeather`、`field_master` ほか
  - `data/processed/weather_database_fieldid.db` — `weather_data` テーブル（field_id, date, TMP_mea/max/min, APCPRA, SSD, GSR, WIND, SWE, RH）。1981年〜の長期気象
  - `data/raw/` は元データ（`FieldData.db`, `weather_database.db`, HLS NDVI CSV など）。`*_fieldid.db` は raw に field_id を付与・修正した派生版（`.bak` が残っている）
- 結合キーは **`(field_id, year)`**。地域コード `muniCd` は `outputs/reverse_geocode/field_addresses.csv` から field_id で結合する。
- **ルート直下のスクリプトは相対パス（`data/processed/...`）を使うため、リポジトリルートから実行する前提**。`rocket/scripts/` のスクリプトは `__file__` 基準でパスを解決するのでどこからでも実行できる。
- DB を直接書き換えるスクリプト（`fix_*.py`, `impute_*.py`, `add_solid_date.py` など）があるので、DB を変更する処理は事前にユーザーに確認する。

## コード構成の全体像

研究は試行錯誤的に進んでおり、ルート直下のスクリプトが多い。命名で役割が分かれる:

- `extract_*`, `build_*`, `generate_*`, `impute_*`, `fix_*`, `add_*` — データ構築・修正（DB や `data/processed`、`outputs/` に書き出す）
- `eval_*`, `yield_prediction_v*`, `tune_lgbm.py`, `run_feature_ablation.py` — モデル評価。年をまたぐ汎化を見るため **LOYO（Leave-One-Year-Out）** が標準の CV。結果は `outputs/<実験名>/` へ
- `inspect_*`, `check_*`, `verify_*`, `scratch_*` — 使い捨ての確認用。新たな一時確認も `scratch_` 接頭辞で作ってよい
- `_v2`, `_v3` のような接尾辞は改版。基本的に番号が最大のものが最新
- `notebooks/exploration/NN_*.py` — 通し番号付きの探索分析（多くは `.py`）。`notebooks/tutorials/` は学習用資料で研究コードではない

ディレクトリ:

- `src/` — 初期の CRNN 系コード（`models/crnn*.py` は Keras モデルのバリエーション、`data/` は気象取得・前処理、`utils/amd_tools.py` は農研機構メッシュ農業気象データ API）。現在の主な作業はここではなくルートと `rocket/`
- `experiments/template/` — MLflow 連携の実験テンプレート（`config.yaml` + `train.py` + 実験メモ `README.md`）
- `ts2vec/` — TS2Vec（外部 OSS）に大豆用の loader（`soybean_pretrain` / `soybean_finetune`）と `tasks/yield_regression.py` を追加したもの。1981〜2014 の気象で自己教師あり事前学習 → 2015〜2016 で追加学習し Ridge で収量回帰。入力は `generate_pretrain_dataset_v2.py` が作る `data/processed/soybean_ts2vec*/`
- `rocket/` — **現在の研究の主作業場所**。VWC 時系列から多収/低収を ROCKET で分類する。目的・データ設計・決定済みの評価方針・未決定事項は [rocket/CLAUDE.md](rocket/CLAUDE.md) を参照
