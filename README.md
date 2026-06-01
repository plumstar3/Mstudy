<img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=for-the-badge&logoColor=white">
<img src="https://img.shields.io/badge/-MLflow-0194E2.svg?logo=mlflow&style=for-the-badge&logoColor=white">

# 大豆収量予測モデルの構築

M1から作成したコードはgit上で保存しています。B4での作成したファイルは載っていません。

:compass: この研究は、こちらの<a href="https://www.frontiersin.org/articles/10.3389/fpls.2019.01750/full" target="_blank">"先行研究"</a>が元になっています。また、<a href="https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction" target="_blank">"こちら"</a>が先行研究のオープンソースコードになります。

---

## ディレクトリ構成

```
Mstudy/
├── src/                    # ソースコード（Gitで管理）
│   ├── models/             # CRNNモデル定義
│   ├── data/               # データ取得・前処理スクリプト
│   └── utils/              # 共通ユーティリティ（AMD_Tools等）
├── experiments/            # 実験管理（MLflow連携）
│   └── template/           # 新実験作成用テンプレート
├── notebooks/              # Jupyter Notebook
│   ├── exploration/        # データ探索
│   └── tutorials/          # 学習資料
├── data/                   # データ（Gitで管理しない）
│   ├── raw/                # 生データ（weather_database.db等）
│   └── processed/          # 前処理済みデータ
├── outputs/                # 実験出力（Gitで管理しない）
│   ├── models/             # 学習済みモデル（*.keras）
│   └── predictions/        # 予測結果（*.npz）
├── requirements.txt        # 依存パッケージ
└── README.md
```

---

## セットアップ

```powershell
# 仮想環境を有効化
amd\Scripts\Activate.ps1

# パッケージをインストール
pip install -r requirements.txt

# MLflow もインストール（未導入の場合）
pip install mlflow
```

---

## 実験の実行

```powershell
# テンプレートをコピーして新実験を作成
Copy-Item -Recurse experiments\template experiments\expXXX_説明

# config.yaml を編集して実験設定を変更後、学習を実行
python experiments\expXXX_説明\train.py

# MLflow UI で実験結果を確認
mlflow ui
# → http://localhost:5000 をブラウザで開く
```

---

## モデル説明

| ファイル | 説明 |
|---------|------|
| `src/models/crnn.py` | 基本CRNNモデル（先行研究ベース）|
| `src/models/crnn_weather.py` | 気象データのみ使用 |
| `src/models/crnn_soil.py` | 土壌データ組み合わせ |
| `src/models/crnn_p.py` | 降水量特化 |
| `src/models/crnn_w.py` | 気象データ拡張版 |
| `src/models/crnn_tr.py` | 転移学習版 |
| `src/models/crnn_my.py` | カスタム改良版 |