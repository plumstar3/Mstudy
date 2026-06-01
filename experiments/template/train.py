"""
実験スクリプトテンプレート
このファイルを experiments/expXXX_説明/ にコピーして使用してください。

使い方:
  python experiments/expXXX_説明/train.py

MLflow UI:
  mlflow ui
  → http://localhost:5000 をブラウザで開く
"""

import sys
import os
import yaml
import mlflow
import mlflow.keras

# プロジェクトルートを sys.path に追加（src/ を import できるように）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

# ---- src からモデルをインポート ----
# from src.models.crnn import build_model  # 実験に合わせて変更

# ============================================================
# 設定読み込み
# ============================================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ============================================================
# MLflow 実験開始
# ============================================================
mlflow.set_tracking_uri(os.path.join(ROOT, "mlruns"))
mlflow.set_experiment("soybean-yield-prediction")  # 実験グループ名

with mlflow.start_run(run_name=cfg["run_name"]):

    # ---- パラメータを記録 ----
    mlflow.log_params(cfg["hyperparams"])
    mlflow.log_params({
        "model_type": cfg["model"]["type"],
        "lstm_units": cfg["model"]["lstm_units"],
        "conv_filters": cfg["model"]["conv_filters"],
    })

    # ===========================================================
    # ここに実験コードを書く
    # ===========================================================

    # --- データ読み込み ---
    # X_train, y_train, X_val, y_val = load_data(cfg["data"])

    # --- モデル構築 ---
    # model = build_model(cfg["model"])

    # --- 学習 ---
    # history = model.fit(
    #     X_train, y_train,
    #     epochs=cfg["hyperparams"]["epochs"],
    #     batch_size=cfg["hyperparams"]["batch_size"],
    #     validation_data=(X_val, y_val),
    #     callbacks=[...],
    # )

    # --- メトリクスを記録 ---
    # for epoch, (loss, val_loss) in enumerate(
    #     zip(history.history["loss"], history.history["val_loss"])
    # ):
    #     mlflow.log_metric("loss",     loss,     step=epoch)
    #     mlflow.log_metric("val_loss", val_loss, step=epoch)

    # --- モデルを保存 ---
    # model_path = os.path.join(cfg["output"]["model_dir"], f"{cfg['run_name']}.keras")
    # model.save(model_path)
    # mlflow.log_artifact(model_path, artifact_path="model")

    print(f"[OK] 実験 '{cfg['run_name']}' が完了しました。")
    print("     MLflow UI: mlflow ui → http://localhost:5000")
