"""
yield_prediction.py
===================
ts2vec の soybean_finetune ローダーと同一のデータセット・同一の年度分割を使用して、
複数の回帰モデルにより収量を予測するスタンドアロンスクリプト。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【データセット】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  データパス : data/processed/soybean_ts2vec/
    X.npy    : (N, 366, 9)  気象時系列（うるう年に合わせて T=366 で NaN パディング）
               インデックス 0 = 1月1日、1 = 1月2日、... と 0-based 通し連番
               気象変数 9 列: TMP_mea, TMP_max, TMP_min, APCP, SSD, GSR, SD, SWE, SFW
    y.npy    : (N,)          収量ラベル（kg/10a など）
    meta.csv : field_id, year, yield のメタデータ

  年度分割（ts2vec 推論と同一）:
    Train : 2015, 2016
    Val   : 2017
    Test  : 2018

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【標準化（normalize 関数）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  X.npy を読み込んだ後、モデルに渡す前に気象変数ごと（F 次元方向）に Z-score 標準化を
  実施します。統計量（mean, std）は **Train split のみ** から計算し、Val・Test には
  同じ統計量を適用します（データ漏洩を防ぐため）。NaN 値は標準化後に 0 で置換します
  （これは標準化後の平均値に相当します）。

  具体的には:
    X_norm[i, t, f] = (X[i, t, f] - mean_train[f]) / std_train[f]
    NaN → 0

  この処理は ts2vec/datautils.py の _normalize_soybean() と同一です。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【対象期間の絞り込み（--crop-window）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  X.npy の時間次元はすべて 1月1日（index=0）始まりで格納されています。
  --crop-window may1_dec27 を指定すると、年ごとに以下のインデックス範囲を切り出します:

    非閏年 (2015, 2017, 2018): index 120 (5/1) 〜 360 (12/27) → 241 日
    閏年   (2016)             : index 121 (5/1) 〜 361 (12/27) → 241 日

  全年度で切り出し後の長さが 241 日に揃うため、サンプル間で形状が統一されます。

  --crop-window full_year（デフォルト）を指定すると、従来通り 1/1〜12/31 (366 日) を使用。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【特徴量エンジニアリング（--feature-type / extract_features 関数）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  標準化・期間絞り込み後の (N, T, 9) 配列から、モデルに渡す (N, D) の固定次元
  特徴量ベクトルを作成します。以下の 3 種類から選択できます。

  1. flatten (次元数: T * 9)
     各日・各気象変数の値をそのまま 1 次元に並べます。
     時間軸の細かいパターンを保持しますが、次元数が最大で 366*9=3294 になります。
     特徴量間の相関が高く、Ridge では強い L2 正則化（大きな alpha）が有効です。

  2. timemean (次元数: 9)
     時間軸方向に平均を取り、9 次元の特徴量にします。
     「栽培期間全体の平均気象条件」を表します。
     非常にコンパクトですが、時間的なパターンの情報が失われます。

  3. timestats (次元数: 9 * 5 = 45)
     時間軸方向に mean, std, min, max, median の 5 統計量を計算し結合します。
     平均的な気象条件に加え、変動性・極値も捉えられます。
     次元数・表現力のバランスが良く、デフォルトとして推奨します。

  4. period3mean (次元数: 9 * 3 = 27)  ★新規
     栽培期間（241日）を 3 等分した各サブ期間の変数ごと平均を結合します。
       P1 (day  0- 79, 5/1 〜 7/19頃) : 播種・発芽期
       P2 (day 80-159, 7/20〜10/6頃) : 生育・開花期
       P3 (day160-240, 10/7〜12/27) : 登熟・収穫期
     timestats（45次元）より低次元で、かつ生育フェーズごとの気象差を表現できます。
     N/D 比が timemean に次いで高く、次元の呪いを回避しやすいです。

  5. period3stats (次元数: 9 * 3 * 5 = 135)  ★新規
     上記 3 期間それぞれに mean/std/min/max/median の 5 統計量を計算し結合します。
     最も情報量が多い一方で次元数も最大です。

  ※ SVR のみ、特徴量次元ごとに StandardScaler（ゼロ平均・単位分散）を追加適用します。
     他モデルは normalize() による標準化のみを使用します。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【サポートモデル】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ridge  : Ridge 回帰（alpha をグリッドサーチ）
  rf     : Random Forest（n_estimators, max_features をグリッドサーチ）
  lgbm   : LightGBM（n_estimators, learning_rate, num_leaves をグリッドサーチ）
  xgb    : XGBoost（n_estimators, learning_rate, max_depth をグリッドサーチ）
  svr    : SVR/RBF（C, epsilon をグリッドサーチ、特徴量に StandardScaler 追加）
  all    : 上記全モデルを実行して比較

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【収量グループ分割（--yield-split）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --yield-split を指定すると、Train の収量 y の平均値を閾値として
  データを 2 グループに分割し、それぞれで独立したモデルを学習・評価します。

    High グループ : y >= threshold  （平均以上の高収量サンプル）
    Low  グループ : y <  threshold  （平均未満の低収量サンプル）

  閾値（threshold）は **Train split の y の平均値のみ** から計算します。
  Val・Test の振り分けも同じ閾値を使用します（データ漏洩防止）。

  実行結果として、High/Low それぞれと、比較用の全データ（all）の
  3 セットの評価が出力されます。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage:
  python yield_prediction.py [options]

  主要オプション:
    --model {ridge,rf,lgbm,xgb,svr,all}
    --feature-type {flatten,timemean,timestats}
    --crop-window {full_year,may1_dec27}   # 栽培期間（5/1-12/27）に絞る
    --yield-split                          # 収量平均値でグループ分割して個別予測
    --all-features                         # 全特徴量タイプで比較実行
    --dataset-dir DATA_DIR
    --output-dir OUTPUT_DIR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import datetime
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ── 定数 ──────────────────────────────────────────────────────────────────────

DEFAULT_DATASET_DIR = os.path.join('data', 'processed', 'soybean_ts2vec')
DEFAULT_OUTPUT_DIR  = os.path.join('outputs', 'yield_pred')
DEFAULT_DB_PATH     = os.path.join('data', 'processed', 'FieldData_fieldid.db')
MAX_SAMPLES = 100_000

# 気象変数名（X.npy の特徴量軸の順番）
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD', 'SWE', 'SFW']

# モデル別ハイパーパラメータ候補
RIDGE_ALPHAS         = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
RF_N_ESTIMATORS      = [50, 100, 200]
RF_MAX_FEATURES      = ['sqrt', 'log2', 1.0]
LGBM_N_ESTIMATORS    = [50, 100, 200, 300]
LGBM_LEARNING_RATES  = [0.01, 0.05, 0.1, 0.2]
LGBM_NUM_LEAVES      = [15, 31, 63]
XGB_N_ESTIMATORS     = [50, 100, 200, 300]
XGB_LEARNING_RATES   = [0.01, 0.05, 0.1, 0.2]
XGB_MAX_DEPTHS       = [3, 5, 7]
SVR_C_LIST           = [0.1, 1, 10, 100]
SVR_EPSILON_LIST     = [0.01, 0.1, 0.5, 1.0]


# ── オプション依存モジュールの lazy import ────────────────────────────────────

def _import_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError("LightGBM が見つかりません。`pip install lightgbm` でインストールしてください。")


def _import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        raise ImportError("XGBoost が見つかりません。`pip install xgboost` でインストールしてください。")


# ── 評価スコア（小さいほど良い：val 選択用）─────────────────────────────────────

def _val_score(pred, target):
    return (np.sqrt(((pred - target) ** 2).mean()) +
            np.abs(pred - target).mean())


# ── 対象期間の絞り込み ────────────────────────────────────────────────────────

def _is_leap(year):
    """閏年判定。"""
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def _day_index(year, month, day):
    """指定した年の month/day が 1/1 を 0 とした何日目（0-based）かを返す。"""
    jan1  = datetime.date(year, 1, 1)
    target = datetime.date(year, month, day)
    return (target - jan1).days  # 0-based


def crop_to_window(X_all, years_all, crop_window):
    """(N, T, F) の X を指定した期間に切り出して (N, T_new, F) を返す。

    crop_window:
      'full_year'   : 1/1 〜 12/31 をそのまま使用（T_new = 366）
      'may1_dec27'  : 5/1 〜 12/27 の 241 日間を切り出す

    年ごとにうるう年の有無を考慮して開始インデックスを決定するため、
    異なる年が混在するケースでも正しく動作します。
    """
    if crop_window == 'full_year':
        return X_all  # 変更なし

    if crop_window == 'may1_dec27':
        T_new = 241  # 全年度で切り出し後は 241 日に統一（うるう年・非閏年共通）
        result = np.zeros((X_all.shape[0], T_new, X_all.shape[2]),
                          dtype=X_all.dtype)
        for i, year in enumerate(years_all):
            start = _day_index(int(year), 5, 1)   # 5/1 の 0-based index
            end   = _day_index(int(year), 12, 27)  # 12/27 の 0-based index
            # end+1 まで = 12/27 を含む 241 日
            result[i] = X_all[i, start:end + 1, :]
        return result

    raise ValueError(f"Unknown crop_window '{crop_window}'. "
                     "Choose 'full_year' or 'may1_dec27'.")


# ── 標準化 ────────────────────────────────────────────────────────────────────

def normalize(X_train, X_val, X_test):
    """Train split の統計で (N, T, F) データを特徴量ごとに Z-score 標準化。

    【処理内容】
      - Train split 全サンプル・全タイムステップから特徴量ごとの mean / std を計算
      - 同じ統計量を Val / Test にも適用（データ漏洩防止）
      - 標準化後の NaN を 0（= 標準化後の平均値）で置換

    Returns:
        X_train_norm, X_val_norm, X_test_norm : float32 の正規化済み配列
        mean (np.ndarray) : shape (F,) — 各特徴量の平均
        std  (np.ndarray) : shape (F,) — 各特徴量の標準偏差
    """
    flat = X_train.reshape(-1, X_train.shape[-1])   # (N_train * T, F)
    mean = np.nanmean(flat, axis=0)                  # (F,)
    std  = np.nanstd(flat, axis=0)
    std[std < 1e-8] = 1.0                            # ゼロ除算防止

    def _apply(X):
        orig = X.shape
        X2d  = (X.reshape(-1, X.shape[-1]) - mean) / std
        return np.nan_to_num(X2d, nan=0.0).reshape(orig).astype(np.float32)

    return _apply(X_train), _apply(X_val), _apply(X_test), mean, std


# ── 特徴量エンジニアリング ────────────────────────────────────────────────────

def extract_features(X, feature_type):
    """(N, T, F) → (N, D) の特徴量に変換。

    Args:
        X            (np.ndarray): shape (N, T, F)  標準化・期間絞り込み済みの配列
        feature_type (str)       : 'flatten' | 'timemean' | 'timestats'
                                   | 'period3mean' | 'period3stats'

    Returns:
        np.ndarray: shape (N, D)

    特徴量の詳細はモジュール docstring を参照。
    """
    if feature_type == 'flatten':
        # 各日×各変数の値をそのまま 1 次元に展開 → D = T * F
        return X.reshape(X.shape[0], -1)

    elif feature_type == 'timemean':
        # 時間軸方向の平均 → D = F (= 9)
        return np.mean(X, axis=1)

    elif feature_type == 'timestats':
        # 時間軸方向の 5 統計量（mean/std/min/max/median）を結合 → D = F*5 (= 45)
        mean   = np.mean(X, axis=1)
        std    = np.std(X, axis=1)
        vmin   = np.min(X, axis=1)
        vmax   = np.max(X, axis=1)
        median = np.median(X, axis=1)
        return np.concatenate([mean, std, vmin, vmax, median], axis=1)

    elif feature_type in ('period3mean', 'period3stats'):
        # 時間軸を 3 等分してサブ期間ごとに統計量を計算
        # 241 日を [0:80], [80:160], [160:241] に分割（80, 80, 81 日）
        T = X.shape[1]
        p1_end = T // 3            # 80
        p2_end = (T * 2) // 3     # 160
        periods = [
            X[:, :p1_end,  :],    # P1: day  0- 79  播種・発芽期
            X[:, p1_end:p2_end, :],  # P2: day 80-159  生育・開花期
            X[:, p2_end:, :],     # P3: day160-end  登熟・収穫期
        ]

        if feature_type == 'period3mean':
            # 各期間の平均のみ → D = F * 3 (= 27)
            parts = [np.mean(p, axis=1) for p in periods]
            return np.concatenate(parts, axis=1)

        else:  # period3stats
            # 各期間に 5 統計量 → D = F * 3 * 5 (= 135)
            parts = []
            for p in periods:
                parts += [
                    np.mean(p, axis=1),
                    np.std(p,  axis=1),
                    np.min(p,  axis=1),
                    np.max(p,  axis=1),
                    np.median(p, axis=1),
                ]
            return np.concatenate(parts, axis=1)

    else:
        raise ValueError(
            f"Unknown feature_type '{feature_type}'. "
            "Choose 'flatten', 'timemean', 'timestats', "
            "'period3mean', or 'period3stats'.")


# ── データロード ───────────────────────────────────────────────────────────────

def load_geo_features(db_path, meta_train, meta_val, meta_test):
    """DBから緒度・経度を取得し、Train 統計で標準化して (N, 2) 配列を返す。

    特徴量の内訳:
      - lat (緒度) ・ lon (経度) の 2 列
      - 圆周期のない平坦コーディネートとして使用
      - 標準化は Train の mean/std のみで計算（データ漏洩防止）
      - DB に lat/lon がない場合は 0 で埋める

    Returns:
        geo_train, geo_val, geo_test : shape (N, 2) float32
        geo_stats (dict)             : {'mean': ..., 'std': ...}
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    geo_df = pd.read_sql(
        'SELECT field_id, lat, lon FROM Questionaire '
        'WHERE field_id IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL',
        conn
    )
    conn.close()
    geo_df['field_id'] = geo_df['field_id'].astype(int)
    # field_id ごとに lat/lon を平均（複数年で同じはずだが念のため）
    geo_df = geo_df.groupby('field_id')[['lat', 'lon']].mean().reset_index()

    def _merge_geo(meta):
        fid = meta['field_id'].astype(int)
        merged = pd.DataFrame({'field_id': fid}).merge(
            geo_df, on='field_id', how='left'
        )
        return merged[['lat', 'lon']].to_numpy(dtype=np.float32)

    geo_train_raw = _merge_geo(meta_train)
    geo_val_raw   = _merge_geo(meta_val)
    geo_test_raw  = _merge_geo(meta_test)

    # Train 統計で Z-score 標準化（NaN → 0 = 平均値）
    mean = np.nanmean(geo_train_raw, axis=0)   # (2,)
    std  = np.nanstd(geo_train_raw,  axis=0)
    std[std < 1e-8] = 1.0

    def _norm(arr):
        return np.nan_to_num((arr - mean) / std, nan=0.0).astype(np.float32)

    return (_norm(geo_train_raw), _norm(geo_val_raw), _norm(geo_test_raw),
            {'mean': mean, 'std': std})


def _align_geo(geo_full, meta_full, meta_sub):
    """geo_full (N_full, 2) から meta_sub の行に対応する geo 行を抽出して返す。

    meta_full と meta_sub はともに field_id 列を持つ DataFrame。
    meta_sub は meta_full のサブセット（マスク後）であるため、
    meta_sub の field_id + year の組み合わせで meta_full の位置を逆引きする。
    """
    if geo_full is None:
        return None
    # meta_full に連番インデックスを付けて逆引きマップを作成
    meta_full_reset = meta_full.reset_index(drop=True)
    key_to_idx = {
        (int(row['field_id']), int(row['year'])): i
        for i, row in meta_full_reset.iterrows()
    }
    indices = [
        key_to_idx[(int(row['field_id']), int(row['year']))]
        for _, row in meta_sub.iterrows()
    ]
    return geo_full[indices]


def load_soybean(dataset_dir, train_years, val_years, test_years, crop_window):
    """soybean データをロードし、期間絞り込み・標準化まで適用して返す。

    Args:
        dataset_dir  (str)   : X.npy / y.npy / meta.csv が置かれたディレクトリ
        train_years  (list)  : 学習に使う年度リスト（例: [2015, 2016]）
        val_years    (list)  : バリデーションに使う年度リスト
        test_years   (list)  : テストに使う年度リスト
        crop_window  (str)   : 'full_year' または 'may1_dec27'

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test : float32 配列
        meta_train, meta_val, meta_test                 : pd.DataFrame
        norm_stats (dict)                               : {'mean': ..., 'std': ...}
        crop_info  (dict)                               : 期間絞り込みの詳細情報
    """
    X    = np.load(os.path.join(dataset_dir, 'X.npy'))           # (N, 366, 9)
    y    = np.load(os.path.join(dataset_dir, 'y.npy'))           # (N,)
    meta = pd.read_csv(os.path.join(dataset_dir, 'meta.csv'))

    years = meta['year'].to_numpy().astype(int)
    train_mask = np.isin(years, list(train_years))
    val_mask   = np.isin(years, list(val_years))
    test_mask  = np.isin(years, list(test_years))

    # ── 期間絞り込み（正規化前に実施） ───────────────────────────────────
    X_tr_raw = crop_to_window(X[train_mask], years[train_mask], crop_window)
    X_va_raw = crop_to_window(X[val_mask],   years[val_mask],   crop_window)
    X_te_raw = crop_to_window(X[test_mask],  years[test_mask],  crop_window)

    # 期間情報を表示用にまとめる
    if crop_window == 'may1_dec27':
        example_year = train_years[0]
        s = _day_index(example_year, 5, 1)
        e = _day_index(example_year, 12, 27)
        crop_info = {
            'window': crop_window,
            'T_new': X_tr_raw.shape[1],
            'example': f'{example_year}: index {s}(5/1) - {e}(12/27)',
        }
    else:
        crop_info = {'window': crop_window, 'T_new': X_tr_raw.shape[1]}

    # ── 標準化（Train 統計を使用） ────────────────────────────────────────
    X_train, X_val, X_test, norm_mean, norm_std = normalize(
        X_tr_raw, X_va_raw, X_te_raw)

    return (X_train, y[train_mask].astype(np.float32),
            X_val,   y[val_mask].astype(np.float32),
            X_test,  y[test_mask].astype(np.float32),
            meta[train_mask].reset_index(drop=True),
            meta[val_mask].reset_index(drop=True),
            meta[test_mask].reset_index(drop=True),
            {'mean': norm_mean, 'std': norm_std},
            crop_info)


# ── モデル学習関数 ────────────────────────────────────────────────────────────

def _subsample(X, y):
    if X.shape[0] > MAX_SAMPLES:
        split = train_test_split(X, y, train_size=MAX_SAMPLES, random_state=0)
        return split[0], split[2]
    return X, y


def fit_ridge(train_X, train_y, val_X, val_y):
    """Ridge 回帰（alpha をグリッドサーチ）。"""
    X_s, y_s = _subsample(train_X, train_y)
    scores = []
    for alpha in RIDGE_ALPHAS:
        pred   = Ridge(alpha=alpha).fit(X_s, y_s).predict(val_X)
        scores.append(_val_score(pred, val_y))
    best_alpha = RIDGE_ALPHAS[int(np.argmin(scores))]
    model = Ridge(alpha=best_alpha).fit(X_s, y_s)
    return model, {'best_alpha': best_alpha}


def fit_rf(train_X, train_y, val_X, val_y):
    """Random Forest（n_estimators, max_features をグリッドサーチ）。"""
    X_s, y_s = _subsample(train_X, train_y)
    best_score, best_params, best_model = float('inf'), {}, None
    for n_est in RF_N_ESTIMATORS:
        for mf in RF_MAX_FEATURES:
            rf    = RandomForestRegressor(n_estimators=n_est, max_features=mf,
                                          random_state=0, n_jobs=-1).fit(X_s, y_s)
            score = _val_score(rf.predict(val_X), val_y)
            if score < best_score:
                best_score  = score
                best_params = {'n_estimators': n_est, 'max_features': mf}
                best_model  = rf
    return best_model, best_params


def fit_lgbm(train_X, train_y, val_X, val_y):
    """LightGBM（n_estimators, learning_rate, num_leaves をグリッドサーチ）。"""
    lgb = _import_lgbm()
    X_s, y_s = _subsample(train_X, train_y)
    best_score, best_params, best_model = float('inf'), {}, None
    for n_est in LGBM_N_ESTIMATORS:
        for lr in LGBM_LEARNING_RATES:
            for nl in LGBM_NUM_LEAVES:
                m = lgb.LGBMRegressor(
                    n_estimators=n_est, learning_rate=lr, num_leaves=nl,
                    random_state=0, n_jobs=-1, verbose=-1
                ).fit(X_s, y_s)
                score = _val_score(m.predict(val_X), val_y)
                if score < best_score:
                    best_score  = score
                    best_params = {'n_estimators': n_est,
                                   'learning_rate': lr,
                                   'num_leaves': nl}
                    best_model  = m
    return best_model, best_params


def fit_xgb(train_X, train_y, val_X, val_y):
    """XGBoost（n_estimators, learning_rate, max_depth をグリッドサーチ）。"""
    xgb = _import_xgb()
    X_s, y_s = _subsample(train_X, train_y)
    best_score, best_params, best_model = float('inf'), {}, None
    for n_est in XGB_N_ESTIMATORS:
        for lr in XGB_LEARNING_RATES:
            for md in XGB_MAX_DEPTHS:
                m = xgb.XGBRegressor(
                    n_estimators=n_est, learning_rate=lr, max_depth=md,
                    random_state=0, n_jobs=-1, verbosity=0
                ).fit(X_s, y_s)
                score = _val_score(m.predict(val_X), val_y)
                if score < best_score:
                    best_score  = score
                    best_params = {'n_estimators': n_est,
                                   'learning_rate': lr,
                                   'max_depth': md}
                    best_model  = m
    return best_model, best_params


def fit_svr(train_X, train_y, val_X, val_y):
    """SVR（C, epsilon をグリッドサーチ、特徴量を StandardScaler で再スケール）。"""
    X_s, y_s = _subsample(train_X, train_y)
    # SVR は特徴量スケールに敏感 → StandardScaler を適用
    scaler    = StandardScaler().fit(X_s)
    X_s_sc    = scaler.transform(X_s)
    val_X_sc  = scaler.transform(val_X)
    best_score, best_params, best_model = float('inf'), {}, None
    for C in SVR_C_LIST:
        for eps in SVR_EPSILON_LIST:
            m     = SVR(C=C, epsilon=eps, kernel='rbf').fit(X_s_sc, y_s)
            score = _val_score(m.predict(val_X_sc), val_y)
            if score < best_score:
                best_score  = score
                best_params = {'C': C, 'epsilon': eps}
                best_model  = m
    # scaler をモデルにバインドして predict 時に透過的に使えるようにする
    best_model._yield_pred_scaler = scaler
    return best_model, best_params


def _predict_with_scaler(model, X):
    """SVR の場合はバインドされた scaler を使って変換してから予測。"""
    if hasattr(model, '_yield_pred_scaler'):
        X = model._yield_pred_scaler.transform(X)
    return model.predict(X)


MODEL_REGISTRY = {
    'ridge': fit_ridge,
    'rf':    fit_rf,
    'lgbm':  fit_lgbm,
    'xgb':   fit_xgb,
    'svr':   fit_svr,
}

MODEL_LABELS = {
    'ridge': 'Ridge',
    'rf':    'RandomForest',
    'lgbm':  'LightGBM',
    'xgb':   'XGBoost',
    'svr':   'SVR',
}


# ── 評価指標 ───────────────────────────────────────────────────────────────────

def calc_metrics(pred, target):
    """RMSE / MAE / R2 / MAPE を計算して辞書で返す。"""
    rmse   = float(np.sqrt(((pred - target) ** 2).mean()))
    mae    = float(np.abs(pred - target).mean())
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    nonzero = np.abs(target) > 0
    mape    = (float(np.mean(np.abs((pred[nonzero] - target[nonzero]) /
                                     target[nonzero])) * 100)
               if nonzero.any() else float('nan'))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


# ── 収量グループ分割 ──────────────────────────────────────────────────────────

def split_by_yield_mean(X_train, y_train, X_val, y_val, X_test, y_test,
                        meta_train, meta_val, meta_test):
    """Train の y 平均を閾値にデータを High / Low グループに分割する。

    閾値は Train の y のみから計算（Val/Test に同じ閾値を適用）。

    Returns:
        threshold (float): 分割閾値（Train y の平均値）
        groups (dict): キー = 'high' / 'low'
                       各値 = dict(X_train, y_train, X_val, y_val,
                                   X_test, y_test,
                                   meta_train, meta_val, meta_test,
                                   n_train, n_val, n_test)
    """
    threshold = float(y_train.mean())

    groups = {}
    for label, cond in [('high', lambda y: y >= threshold),
                        ('low',  lambda y: y <  threshold)]:
        tr_mask = cond(y_train)
        va_mask = cond(y_val)
        te_mask = cond(y_test)

        groups[label] = dict(
            X_train   = X_train[tr_mask],
            y_train   = y_train[tr_mask],
            X_val     = X_val[va_mask],
            y_val     = y_val[va_mask],
            X_test    = X_test[te_mask],
            y_test    = y_test[te_mask],
            meta_train = meta_train[tr_mask].reset_index(drop=True),
            meta_val   = meta_val[va_mask].reset_index(drop=True),
            meta_test  = meta_test[te_mask].reset_index(drop=True),
            n_train   = int(tr_mask.sum()),
            n_val     = int(va_mask.sum()),
            n_test    = int(te_mask.sum()),
        )

    return threshold, groups


# ── 結果保存 ───────────────────────────────────────────────────────────────────

def save_results(output_dir, model_name, feature_type, crop_window,
                 train_pred, train_labels,
                 val_pred,   val_labels,
                 test_pred,  test_labels,
                 meta_train, meta_val, meta_test,
                 metrics, best_params, subset='all'):
    """予測値・メトリクスを CSV で保存する。"""
    os.makedirs(output_dir, exist_ok=True)

    def _build_df(meta, labels, pred, split):
        df = meta[['field_id', 'year']].copy().reset_index(drop=True)
        df['yield_true'] = labels
        df['yield_pred'] = pred
        df['split']      = split
        return df

    pred_df = pd.concat([
        _build_df(meta_train, train_labels, train_pred, 'train'),
        _build_df(meta_val,   val_labels,   val_pred,   'val'),
        _build_df(meta_test,  test_labels,  test_pred,  'test'),
    ], ignore_index=True)

    tag      = f'{model_name}_{feature_type}_{crop_window}_{subset}'
    pred_csv = os.path.join(output_dir, f'predictions_{tag}.csv')
    pred_df.to_csv(pred_csv, index=False)
    print(f"  Predictions saved -> {pred_csv}")

    rows = []
    for split, m in metrics.items():
        rows.append({'model': model_name, 'feature_type': feature_type,
                     'crop_window': crop_window, 'subset': subset,
                     'split': split, **best_params, **m})
    metrics_df  = pd.DataFrame(rows)
    metrics_csv = os.path.join(output_dir, f'metrics_{tag}.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  Metrics   saved  -> {metrics_csv}")

    return pred_df, metrics_df


# ── 単一実行 ───────────────────────────────────────────────────────────────────

def run_single(model_name, feature_type, crop_window,
               X_train, y_train, X_val, y_val, X_test, y_test,
               meta_train, meta_val, meta_test, output_dir,
               subset='all',
               geo_train=None, geo_val=None, geo_test=None,
               pca_components=None):
    """1 モデル × 1 特徴量タイプ × 1 期間設定 × 1 サブセットで学習・評価・保存を実行。

    Args:
        subset         : 'all' | 'high' | 'low'  —  収量グループ識別ラベル
        geo_*          : shape (N, 2) の標準化済み lat/lon。None の場合は使用しない。
        pca_components : int   — 主成分数（例: 20）
                         float — 説明分散率（例: 0.95 = 95%）
                         None  — PCA を適用しない
    """
    from sklearn.decomposition import PCA

    print(f"\n{'='*64}")
    print(f"  Model        : {MODEL_LABELS[model_name]}")
    print(f"  Feature type : {feature_type}")
    print(f"  Crop window  : {crop_window}")
    if subset != 'all':
        print(f"  Yield subset : {subset.upper()}  "
              f"(train={len(y_train)} / val={len(y_val)} / test={len(y_test)})")

    Xf_train = extract_features(X_train, feature_type)
    Xf_val   = extract_features(X_val,   feature_type)
    Xf_test  = extract_features(X_test,  feature_type)

    # 緒度・経度を特徴量ベクトルに結合
    if geo_train is not None:
        Xf_train = np.concatenate([Xf_train, geo_train], axis=1)
        Xf_val   = np.concatenate([Xf_val,   geo_val],   axis=1)
        Xf_test  = np.concatenate([Xf_test,  geo_test],  axis=1)
        print(f"  Geo features : lat/lon appended (+2 dims)")

    # ── PCA 次元削減 ──────────────────────────────────────
    if pca_components is not None:
        n_before = Xf_train.shape[1]
        # PCA は Train のみで fit（Val/Test には transform のみ）
        pca = PCA(n_components=pca_components, random_state=0)
        Xf_train = pca.fit_transform(Xf_train)
        Xf_val   = pca.transform(Xf_val)
        Xf_test  = pca.transform(Xf_test)
        n_after      = Xf_train.shape[1]
        var_ratio    = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA          : {n_before}D → {n_after}D  "
              f"(explained variance {var_ratio:.1f}%)")

    print(f"  Feature dims : {Xf_train.shape[1]}")
    print(f"  Samples      : {len(y_train)} / {len(y_val)} / {len(y_test)} "
          f"(train / val / test)")

    fit_fn = MODEL_REGISTRY[model_name]
    t0 = time.time()
    model, best_params = fit_fn(Xf_train, y_train, Xf_val, y_val)
    elapsed = time.time() - t0

    param_str = '  '.join(f'{k}={v}' for k, v in best_params.items())
    print(f"  Best params  : {param_str}")
    print(f"  Train time   : {elapsed:.1f}s")

    train_pred = _predict_with_scaler(model, Xf_train)
    val_pred   = _predict_with_scaler(model, Xf_val)
    test_pred  = _predict_with_scaler(model, Xf_test)

    metrics = {
        'train': calc_metrics(train_pred, y_train),
        'val':   calc_metrics(val_pred,   y_val),
        'test':  calc_metrics(test_pred,  y_test),
    }

    print(f"\n  {'Split':<8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R2':>8}")
    print(f"  {'─'*44}")
    for split in ('train', 'val', 'test'):
        m = metrics[split]
        print(f"  {split:<8} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
              f"{m['MAPE']:>7.2f}% {m['R2']:>8.4f}")

    save_results(
        output_dir, model_name, feature_type, crop_window,
        train_pred, y_train, val_pred, y_val, test_pred, y_test,
        meta_train, meta_val, meta_test,
        metrics, best_params, subset=subset
    )

    return metrics, best_params


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-model yield prediction on soybean dataset '
                    '(same split as ts2vec soybean_finetune)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='ridge',
                   choices=list(MODEL_REGISTRY.keys()) + ['all'],
                   help='Regression model (default: ridge). '
                        '"all" runs every model and prints a comparison.')
    p.add_argument('--feature-type', default='period3mean',
                   choices=['flatten', 'timemean', 'timestats',
                            'period3mean', 'period3stats'],
                   help='Feature engineering method (default: period3mean)')
    p.add_argument('--crop-window', default='may1_dec27',
                   choices=['full_year', 'may1_dec27'],
                   help='Time window to use from X.npy. '
                        '"full_year" = 1/1-12/31 (T=366), '
                        '"may1_dec27" = 5/1-12/27 (T=241, default)')
    p.add_argument('--all-features', action='store_true',
                   help='Run all 3 feature types for each model')
    p.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR,
                   help=f'Path to soybean_ts2vec directory '
                        f'(default: {DEFAULT_DATASET_DIR})')
    p.add_argument('--train-years', nargs='+', type=int,
                   default=[2015, 2016])
    p.add_argument('--val-years',   nargs='+', type=int,
                   default=[2017])
    p.add_argument('--test-years',  nargs='+', type=int,
                   default=[2018])
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                   help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    p.add_argument('--add-geo', action='store_true',
                   help='Append normalized lat/lon from DB as additional '
                        'field features (2 dims added to feature vector)')
    p.add_argument('--db-path', default=DEFAULT_DB_PATH,
                   help=f'Path to FieldData DB for lat/lon '
                        f'(default: {DEFAULT_DB_PATH})')
    p.add_argument('--pca-components', default=None,
                   help='PCA dimensionality reduction applied after feature '
                        'extraction (and geo concat). '
                        'int -> fixed number of components (e.g. 20). '
                        'float 0<v<1 -> explained variance ratio (e.g. 0.95). '
                        'default: None (no PCA)')
    p.add_argument('--yield-split', action='store_true',
                   help='Split data into High/Low groups by train-set mean yield '
                        'and run separate models for each group')
    return p.parse_args()


# ── メイン ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model_names   = list(MODEL_REGISTRY.keys()) if args.model == 'all' \
                    else [args.model]
    feature_types = (
        ['flatten', 'timemean', 'timestats', 'period3mean', 'period3stats']
        if args.all_features else [args.feature_type]
    )

    # --pca-components の型変換（str -> int or float or None）
    pca_components = None
    if args.pca_components is not None:
        try:
            v = float(args.pca_components)
            pca_components = int(v) if v >= 1.0 else v
        except ValueError:
            raise ValueError(f"--pca-components must be int or float, got: {args.pca_components}")

    print("=" * 64)
    print("  Yield Prediction (Multi-model)")
    print("  (Same dataset/split as TS2Vec soybean_finetune)")
    print("=" * 64)
    print(f"  Models       : {', '.join(MODEL_LABELS[m] for m in model_names)}")
    print(f"  Features     : {', '.join(feature_types)}")
    print(f"  Crop window  : {args.crop_window}")
    print(f"  Yield split  : {'ON (High/Low by train mean)' if args.yield_split else 'OFF'}")
    print(f"  Add geo      : {'ON (lat/lon from DB)' if args.add_geo else 'OFF'}")
    pca_label = str(pca_components) if pca_components is not None else 'OFF'
    print(f"  PCA          : {pca_label}")
    print(f"  Dataset dir  : {args.dataset_dir}")
    print(f"  Train years  : {args.train_years}")
    print(f"  Val years    : {args.val_years}")
    print(f"  Test years   : {args.test_years}")
    print(f"  Output dir   : {args.output_dir}")

    # ── データロード・前処理 ──────────────────────────────────────────────
    print("\nLoading data...", end=" ")
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     meta_train, meta_val, meta_test,
     norm_stats, crop_info) = load_soybean(
        args.dataset_dir,
        args.train_years,
        args.val_years,
        args.test_years,
        args.crop_window,
    )
    print("done")
    print(f"  Crop window  : {crop_info['window']}  T={crop_info['T_new']} days")
    if 'example' in crop_info:
        print(f"               ({crop_info['example']})")
    print(f"  X shape      : {X_train.shape} / {X_val.shape} / {X_test.shape} "
          f"(train / val / test)")
    print(f"  y range      : [{y_train.min():.2f}, {y_train.max():.2f}] (train)")
    print(f"  Norm stats   : mean[0]={norm_stats['mean'][0]:.3f}  "
          f"std[0]={norm_stats['std'][0]:.3f}  (TMP_mea)")

    # ── 緒度・経度の読み込み ───────────────────────────────────────────────
    if args.add_geo:
        print("\nLoading geo features (lat/lon)...", end=" ")
        geo_train, geo_val, geo_test, geo_stats = load_geo_features(
            args.db_path, meta_train, meta_val, meta_test
        )
        print("done")
        print(f"  lat: mean={geo_stats['mean'][0]:.4f}  std={geo_stats['std'][0]:.4f}")
        print(f"  lon: mean={geo_stats['mean'][1]:.4f}  std={geo_stats['std'][1]:.4f}")
    else:
        geo_train = geo_val = geo_test = None

    # ── 収量グループ分割の準備 ─────────────────────────────────────────────
    if args.yield_split:
        threshold, yield_groups = split_by_yield_mean(
            X_train, y_train, X_val, y_val, X_test, y_test,
            meta_train, meta_val, meta_test
        )
        print(f"\n  Yield threshold  : {threshold:.3f} (Train y mean)")
        for grp_label, grp in yield_groups.items():
            print(f"  Group '{grp_label}' : "
                  f"train={grp['n_train']}  val={grp['n_val']}  test={grp['n_test']}")
        # 'all' を先頭に、その後 High/Low の順で実行
        run_subsets = [('all', X_train, y_train, X_val, y_val, X_test, y_test,
                         meta_train, meta_val, meta_test)]
        for grp_label, grp in yield_groups.items():
            run_subsets.append((
                grp_label,
                grp['X_train'], grp['y_train'],
                grp['X_val'],   grp['y_val'],
                grp['X_test'],  grp['y_test'],
                grp['meta_train'], grp['meta_val'], grp['meta_test'],
            ))
    else:
        threshold = None
        run_subsets = [('all', X_train, y_train, X_val, y_val, X_test, y_test,
                         meta_train, meta_val, meta_test)]

    # ── 全組み合わせを実行 ────────────────────────────────────────────────
    # key = (model_name, feature_type, subset)
    all_results = {}
    for (subset_label, Xtr, ytr, Xva, yva, Xte, yte,
         mtr, mva, mte) in run_subsets:
        # geo features もサブセットに応じてスライスが必要
        if geo_train is not None:
            tr_mask_idx = mtr.index if hasattr(mtr, 'index') else range(len(mtr))
            # run_subsets は既にマスク済み meta を持つため、
            # geo もフィールドIDで対応させる
            geo_tr = _align_geo(geo_train, meta_train, mtr)
            geo_va = _align_geo(geo_val,   meta_val,   mva)
            geo_te = _align_geo(geo_test,  meta_test,  mte)
        else:
            geo_tr = geo_va = geo_te = None

        for model_name in model_names:
            for ft in feature_types:
                key = (model_name, ft, subset_label)
                try:
                    metrics, best_params = run_single(
                        model_name, ft, args.crop_window,
                        Xtr, ytr, Xva, yva, Xte, yte,
                        mtr, mva, mte,
                        args.output_dir,
                        subset=subset_label,
                        geo_train=geo_tr, geo_val=geo_va, geo_test=geo_te,
                        pca_components=pca_components,
                    )
                    all_results[key] = {'metrics': metrics,
                                        'best_params': best_params,
                                        'subset': subset_label}
                except ImportError as e:
                    print(f"\n  [SKIP] {model_name}: {e}")

    # ── サマリー ─────────────────────────────────────────────────────────
    if len(all_results) > 0:
        print(f"\n\n{'='*82}")
        print("  SUMMARY -- Test split")
        hdr_subset = '  Subset  ' if args.yield_split else ''
        print(f"  {'Model':<14} {'Feature':<12}{hdr_subset} "
              f"{'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R2':>8}")
        print(f"  {'─'*76}")
        for (mn, ft, sb), res in all_results.items():
            m = res['metrics']['test']
            sb_col = f"  {sb:<7}" if args.yield_split else ''
            print(f"  {MODEL_LABELS[mn]:<14} {ft:<12}{sb_col} "
                  f"{m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
                  f"{m['MAPE']:>7.2f}% {m['R2']:>8.4f}")
        print(f"{'='*82}")

        # 統合サマリー CSV
        summary_rows = []
        for (mn, ft, sb), res in all_results.items():
            for split_name in ('train', 'val', 'test'):
                m = res['metrics'][split_name]
                summary_rows.append({
                    'model':        mn,
                    'feature_type': ft,
                    'crop_window':  args.crop_window,
                    'subset':       sb,
                    'split':        split_name,
                    **m,
                })
        os.makedirs(args.output_dir, exist_ok=True)
        summary_csv = os.path.join(args.output_dir, 'summary_all.csv')
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"\n  Summary CSV -> {summary_csv}")
        if args.yield_split:
            print(f"  Yield threshold used: {threshold:.3f}")

    print("\nFinished.")


if __name__ == '__main__':
    main()
