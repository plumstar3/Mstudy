"""
yield_prediction_v2.py  (v2.2 — アブレーションスタディ対応)
============================================================
【変更点】
  1. 検証軸A（CV方式）を 3 種類に拡張
       kfold      : KFold(n_splits=5, shuffle=True, random_state=42)
       loyo       : Leave-One-Year-Out（各年を1回ずつtestにする4fold）
       year_fixed : 固定年度分割（2018をtest、2015-2017をtrain）
  2. 検証軸B（PCA）を 2 種類に対応
       PCAなし（デフォルト）
       PCAあり（StandardScaler → PCA(n_components=30) → モデル）
  3. --ablation フラグで全組み合わせ（3×2=6パターン）を一括実行
  4. run_cv は後方互換で残しつつ、内部ロジックを共通関数に切り出し

【データ】
  - 気象変数: TMP_mea, TMP_max, TMP_min, APCPRA, SSD, GSR, WIND, SWE, RH
  - 特徴量: 9変数 × 3期間 × 5統計量 = 135次元（+ geo 2次元）
  - 期間: 各圃場の播種日〜収穫日（Questionaire テーブル）
  - 目的変数: yield（大豆収量）
"""

import argparse
import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# ── 定数 ──────────────────────────────────────────────────────────────────────

WEATHER_DB = os.path.join('data', 'processed', 'weather_database_fieldid.db')
FIELD_DB   = os.path.join('data', 'processed', 'FieldData_fieldid.db')
OUTPUT_DIR = os.path.join('outputs', 'yield_pred_v2')

WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
N_VARS = len(WEATHER_COLS)  # 9

TARGET_YEARS        = [2015, 2016, 2017, 2018]
DEFAULT_GROWING_DAYS = 180   # harvest_date 欠損時のデフォルト栽培日数

# ベースライン用固定ハイパーパラメータ
RIDGE_ALPHA       = 100
LGBM_N_ESTIMATORS = 200
LGBM_LR           = 0.05
LGBM_NUM_LEAVES   = 31

N_SPLITS     = 5
RANDOM_STATE = 42
PCA_N_DEFAULT = 30

# year_fixed の固定テスト年
FIXED_TEST_YEAR = 2018

# アブレーション全パターン定義（cv_mode, use_pca）
ABLATION_PATTERNS = [
    ('kfold',      False),
    ('kfold',      True),
    ('loyo',       False),
    ('loyo',       True),
    ('year_fixed', False),
    ('year_fixed', True),
]


# ── データ取得 ────────────────────────────────────────────────────────────────

def load_questionaire(field_db):
    """Questionaire テーブルから field_id, year, dates, yield, lat, lon を取得。"""
    conn = sqlite3.connect(field_db)
    df = pd.read_sql('''
        SELECT field_id, year, seed_date, harvest_date, yield, lat, lon
        FROM Questionaire
        WHERE field_id IS NOT NULL AND yield IS NOT NULL
          AND year BETWEEN 2015 AND 2018
    ''', conn)
    conn.close()
    df['field_id']     = df['field_id'].astype(int)
    df['year']         = df['year'].astype(int)
    df['seed_date']    = pd.to_datetime(df['seed_date'],    errors='coerce')
    df['harvest_date'] = pd.to_datetime(df['harvest_date'], errors='coerce')
    df['yield']        = df['yield'].astype(float)
    df['lat']          = pd.to_numeric(df['lat'], errors='coerce')
    df['lon']          = pd.to_numeric(df['lon'], errors='coerce')
    return df.reset_index(drop=True)


def load_weather(weather_db, field_ids, years):
    """weather_data テーブルから指定 field_id・年度の気象データを取得。"""
    fid_str = ','.join(map(str, field_ids))
    conn    = sqlite3.connect(weather_db)
    query   = f'''
        SELECT field_id, date,
               TMP_mea, TMP_max, TMP_min, APCPRA, SSD, GSR, WIND, SWE, RH
        FROM weather_data
        WHERE field_id IN ({fid_str})
          AND strftime('%Y', date) IN ({','.join(f"'{y}'" for y in years)})
    '''
    df = pd.read_sql(query, conn)
    conn.close()
    df['field_id'] = df['field_id'].astype(int)
    df['date']     = pd.to_datetime(df['date'])
    return df.sort_values(['field_id', 'date']).reset_index(drop=True)


# ── 栽培期間の決定 ────────────────────────────────────────────────────────────

def resolve_period(row):
    """各サンプルの栽培開始日・終了日を決定する。

    優先順位:
      1. seed_date + harvest_date が両方あれば そのまま使用
      2. seed_date のみ: harvest_date = seed_date + DEFAULT_GROWING_DAYS
      3. 両方 NULL: year/5/1 ~ year/10/31 をデフォルトとして使用
    """
    year = int(row['year'])
    sd, hd = row['seed_date'], row['harvest_date']
    if pd.notna(sd) and pd.notna(hd):
        return sd, hd, 'both'
    elif pd.notna(sd):
        return sd, sd + pd.Timedelta(days=DEFAULT_GROWING_DAYS), 'seed_only'
    else:
        return pd.Timestamp(year, 5, 1), pd.Timestamp(year, 10, 31), 'default'


# ── サンプルごとの特徴量抽出（period3stats） ─────────────────────────────────

def extract_sample_features(weather_df_field, start_date, end_date):
    """1圃場1年度の気象データを栽培期間で切り出し、period3stats 特徴量を返す。

    Returns:
        np.ndarray: shape (N_VARS * 3 * 5,) = 135次元、またはデータ不足時は None
    """
    mask = (weather_df_field['date'] >= start_date) & \
           (weather_df_field['date'] <= end_date)
    period_df = weather_df_field.loc[mask, WEATHER_COLS]

    if len(period_df) < 10:
        return None

    arr   = period_df.to_numpy(dtype=np.float32)  # (T, 9)
    T     = len(arr)
    p1e   = T // 3
    p2e   = (T * 2) // 3
    parts = []
    for p in [arr[:p1e], arr[p1e:p2e], arr[p2e:]]:
        if len(p) == 0:
            parts += [np.zeros(N_VARS)] * 5
        else:
            parts += [np.nanmean(p, 0), np.nanstd(p, 0),
                      np.nanmin(p, 0),  np.nanmax(p, 0), np.nanmedian(p, 0)]
    return np.concatenate(parts)  # (135,)


# ── データセット構築 ──────────────────────────────────────────────────────────

def build_dataset(field_db, weather_db):
    """Questionaire + weather_data を結合してサンプルごとの特徴量行列を構築。

    Returns:
        X    (N, 135)  気象統計特徴量
        y    (N,)      収量
        geo  (N, 2)    緯度・経度（lat, lon）
        meta DataFrame
    """
    print('Questionaire テーブル読み込み...', end=' ')
    qdf = load_questionaire(field_db)
    print(f'{len(qdf)} サンプル')

    all_fids  = sorted(qdf['field_id'].unique().tolist())
    all_years = sorted(qdf['year'].unique().tolist())

    print('気象データ読み込み...', end=' ')
    wdf = load_weather(weather_db, all_fids, all_years)
    print(f'{len(wdf):,} 行')

    weather_by_fid = {fid: grp.reset_index(drop=True)
                      for fid, grp in wdf.groupby('field_id')}

    print('特徴量抽出...')
    X_list, y_list, lat_list, lon_list, meta_list = [], [], [], [], []
    src_count = {'both': 0, 'seed_only': 0, 'default': 0, 'skip': 0}

    for _, row in qdf.iterrows():
        fid  = int(row['field_id'])
        year = int(row['year'])
        yval = float(row['yield'])
        start, end, src = resolve_period(row)

        if fid not in weather_by_fid:
            src_count['skip'] += 1
            continue
        feat = extract_sample_features(weather_by_fid[fid], start, end)
        if feat is None:
            src_count['skip'] += 1
            continue

        X_list.append(feat)
        y_list.append(yval)
        lat_list.append(float(row['lat']) if pd.notna(row['lat']) else np.nan)
        lon_list.append(float(row['lon']) if pd.notna(row['lon']) else np.nan)
        meta_list.append({'field_id': fid, 'year': year, 'yield': yval,
                          'period_source': src,
                          'start_date': start.date(), 'end_date': end.date()})
        src_count[src] += 1

    print(f'  採用: {len(X_list)} サンプル  '
          f'(both={src_count["both"]} seed_only={src_count["seed_only"]} '
          f'default={src_count["default"]} skip={src_count["skip"]})')

    X   = np.array(X_list, dtype=np.float32)
    y   = np.array(y_list, dtype=np.float32)
    geo = np.column_stack([lat_list, lon_list]).astype(np.float32)
    return X, y, geo, pd.DataFrame(meta_list)


# ── IQR 外れ値除外 ────────────────────────────────────────────────────────────

def apply_iqr(X, y, geo, meta):
    """全データセットの yield に IQR を適用して外れ値を除外する。

    ※ KFold CV の前に適用するため、fold をまたいで同じ基準が使われる点に注意。
       理想的には fold 内の train から IQR を算出すべきだが、
       ベースライン検証の簡易化のためグローバルに適用する。
    """
    q1  = float(np.percentile(y, 25))
    q3  = float(np.percentile(y, 75))
    iqr = q3 - q1
    lb  = q1 - 1.5 * iqr
    ub  = q3 + 1.5 * iqr
    keep = (y >= lb) & (y <= ub)
    n_out = (~keep).sum()
    out_fids = sorted(set(meta.loc[~keep, 'field_id'].tolist()))
    print(f'\nIQR 外れ値除外')
    print(f'  Q1={q1:.1f}  Q3={q3:.1f}  IQR={iqr:.1f}  '
          f'許容範囲: [{lb:.1f}, {ub:.1f}]')
    print(f'  除外: {n_out} 件', end='')
    if out_fids:
        print(f'  field_id={out_fids}', end='')
    print(f'  → 残り {keep.sum()} 件')
    return (X[keep], y[keep], geo[keep],
            meta[keep].reset_index(drop=True))


# ── 評価指標 ─────────────────────────────────────────────────────────────────

def calc_metrics(pred, target):
    rmse   = float(np.sqrt(mean_squared_error(target, pred)))
    mae    = float(np.abs(pred - target).mean())
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    nz     = np.abs(target) > 0
    mape   = float(np.mean(np.abs((pred[nz] - target[nz]) / target[nz])) * 100) \
             if nz.any() else float('nan')
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


# ── モデル定義（Pipeline: Imputer → Scaler [→ PCA] → モデル） ────────────────

def make_models(use_pca=False, pca_n=PCA_N_DEFAULT):
    """Ridge と LightGBM の Pipeline を返す。

    Args:
        use_pca (bool): True の場合、StandardScaler の後に PCA を挿入する。
        pca_n   (int):  PCA の出力次元数。
    """
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer

    def _build(model_obj):
        steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
        ]
        if use_pca:
            steps.append(('pca', PCA(n_components=pca_n, random_state=RANDOM_STATE)))
        steps.append(('model', model_obj))
        return Pipeline(steps)

    ridge = _build(Ridge(alpha=RIDGE_ALPHA))

    lgbm = _build(lgb.LGBMRegressor(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LR,
        num_leaves=LGBM_NUM_LEAVES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ))

    return {'Ridge': ridge, 'LightGBM': lgbm}


# ── 共通: fold ループ実行 ─────────────────────────────────────────────────────

def _run_folds(X, y, splits_iter, n_folds_label, models):
    """(train_idx, val_idx) のイテレータを受け取り、fold ごとに学習・評価する。

    Args:
        splits_iter   : enumerate 済みの (fold_idx, train_idx, val_idx) イテラブル
                        ※ 実際には enumerate なしで (tr, va) のペアを渡す
        n_folds_label : サマリー表示用のフォールド数文字列 (例: "5", "LOYO-4")
        models        : make_models() が返す dict（毎 fold でリセットされない！
                        ※ Pipeline は fit のたびに上書きされるので問題なし）

    Returns:
        metrics  : {model_name: [fold_metric_dict, ...]}
        fold_log : list of dicts（CSV 保存用）
    """
    metrics = {name: [] for name in models}

    for fold_idx, (tr_idx, va_idx) in enumerate(splits_iter):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        print(f'  Fold {fold_idx + 1}  (train={len(y_tr)} val={len(y_va)})')

        for model_name, pipeline in models.items():
            t0   = time.time()
            pipeline.fit(X_tr, y_tr)
            pred = pipeline.predict(X_va)
            m    = calc_metrics(pred, y_va)
            metrics[model_name].append(m)
            print(f'    {model_name:<10} RMSE={m["RMSE"]:7.3f}  MAE={m["MAE"]:7.3f}  '
                  f'MAPE={m["MAPE"]:6.2f}%  R2={m["R2"]:7.4f}  '
                  f'({time.time()-t0:.1f}s)')

    return metrics


def _print_summary(metrics, cv_label, use_pca, pca_n):
    """fold 結果のサマリーを表示し、summary_rows を返す。"""
    pca_label = f'PCA({pca_n}d)' if use_pca else 'NoPCA'
    print(f'\n{"=" * 65}')
    print(f'  SUMMARY  [{cv_label} / {pca_label}]  mean ± std')
    print(f'{"=" * 65}')
    header = f'  {"Model":<12} {"RMSE":>14} {"MAE":>14} {"MAPE":>12} {"R2":>12}'
    print(header)
    print(f'  {"─" * 60}')

    summary_rows = []
    for model_name, fold_data in metrics.items():
        stats = {}
        for key in ('RMSE', 'MAE', 'MAPE', 'R2'):
            vals       = [f[key] for f in fold_data]
            stats[key] = (np.mean(vals), np.std(vals))

        print(f'  {model_name:<12} '
              f'{stats["RMSE"][0]:>7.3f}±{stats["RMSE"][1]:<5.3f}  '
              f'{stats["MAE"][0]:>7.3f}±{stats["MAE"][1]:<5.3f}  '
              f'{stats["MAPE"][0]:>5.2f}%±{stats["MAPE"][1]:.2f}  '
              f'{stats["R2"][0]:>6.4f}±{stats["R2"][1]:.4f}')

        for fold_idx, m in enumerate(fold_data):
            summary_rows.append({
                'cv_mode': cv_label, 'pca': use_pca,
                'model': model_name, 'fold': fold_idx + 1, **m,
            })
    print(f'{"=" * 65}')
    return summary_rows


# ── CV方式ごとの実行関数 ──────────────────────────────────────────────────────

def run_kfold(X, y, use_pca, pca_n):
    """5-Fold CV を実行する。

    Returns:
        summary_rows : list of dicts（CSV 保存用）
    """
    print(f'\n{"─" * 65}')
    print(f'  [5-Fold CV]  PCA={"あり(" + str(pca_n) + "d)" if use_pca else "なし"}')
    print(f'  KFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})')
    print(f'{"─" * 65}')

    kf      = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models  = make_models(use_pca=use_pca, pca_n=pca_n)
    metrics = _run_folds(X, y, kf.split(X), N_SPLITS, models)
    return _print_summary(metrics, 'kfold', use_pca, pca_n)


def run_loyo(X, y, meta, use_pca, pca_n):
    """Leave-One-Year-Out CV を実行する。

    各年を 1 回ずつ test セットにする（年度数 = フォールド数）。

    Returns:
        summary_rows : list of dicts（CSV 保存用）
    """
    years = sorted(meta['year'].unique().tolist())
    print(f'\n{"─" * 65}')
    print(f'  [Leave-One-Year-Out]  PCA={"あり(" + str(pca_n) + "d)" if use_pca else "なし"}')
    print(f'  テスト年: {years}  ({len(years)}-fold)')
    print(f'{"─" * 65}')

    year_arr = meta['year'].to_numpy()

    def _splits():
        for test_year in years:
            va_mask = year_arr == test_year
            tr_idx  = np.where(~va_mask)[0]
            va_idx  = np.where(va_mask)[0]
            print(f'  [test_year={test_year}]  '
                  f'train={len(tr_idx)}  val={len(va_idx)}')
            yield tr_idx, va_idx

    models  = make_models(use_pca=use_pca, pca_n=pca_n)
    metrics = _run_folds(X, y, _splits(), len(years), models)
    return _print_summary(metrics, 'loyo', use_pca, pca_n)


def run_year_fixed(X, y, meta, use_pca, pca_n, test_year=FIXED_TEST_YEAR):
    """固定年度分割を実行する（test_year をテストセット、残りをトレーニングセット）。

    1 fold のみなので標準偏差は 0 となる。

    Returns:
        summary_rows : list of dicts（CSV 保存用）
    """
    print(f'\n{"─" * 65}')
    print(f'  [固定年度分割]  PCA={"あり(" + str(pca_n) + "d)" if use_pca else "なし"}')
    print(f'  train: {[y for y in sorted(meta["year"].unique()) if y != test_year]}  '
          f'test: {test_year}')
    print(f'{"─" * 65}')

    year_arr = meta['year'].to_numpy()
    va_mask  = year_arr == test_year
    tr_idx   = np.where(~va_mask)[0]
    va_idx   = np.where(va_mask)[0]

    if len(va_idx) == 0:
        print(f'  !! test_year={test_year} のサンプルが存在しません。スキップします。')
        return []

    def _splits():
        yield tr_idx, va_idx

    models  = make_models(use_pca=use_pca, pca_n=pca_n)
    metrics = _run_folds(X, y, _splits(), 1, models)
    return _print_summary(metrics, f'year_fixed({test_year})', use_pca, pca_n)


# ── 単一パターン実行ラッパー ──────────────────────────────────────────────────

def run_single(X, y, meta, cv_mode, use_pca, pca_n):
    """1 パターン（cv_mode × PCA有無）を実行して summary_rows を返す。"""
    if cv_mode == 'kfold':
        return run_kfold(X, y, use_pca, pca_n)
    elif cv_mode == 'loyo':
        return run_loyo(X, y, meta, use_pca, pca_n)
    elif cv_mode == 'year_fixed':
        return run_year_fixed(X, y, meta, use_pca, pca_n)
    else:
        raise ValueError(f'不明な cv_mode: {cv_mode}')


# ── アブレーション一括実行 ────────────────────────────────────────────────────

def run_ablation(X, y, meta, pca_n, output_dir):
    """全 6 パターン（3 CV方式 × 2 PCA設定）を一括実行する。"""
    all_rows = []

    for pattern_idx, (cv_mode, use_pca) in enumerate(ABLATION_PATTERNS, start=1):
        pca_str = f'PCA({pca_n}d)' if use_pca else 'NoPCA'
        print(f'\n\n{"#" * 65}')
        print(f'#  アブレーション [{pattern_idx}/{len(ABLATION_PATTERNS)}]  '
              f'cv={cv_mode}  pca={pca_str}')
        print(f'{"#" * 65}')

        rows = run_single(X, y, meta, cv_mode, use_pca, pca_n)
        all_rows.extend(rows)

        # パターンごとの CSV 保存
        pca_tag  = f'pca{pca_n}d' if use_pca else 'nopca'
        csv_name = f'ablation_{cv_mode}_{pca_tag}.csv'
        csv_path = os.path.join(output_dir, csv_name)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f'\n  → {csv_path}')

    # ── アブレーションまとめ表示 ──────────────────────────────────────────────
    print(f'\n\n{"=" * 75}')
    print(f'  ABLATION SUMMARY (全{len(ABLATION_PATTERNS)}パターン / mean of folds)')
    print(f'{"=" * 75}')

    df_all = pd.DataFrame(all_rows)
    grp = df_all.groupby(['cv_mode', 'pca', 'model'])[['RMSE', 'MAE', 'MAPE', 'R2']].mean()

    # 読みやすい形でテーブル表示
    header = f'  {"cv_mode":<22} {"PCA":<6} {"Model":<10} ' \
             f'{"RMSE":>8} {"MAE":>8} {"MAPE%":>8} {"R2":>8}'
    print(header)
    print(f'  {"─" * 70}')
    for (cv_mode, use_pca, model_name), row in grp.iterrows():
        pca_str = f'Y({pca_n}d)' if use_pca else 'N'
        print(f'  {cv_mode:<22} {pca_str:<6} {model_name:<10} '
              f'{row["RMSE"]:>8.3f} {row["MAE"]:>8.3f} '
              f'{row["MAPE"]:>8.2f} {row["R2"]:>8.4f}')
    print(f'{"=" * 75}')

    # まとめ CSV 保存
    abl_path = os.path.join(output_dir, 'ablation_results.csv')
    df_all.to_csv(abl_path, index=False)
    print(f'\n  全パターン結果 CSV → {abl_path}')


# ── メインエントリ ────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── ヘッダー表示 ────────────────────────────────────────────────────────
    print('=' * 65)
    print('  Yield Prediction v2.2  (アブレーションスタディ対応)')
    print(f'  気象変数  : {WEATHER_COLS}')
    print(f'  特徴量    : {N_VARS}変数 × 3期間 × 5統計量 = {N_VARS*3*5}次元'
          + (' + lat/lon 2次元' if args.add_geo else ''))
    if args.ablation:
        print(f'  モード    : アブレーション一括実行 (全{len(ABLATION_PATTERNS)}パターン)')
    else:
        pca_str = f'あり({args.pca_n}d)' if args.pca else 'なし'
        print(f'  CV方式    : {args.cv_mode}')
        print(f'  PCA       : {pca_str}')
    print(f'  IQR 除外  : {"ON" if args.iqr else "OFF"}')
    print(f'  モデル    : Ridge(alpha={RIDGE_ALPHA}), '
          f'LightGBM(n_est={LGBM_N_ESTIMATORS} lr={LGBM_LR} leaves={LGBM_NUM_LEAVES})')
    print('=' * 65)

    # ── データ構築 ──────────────────────────────────────────────────────────
    X, y, geo, meta = build_dataset(args.field_db, args.weather_db)

    # ── IQR 外れ値除外（任意） ───────────────────────────────────────────────
    if args.iqr:
        X, y, geo, meta = apply_iqr(X, y, geo, meta)

    # ── 緯度・経度を特徴量に結合（任意） ─────────────────────────────────────
    if args.add_geo:
        geo_filled = np.where(np.isnan(geo),
                              np.nanmean(geo, axis=0), geo).astype(np.float32)
        X = np.concatenate([X, geo_filled], axis=1)

    n_total, n_feat = X.shape
    print(f'\n総サンプル数: {n_total}  特徴量次元: {n_feat}')
    print(f'yield: min={y.min():.1f}  max={y.max():.1f}  '
          f'mean={y.mean():.1f}  std={y.std():.1f}')

    # ── 実行モードの分岐 ─────────────────────────────────────────────────────
    if args.ablation:
        run_ablation(X, y, meta, pca_n=args.pca_n, output_dir=args.output_dir)

    else:
        rows = run_single(X, y, meta,
                          cv_mode=args.cv_mode,
                          use_pca=args.pca,
                          pca_n=args.pca_n)

        csv_path = os.path.join(args.output_dir, 'cv_results_v2.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f'\n  Fold 詳細 CSV → {csv_path}')

    print('\nFinished.')


# ── 後方互換: run_cv ──────────────────────────────────────────────────────────

def run_cv(args):
    """後方互換のためのエントリポイント。内部で main() を呼び出す。

    旧来の --cv-mode / --pca フラグがない場合はデフォルト
    （kfold / PCAなし）として動作する。
    """
    # run_cv 時代は kfold かつ PCAなし が固定設定だったため、
    # args に cv_mode / pca / pca_n / ablation がなければデフォルト値を付与する。
    if not hasattr(args, 'cv_mode'):
        args.cv_mode = 'kfold'
    if not hasattr(args, 'pca'):
        args.pca = False
    if not hasattr(args, 'pca_n'):
        args.pca_n = PCA_N_DEFAULT
    if not hasattr(args, 'ablation'):
        args.ablation = False
    main(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Yield prediction v2.2: アブレーションスタディ (3 CV方式 × 2 PCA設定)',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── 検証軸A: CV方式 ─────────────────────────────────────────────────────
    p.add_argument(
        '--cv-mode',
        choices=['kfold', 'loyo', 'year_fixed'],
        default='kfold',
        help=(
            'CV方式を指定 (デフォルト: kfold)\n'
            '  kfold      : 5-Fold CV (KFold, shuffle)\n'
            '  loyo       : Leave-One-Year-Out (各年をtestに)\n'
            '  year_fixed : 固定年度分割 (2018をtest, 残りをtrain)'
        ),
    )

    # ── 検証軸B: PCA ─────────────────────────────────────────────────────────
    p.add_argument('--pca',   action='store_true',
                   help='PCAを有効にする (StandardScaler → PCA → モデル)')
    p.add_argument('--pca-n', type=int, default=PCA_N_DEFAULT,
                   dest='pca_n',
                   help=f'PCAの出力次元数 (デフォルト: {PCA_N_DEFAULT})')

    # ── アブレーション一括実行 ───────────────────────────────────────────────
    p.add_argument(
        '--ablation', action='store_true',
        help=(
            '全パターンを一括実行する\n'
            '  (kfold/loyo/year_fixed) × (PCAなし/あり) = 6パターン\n'
            '  ※ このフラグがある場合 --cv-mode / --pca は無視される'
        ),
    )

    # ── その他オプション ─────────────────────────────────────────────────────
    p.add_argument('--add-geo', action='store_true',
                   help='lat/lon を特徴量ベクトルに追加する')
    p.add_argument('--iqr', action='store_true',
                   help='IQR外れ値除去を適用する')
    p.add_argument('--weather-db', default=WEATHER_DB, dest='weather_db')
    p.add_argument('--field-db',   default=FIELD_DB,   dest='field_db')
    p.add_argument('--output-dir', default=OUTPUT_DIR, dest='output_dir')

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
