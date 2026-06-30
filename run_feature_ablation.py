"""
run_feature_ablation.py
================================================================
yield_prediction_v3.py の追加特徴量フラグ 4 種
  --add-geo / --add-harm / --add-spacing / --add-vwc
の全組み合わせ（2^4 = 16 通り）× CV 方式（kfold / loyo）
= 最大 32 パターンを一括実行し、結果を比較する。

【出力】
  outputs/yield_pred_v3/feature_ablation_summary.csv
  outputs/yield_pred_v3/feature_ablation_top.txt   (上位ランキング)
"""

import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── yield_prediction_v3 から必要なものをインポート ─────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from yield_prediction_v3 import (
    WEATHER_COLS, GDD_THRESHOLDS, STAT_FUNCS,
    HARM_COLS, SPACING_COLS, VWC_COLS,
    FIELD_DB, WEATHER_DB, GDD_CSV,
    build_dataset, apply_iqr, make_models,
    run_kfold, run_loyo,
)

OUTPUT_DIR  = os.path.join('outputs', 'yield_pred_v3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 実行設定 ──────────────────────────────────────────────────────────────────
CV_MODES    = ['kfold', 'loyo']
USE_IQR     = False   # 全パターンで IQR は OFF に統一
USE_PCA     = False   # PCA は OFF に統一

FLAGS = ['add_geo', 'add_harm', 'add_spacing', 'add_vwc']

# ── ヘルパー: フラグの組み合わせ名 ───────────────────────────────────────────

def combo_label(flags: dict) -> str:
    active = [k for k, v in flags.items() if v]
    return '+'.join(active) if active else 'base'


def combo_dim(flags: dict, base_dim: int) -> int:
    d = base_dim
    if flags.get('add_geo'):     d += 2
    if flags.get('add_harm'):    d += len(HARM_COLS)
    if flags.get('add_spacing'): d += len(SPACING_COLS)
    if flags.get('add_vwc'):     d += len(VWC_COLS)
    return d


# ── メインループ ──────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    base_dim = len(WEATHER_COLS) * len(GDD_THRESHOLDS) * len(STAT_FUNCS)
    # GDD期間は3なので実際は N_PERIODS=3
    base_dim = len(WEATHER_COLS) * 3 * len(STAT_FUNCS)

    all_rows = []
    combos   = list(itertools.product([False, True], repeat=4))
    n_combos = len(combos)

    print('=' * 70)
    print(f'  Feature Ablation: {n_combos} combinations × {len(CV_MODES)} CV modes'
          f' = {n_combos * len(CV_MODES)} runs')
    print(f'  Flags: {FLAGS}')
    print('=' * 70)

    for ci, combo_vals in enumerate(combos, 1):
        flags = dict(zip(FLAGS, combo_vals))
        label = combo_label(flags)
        dim   = combo_dim(flags, base_dim)

        print(f'\n[{ci:2d}/{n_combos}] {label}  (dim={dim})')

        # ── データセット構築 ──────────────────────────────────────────────
        t0 = time.time()
        try:
            X, y, geo, meta, feat_cols = build_dataset(
                FIELD_DB, WEATHER_DB, GDD_CSV,
                add_harm    = flags['add_harm'],
                add_spacing = flags['add_spacing'],
                add_vwc     = flags['add_vwc'],
            )
        except Exception as e:
            print(f'  [ERROR] build_dataset: {e}')
            continue

        if flags['add_geo']:
            geo_filled = np.where(
                np.isnan(geo), np.nanmean(geo, axis=0), geo
            ).astype(np.float32)
            X = np.concatenate([X, geo_filled], axis=1)

        print(f'  dataset OK: N={len(y)}  dim={X.shape[1]}  ({time.time()-t0:.1f}s)')

        # ── CV 方式ループ ─────────────────────────────────────────────────
        for cv_mode in CV_MODES:
            print(f'  CV={cv_mode}', end=' ')
            t1 = time.time()

            try:
                if cv_mode == 'kfold':
                    fold_rows = run_kfold(X, y,
                                         use_pca=USE_PCA,
                                         pca_n=30,
                                         output_dir=None)
                else:  # loyo
                    fold_rows = run_loyo(X, y, meta,
                                         use_pca=USE_PCA,
                                         pca_n=30,
                                         output_dir=None)
            except Exception as e:
                print(f'[ERROR] {e}')
                continue

            elapsed = time.time() - t1

            # fold_rows を集計
            df_folds = pd.DataFrame(fold_rows)
            for model_name in df_folds['model'].unique():
                sub = df_folds[df_folds['model'] == model_name]
                row = {
                    'combo':      label,
                    'cv_mode':    cv_mode,
                    'model':      model_name,
                    'n_features': X.shape[1],
                    'add_geo':    flags['add_geo'],
                    'add_harm':   flags['add_harm'],
                    'add_spacing':flags['add_spacing'],
                    'add_vwc':    flags['add_vwc'],
                    'RMSE_mean':  sub['RMSE'].mean(),
                    'RMSE_std':   sub['RMSE'].std(),
                    'MAE_mean':   sub['MAE'].mean(),
                    'MAE_std':    sub['MAE'].std(),
                    'R2_mean':    sub['R2'].mean(),
                    'R2_std':     sub['R2'].std(),
                    'MAPE_mean':  sub['MAPE'].mean(),
                    'elapsed_s':  round(elapsed, 1),
                }
                all_rows.append(row)
                print(f'  {model_name}: RMSE={row["RMSE_mean"]:.3f}  '
                      f'R2={row["R2_mean"]:.4f}')

    # ── 結果保存 ──────────────────────────────────────────────────────────
    if not all_rows:
        print('\n結果が1件もありませんでした。')
        return

    result_df = pd.DataFrame(all_rows)
    csv_path  = os.path.join(OUTPUT_DIR, 'feature_ablation_summary.csv')
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n結果 CSV: {csv_path}')

    # ── ランキング表示 ────────────────────────────────────────────────────
    top_path = os.path.join(OUTPUT_DIR, 'feature_ablation_top.txt')
    lines = []

    for cv_mode in CV_MODES:
        for model_name in ['Ridge', 'LightGBM']:
            sub = result_df[
                (result_df['cv_mode'] == cv_mode) &
                (result_df['model']   == model_name)
            ].sort_values('R2_mean', ascending=False).reset_index(drop=True)

            header = f'\n=== {cv_mode.upper()}  /  {model_name}  (R2 降順) ==='
            lines.append(header)
            print(header)
            col_fmt = '{:3s}  {:35s}  {:5s}  {:7s}  {:7s}  {:7s}  {:7s}'
            row_fmt = '{:3d}  {:35s}  {:5d}  {:7.4f}  {:7.3f}  {:7.3f}  {:6.2f}%'
            hdr = col_fmt.format('Rank', 'Combo', 'Dim', 'R2', 'RMSE', 'MAE', 'MAPE')
            lines.append(hdr)
            print(hdr)
            lines.append('-' * 75)
            print('-' * 75)
            for i, r in sub.iterrows():
                line = row_fmt.format(
                    i + 1, r['combo'], int(r['n_features']),
                    r['R2_mean'], r['RMSE_mean'], r['MAE_mean'], r['MAPE_mean']
                )
                lines.append(line)
                print(line)

    with open(top_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nランキング: {top_path}')
    print(f'\n総経過時間: {time.time() - t_total:.1f}s')


if __name__ == '__main__':
    main()
