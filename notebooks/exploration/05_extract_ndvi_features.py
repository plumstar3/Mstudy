"""
05_extract_ndvi_features.py
============================
全圃場のNDVI時系列に対して以下を一括処理する。
  1. 播種日（DAP=0）と収穫日（DAP=harvest_dap）に裸地アンカー点を追加
  2. 3次スプライン補間（4点以上）または線形補間（2〜3点）で日次データを生成
  3. Savitzky-Golayフィルタで平滑化
  4. 特徴量（Peak NDVI, AUC, DAP60-90 NDVI, Senescence Rate など）を抽出
  5. 1圃場1行の横持ちCSV（機械学習用）として出力

除外対象:
  - 生育期間内の実観測が0点の圃場（アンカー2点のみでは曲線が意味をなさない）
  - no_data（CSVにデータが存在しない）圃場

出力:
  data/processed/ndvi_features.csv
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from tqdm import tqdm

# =============================================================================
# パラメータ設定
# =============================================================================
NDVI_CSV   = '../../data/raw/hls_50m_ndvi_timeseries.csv'
FIELD_DB   = '../../data/processed/FieldData_fieldid.db'
OUTPUT_CSV = '../../data/processed/ndvi_features_v2.csv'

# 裸地NDVI（播種日・収穫日のアンカー点に使用）
BARE_SOIL_NDVI = 0.15

# SGフィルタのパラメータ
SG_WINDOW    = 15   # 窓幅（奇数）
SG_POLYORDER = 3    # 多項式次数

# 子実肥大期相当のDAP範囲（大豆のR5〜R6ステージ目安）
R5R6_DAP_START = 60
R5R6_DAP_END   = 90

# =============================================================================
# 補間 + SGフィルタ関数（04_ndvi_smoothing_test.py と同じロジック）
# =============================================================================
def process_field(obs_df: pd.DataFrame,
                  harvest_dap: int) -> dict:
    """
    1圃場分のNDVI時系列を平滑化して特徴量を抽出する。

    Parameters
    ----------
    obs_df : pd.DataFrame
        'DAP', 'ndvi' カラムを持つ、播種〜収穫期間内の観測データ
    harvest_dap : int
        収穫日のDAP（生育日数）

    Returns
    -------
    dict : 抽出した特徴量と品質フラグ
    """
    # ── アンカー点の追加 ─────────────────────────────────────────
    interp_src = obs_df.groupby('DAP')['ndvi'].mean().reset_index()

    anchor_start = pd.DataFrame({'DAP': [0],           'ndvi': [BARE_SOIL_NDVI]})
    anchor_end   = pd.DataFrame({'DAP': [harvest_dap], 'ndvi': [BARE_SOIL_NDVI]})

    if 0 not in interp_src['DAP'].values:
        interp_src = pd.concat([anchor_start, interp_src], ignore_index=True)
    if harvest_dap not in interp_src['DAP'].values:
        interp_src = pd.concat([interp_src, anchor_end], ignore_index=True)

    interp_src = interp_src.sort_values('DAP').reset_index(drop=True)
    n_obs = len(obs_df)   # 実観測点数（アンカーを除く）

    # ── 補間方法の決定 ─────────────────────────────────────────────
    dap_daily = np.arange(0, harvest_dap + 1, dtype=float)

    if len(interp_src) >= 4:
        # 3次スプライン補間
        cs = CubicSpline(
            interp_src['DAP'].values,
            interp_src['ndvi'].values,
            extrapolate=False
        )
        ndvi_interp   = np.clip(cs(dap_daily), -1.0, 1.0)
        interp_method = 'cubic_spline'
    else:
        # 線形補間
        ndvi_interp   = np.clip(
            np.interp(dap_daily,
                      interp_src['DAP'].values,
                      interp_src['ndvi'].values),
            -1.0, 1.0
        )
        interp_method = 'linear'

    # ── SGフィルタで平滑化 ─────────────────────────────────────────
    win = min(SG_WINDOW, len(ndvi_interp))
    if win % 2 == 0:
        win -= 1
    win = max(win, SG_POLYORDER + 1)   # polyorder < window_length を保証

    ndvi_sg = np.clip(
        savgol_filter(ndvi_interp, window_length=win,
                      polyorder=SG_POLYORDER, mode='nearest'),
        -1.0, 1.0
    )

    # ── 特徴量の抽出 ───────────────────────────────────────────────

    # ①  固定DAP点のNDVI（生育期間外の場合は NaN）
    # LightGBMは NaN をネイティブに扱えるため、無理な補完はしない
    DAP_POINTS = [30, 45, 60, 75, 90, 105, 120]
    ndvi_at_dap = {}
    for dap in DAP_POINTS:
        key = f'ndvi_dap{dap:03d}'
        if dap <= harvest_dap:
            ndvi_at_dap[key] = float(ndvi_sg[dap])   # dap_daily[i] == i なので直接インデックス
        else:
            ndvi_at_dap[key] = np.nan                 # 生育期間外 → NaN

    # ②  要約統計量
    peak_idx  = int(np.argmax(ndvi_sg))
    peak_ndvi = float(ndvi_sg[peak_idx])
    peak_dap  = int(dap_daily[peak_idx])

    # 総AUC（播種〜収穫）
    auc = float(np.trapz(ndvi_sg, dap_daily))

    # 登熟期AUC（ピーク以降の積分：実が充実する期間の総量）
    auc_late = float(np.trapz(ndvi_sg[peak_idx:], dap_daily[peak_idx:]))

    # 子実肥大期（DAP60〜90）平均NDVI
    mask_r5r6  = (dap_daily >= R5R6_DAP_START) & (dap_daily <= R5R6_DAP_END)
    ndvi_r5r6  = float(ndvi_sg[mask_r5r6].mean()) if mask_r5r6.any() else np.nan

    # 枯れ下がり速度（ピーク以降の平均低下速度 NDVI/日）
    post_peak = ndvi_sg[peak_idx:]
    senescence_rate = float((post_peak[0] - post_peak[-1]) / max(len(post_peak) - 1, 1))

    # 立ち上がり速度（播種日からピーク到達までの平均上昇速度 NDVI/日）
    green_up_rate = float((peak_ndvi - BARE_SOIL_NDVI) / max(peak_dap, 1))

    # ── 品質フラグ ──────────────────────────────────────────────────
    # ※ アンカー2点が追加されるため、実観測2点以上なら必ず3次スプライン補間になる
    #   n_obs=0 → 合計2点 → 線形補間（除外推奨）
    #   n_obs=1 → 合計3点 → 線形補間（除外推奨）
    #   n_obs=2-3 → 合計4-5点 → 3次スプライン ※ quality='sparse'
    #   n_obs≥4  → 合計6点以上 → 3次スプライン ※ quality='good'
    if n_obs >= 4:
        quality = 'good'
    elif n_obs >= 2:
        quality = 'sparse'         # 実観測2〜3点（アンカー込みで3次スプライン）
    else:
        quality = 'insufficient'   # 実観測0〜1点（線形補間しかできない → 除外推奨）

    return {
        # 固定DAP点のNDVI（主軸特徴量）
        **ndvi_at_dap,
        # 要約統計量（補助特徴量）
        'ndvi_r5r6_mean' : ndvi_r5r6,
        'auc'            : auc,
        'auc_late'       : auc_late,
        'green_up_rate'  : green_up_rate,
        'senescence_rate': senescence_rate,
        'peak_ndvi'      : peak_ndvi,
        'peak_dap'       : peak_dap,
        # メタ情報
        'n_obs'          : n_obs,
        'interp_method'  : interp_method,
        'quality'        : quality,
    }


# =============================================================================
# メイン処理
# =============================================================================
print('=== NDVI特徴量 一括抽出 ===')

# ── データ読み込み ─────────────────────────────────────────────────
print('データ読み込み中...')
ndvi_all = pd.read_csv(NDVI_CSV, parse_dates=['date'])

conn = sqlite3.connect(FIELD_DB)
fields_df = pd.read_sql("""
    SELECT field_id, year, lat, lon, seed_date, harvest_date, yield
    FROM Questionaire
    WHERE seed_date IS NOT NULL AND harvest_date IS NOT NULL
""", conn)
conn.close()

fields_df['seed_date']    = pd.to_datetime(fields_df['seed_date'])
fields_df['harvest_date'] = pd.to_datetime(fields_df['harvest_date'])

print(f'  対象圃場: {len(fields_df)} 件')
print(f'  NDVI観測数: {len(ndvi_all):,} 件')

# ── 圃場ごとに処理 ────────────────────────────────────────────────
results = []
skip_log = []

for _, frow in tqdm(fields_df.iterrows(), total=len(fields_df),
                    desc='特徴量抽出', unit='圃場'):
    fid          = int(frow['field_id'])
    year         = int(frow['year'])
    seed_date    = frow['seed_date']
    harvest_date = frow['harvest_date']
    harvest_dap  = (harvest_date - seed_date).days

    # 生育期間内の観測データを抽出
    obs = ndvi_all[
        (ndvi_all['field_id'] == fid) &
        (ndvi_all['year']     == year)
    ].copy()

    if obs.empty:
        skip_log.append({'field_id': fid, 'year': year, 'reason': 'no_data（CSVにデータなし）'})
        continue

    obs['DAP'] = (obs['date'] - seed_date).dt.days
    in_season  = obs[(obs['DAP'] >= 0) & (obs['DAP'] <= harvest_dap)].copy()

    # 実観測0〜1点はアンカーのみ（または線形補間しかできない）→ 除外
    if len(in_season) <= 1:
        skip_log.append({'field_id': fid, 'year': year,
                         'reason': f'insufficient（期間内観測{len(in_season)}点）'})
        continue

    # 特徴量の抽出
    feats = process_field(in_season[['DAP', 'ndvi']], harvest_dap)

    results.append({
        'field_id'       : fid,
        'year'           : year,
        'lat'            : float(frow['lat']),
        'lon'            : float(frow['lon']),
        'yield'          : float(frow['yield']),
        'growing_days'   : harvest_dap,
        **feats,
    })

# ── 結果を保存 ────────────────────────────────────────────────────
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_CSV, index=False, float_format='%.5f')

# ── サマリー表示 ─────────────────────────────────────────────────
print()
print('=' * 55)
print('   処理完了')
print('=' * 55)
print(f'  成功: {len(result_df):,} 件 → {OUTPUT_CSV}')
print(f'  除外: {len(skip_log):,} 件')

if skip_log:
    print('\n--- 除外された圃場 ---')
    for s in skip_log:
        print(f"  field_id={s['field_id']}, year={s['year']}: {s['reason']}")

print()
print('--- 品質フラグ分布 ---')
print(result_df['quality'].value_counts().to_string())

print()
print('--- 抽出された特徴量の基本統計 ---')
feat_cols = ['peak_ndvi', 'peak_dap', 'auc', 'ndvi_r5r6_mean', 'senescence_rate', 'growing_days']
print(result_df[feat_cols].describe().round(3).to_string())

print()
print('--- 先頭5行 ---')
print(result_df.head().to_string())
