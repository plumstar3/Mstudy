"""
06_save_ndvi_daily.py
============================
05_extract_ndvi_features.py のスプライン補間 + SGフィルタ処理を再利用し、
平滑化後の日次NDVI時系列を長形式CSVとして保存する。

出力形式（long format）:
    field_id, year, DAP, ndvi_smooth
    17,        2015, 0,   0.150
    17,        2015, 1,   0.152
    ...

出力先: ../../data/processed/ndvi_daily_smooth.csv
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from tqdm import tqdm

# =============================================================================
# パラメータ設定（05_extract_ndvi_features.py と同一）
# =============================================================================
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NDVI_CSV   = os.path.join(_ROOT, 'data', 'raw', 'hls_50m_ndvi_timeseries.csv')
FIELD_DB   = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
OUTPUT_CSV = os.path.join(_ROOT, 'data', 'processed', 'ndvi_daily_smooth.csv')

BARE_SOIL_NDVI = 0.15
SG_WINDOW      = 15
SG_POLYORDER   = 3

# =============================================================================
# スプライン補間 + SGフィルタ関数
# 05_extract_ndvi_features.py の process_field を改変し、
# 日次NDVI配列全体を返すように変更
# =============================================================================
def smooth_ndvi_series(obs_df: pd.DataFrame, harvest_dap: int):
    """
    1圃場分のNDVI時系列を補間・平滑化して日次配列を返す。

    Parameters
    ----------
    obs_df : pd.DataFrame
        'DAP', 'ndvi' 列を持つ、播種〜収穫期間内の観測データ
    harvest_dap : int
        収穫日のDAP（生育日数）

    Returns
    -------
    ndvi_sg : np.ndarray | None
        平滑化後の日次NDVI値（長さ = harvest_dap + 1）
        実観測が1点以下の場合は None を返す
    interp_method : str
    """
    # アンカー点の追加
    interp_src = obs_df.groupby('DAP')['ndvi'].mean().reset_index()

    anchor_start = pd.DataFrame({'DAP': [0],           'ndvi': [BARE_SOIL_NDVI]})
    anchor_end   = pd.DataFrame({'DAP': [harvest_dap], 'ndvi': [BARE_SOIL_NDVI]})

    if 0 not in interp_src['DAP'].values:
        interp_src = pd.concat([anchor_start, interp_src], ignore_index=True)
    if harvest_dap not in interp_src['DAP'].values:
        interp_src = pd.concat([interp_src, anchor_end], ignore_index=True)

    interp_src = interp_src.sort_values('DAP').reset_index(drop=True)
    n_obs = len(obs_df)  # 実観測点数（アンカー除く）

    # 実観測が1点以下は除外（05 と同じ基準）
    if n_obs <= 1:
        return None, 'insufficient'

    dap_daily = np.arange(0, harvest_dap + 1, dtype=float)

    # 補間
    if len(interp_src) >= 4:
        cs = CubicSpline(
            interp_src['DAP'].values,
            interp_src['ndvi'].values,
            extrapolate=False
        )
        ndvi_interp = np.clip(cs(dap_daily), -1.0, 1.0)
        interp_method = 'cubic_spline'
    else:
        ndvi_interp = np.clip(
            np.interp(dap_daily,
                      interp_src['DAP'].values,
                      interp_src['ndvi'].values),
            -1.0, 1.0
        )
        interp_method = 'linear'

    # SGフィルタ
    win = min(SG_WINDOW, len(ndvi_interp))
    if win % 2 == 0:
        win -= 1
    win = max(win, SG_POLYORDER + 1)

    ndvi_sg = np.clip(
        savgol_filter(ndvi_interp, window_length=win,
                      polyorder=SG_POLYORDER, mode='nearest'),
        -1.0, 1.0
    )

    return ndvi_sg, interp_method


# =============================================================================
# メイン処理
# =============================================================================
print('=== 日次NDVI時系列 保存スクリプト ===')
print(f'入力 NDVI CSV : {NDVI_CSV}')
print(f'入力 FieldDB  : {FIELD_DB}')
print(f'出力 CSV      : {OUTPUT_CSV}')
print()

# ── データ読み込み ─────────────────────────────────────────────────
print('データ読み込み中...')
ndvi_all = pd.read_csv(NDVI_CSV, parse_dates=['date'])
ndvi_all['field_id'] = ndvi_all['field_id'].astype(int)

conn = sqlite3.connect(FIELD_DB)
fields_df = pd.read_sql("""
    SELECT field_id, year, seed_date, harvest_date
    FROM Questionaire
    WHERE seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND year BETWEEN 2015 AND 2018
""", conn)
conn.close()

fields_df['seed_date']    = pd.to_datetime(fields_df['seed_date'])
fields_df['harvest_date'] = pd.to_datetime(fields_df['harvest_date'])
fields_df['field_id']     = fields_df['field_id'].astype(int)
fields_df['year']         = fields_df['year'].astype(int)

print(f'  対象圃場: {len(fields_df)} 件')
print(f'  NDVI観測数: {len(ndvi_all):,} 件')

# ── 圃場ごとに処理 ────────────────────────────────────────────────
rows = []
skip_log = []

for _, frow in tqdm(fields_df.iterrows(), total=len(fields_df),
                    desc='日次NDVI保存', unit='圃場'):
    fid          = int(frow['field_id'])
    year         = int(frow['year'])
    seed_date    = frow['seed_date']
    harvest_date = frow['harvest_date']
    harvest_dap  = (harvest_date - seed_date).days

    # 生育期間内の観測データを抽出
    obs = ndvi_all[
        (ndvi_all['field_id'] == fid)
    ].copy()

    if obs.empty:
        skip_log.append({'field_id': fid, 'year': year, 'reason': 'no_data'})
        continue

    obs['date'] = pd.to_datetime(obs['date'])
    obs['DAP'] = (obs['date'] - seed_date).dt.days
    in_season = obs[(obs['DAP'] >= 0) & (obs['DAP'] <= harvest_dap)].copy()

    if len(in_season) <= 1:
        skip_log.append({'field_id': fid, 'year': year,
                         'reason': f'insufficient（期間内観測{len(in_season)}点）'})
        continue

    # スプライン補間 + SGフィルタ
    ndvi_sg, method = smooth_ndvi_series(in_season[['DAP', 'ndvi']], harvest_dap)

    if ndvi_sg is None:
        skip_log.append({'field_id': fid, 'year': year, 'reason': 'smooth_failed'})
        continue

    # long format に展開
    for dap, val in enumerate(ndvi_sg):
        rows.append({
            'field_id'   : fid,
            'year'       : year,
            'DAP'        : dap,
            'ndvi_smooth': round(float(val), 5),
        })

# ── 保存 ─────────────────────────────────────────────────────────
result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_CSV, index=False)

print()
print(f'保存完了: {OUTPUT_CSV}')
print(f'  総行数       : {len(result_df):,} 行')
print(f'  圃場×年 数   : {result_df.groupby(["field_id","year"]).ngroups} 件')
print(f'  除外件数     : {len(skip_log)} 件')
if skip_log:
    reasons = pd.DataFrame(skip_log)['reason'].value_counts()
    print(f'  除外理由内訳 :\n{reasons.to_string()}')
