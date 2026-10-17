"""
松本市の全圃場データをCSVに出力するスクリプト。
特徴量: 収量・群（High/Low/Mid）・NDVI3期間・病害・気象フラグ・栽培変数・品種
"""
import sqlite3, pandas as pd, numpy as np
from tqdm import tqdm

MATSUMOTO_MUNI = 20202
QUANTILE       = 0.33
TARGET_YEARS   = [2015, 2016, 2017, 2018]

# ── 病害変数定義 ─────────────────────────────────────────────────
HARM_BINARY_COLS   = ['sick', 'weed', 'wet', 'bug']
HARM_BINARY_UNKNOWN= {'sick': [3], 'bug': [4]}
HARM_ORDINAL_COLS  = ['unripen', 'lay', 'loss']
HARM_FLAG_COLS     = ['heavy_rain', 'drought', 'typhoon', 'gale', 'long_rain', 'few_solar']
HARM_ALL_COLS      = HARM_BINARY_COLS + HARM_ORDINAL_COLS + HARM_FLAG_COLS
CULTIVAR_COLS      = ['between_lines', 'between_stocks']

NDVI_PERIODS = {
    'ndvi_early': ( 0,  33),
    'ndvi_mid':   (33,  67),
    'ndvi_late':  (67, 100),
}

# =============================================================================
# 1. 松本市の field_id 一覧取得
# =============================================================================
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce')
matsumoto_ids   = set(geo[geo['muniCd'] == MATSUMOTO_MUNI]['field_id'].unique())
print(f'松本市 field_id 数: {len(matsumoto_ids)}')

# =============================================================================
# 2. 質問票データ（収量・播種日・栽培変数・品種）
# =============================================================================
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
q = pd.read_sql(f"""
    SELECT field_id, year, yield, seed_date, harvest_date,
           breed, between_lines, between_stocks
    FROM Questionaire
    WHERE year BETWEEN {TARGET_YEARS[0]} AND {TARGET_YEARS[-1]}
      AND yield IS NOT NULL
      AND seed_date IS NOT NULL
      AND harvest_date IS NOT NULL
""", conn)
q = q.dropna(subset=['field_id'])
q['field_id'] = q['field_id'].astype(int)
q['year']     = q['year'].astype(int)
q = q[q['field_id'].isin(matsumoto_ids)].copy()
q['seed_date']    = pd.to_datetime(q['seed_date'])
q['harvest_date'] = pd.to_datetime(q['harvest_date'])
q['growing_days'] = (q['harvest_date'] - q['seed_date']).dt.days
for c in CULTIVAR_COLS:
    q[c] = pd.to_numeric(q[c], errors='coerce')

# =============================================================================
# 3. 病害データ
# =============================================================================
harm_col_str = ', '.join(HARM_ALL_COLS)
harm = pd.read_sql(f"""
    SELECT field_id, year, {harm_col_str}
    FROM harm WHERE field_id IS NOT NULL
""", conn)
conn.close()

harm = harm.dropna(subset=['field_id'])
harm['field_id'] = harm['field_id'].astype(int)
harm['year']     = harm['year'].astype(int)
harm = harm[harm['field_id'].isin(matsumoto_ids)].copy()

for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS:
    harm[c] = pd.to_numeric(harm[c], errors='coerce')
for c in HARM_FLAG_COLS:
    harm[c] = harm[c].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    harm[c] = pd.to_numeric(harm[c], errors='coerce').fillna(0).astype(int)

agg_d = {c: 'mean' for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS}
agg_d.update({c: 'max' for c in HARM_FLAG_COLS})
harm = harm.groupby(['field_id', 'year'])[HARM_ALL_COLS].agg(agg_d).reset_index()

# 不明センチネル値をNaNに変換
for col, unknowns in HARM_BINARY_UNKNOWN.items():
    for uval in unknowns:
        harm[col] = harm[col].where(harm[col] != uval, other=np.nan)

# =============================================================================
# 4. NDVI 3期間平均
# =============================================================================
ndvi_df = pd.read_csv('data/processed/ndvi_daily_smooth.csv')
ndvi_df['field_id'] = ndvi_df['field_id'].astype(int)
ndvi_df['year']     = ndvi_df['year'].astype(int)
ndvi_df = ndvi_df[ndvi_df['field_id'].isin(matsumoto_ids)].copy()

ndvi_rows = []
for _, frow in tqdm(q.iterrows(), total=len(q), desc='NDVI期間平均'):
    fid   = int(frow['field_id'])
    year  = int(frow['year'])
    gdays = frow['growing_days']
    row   = {'field_id': fid, 'year': year}
    if gdays <= 0:
        for col in NDVI_PERIODS: row[col] = np.nan
        ndvi_rows.append(row); continue
    nf  = ndvi_df[(ndvi_df['field_id'] == fid) & (ndvi_df['year'] == year)]
    if len(nf) < 2:
        for col in NDVI_PERIODS: row[col] = np.nan
        ndvi_rows.append(row); continue
    rel  = nf['DAP'].values / gdays * 100
    vals = nf['ndvi_smooth'].values.astype(float)
    for col, (lo, hi) in NDVI_PERIODS.items():
        mask = (rel >= lo) & (rel < hi)
        row[col] = float(np.nanmean(vals[mask])) if mask.sum() >= 1 else np.nan
    ndvi_rows.append(row)

ndvi_period_df = pd.DataFrame(ndvi_rows)

# =============================================================================
# 5. 結合 & High/Low/Mid ラベル付け
# =============================================================================
df = q.merge(harm, on=['field_id', 'year'], how='left')
df = df.merge(ndvi_period_df, on=['field_id', 'year'], how='left')

df['group'] = 'Mid'
for yr in TARGET_YEARS:
    mask = df['year'] == yr
    sub  = df[mask]
    if len(sub) < 6: continue
    lo_thr = sub['yield'].quantile(QUANTILE)
    hi_thr = sub['yield'].quantile(1 - QUANTILE)
    df.loc[mask & (df['yield'] <= lo_thr), 'group'] = 'Low'
    df.loc[mask & (df['yield'] >= hi_thr), 'group'] = 'High'

# =============================================================================
# 6. 列の並び替えと出力
# =============================================================================
output_cols = (
    ['field_id', 'year', 'yield', 'group']
    + list(NDVI_PERIODS.keys())
    + HARM_BINARY_COLS
    + HARM_ORDINAL_COLS
    + HARM_FLAG_COLS
    + CULTIVAR_COLS
    + ['breed']
)
# 存在する列のみ
output_cols = [c for c in output_cols if c in df.columns]

out_df = df[output_cols].sort_values(['year', 'yield']).reset_index(drop=True)

# 数値列を適切な桁数に丸める
for c in list(NDVI_PERIODS.keys()):
    out_df[c] = out_df[c].round(4)
out_df['yield'] = out_df['yield'].round(2)

out_path = 'outputs/yield_intra_analysis/matsumoto_fields_all.csv'
out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f'\n保存: {out_path}')
print(f'行数: {len(out_df)} 件 / 列数: {len(out_df.columns)} 列')
print(f'\n列一覧:')
for c in out_df.columns:
    n_valid = out_df[c].notna().sum()
    print(f'  {c:<22}: {n_valid}/{len(out_df)} 件有効')

print('\n先頭5行:')
print(out_df.head().to_string(index=False))
