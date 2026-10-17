"""
18_ndvi_intra_plot.py
=====================
NDVI時系列の「同一地域内」多収 vs 低収の可視化。

[対象地域] 収量とNDVIが揃っている圃場×年の件数が最多の市町村（TARGET_MUNI で手動指定可）
[グループ] 市町村内で年ごとに収量上位33%→High、下位33%→Low、他はMid
[出力]    outputs/ndvi_intra_analysis/
    18_muni_candidates.csv
    18_{muniCd}_{city}_ndvi_all.png      全圃場×年を1つの図に（High=赤 / Low=青 / Mid=灰）+ 群平均
    18_{muniCd}_{city}_ndvi_by_year.png  年別パネル版
    18_{muniCd}_{city}_fields.csv        対象圃場×年の一覧（group付き）
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB    = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
NDVI_CSV    = os.path.join(_ROOT, 'data', 'processed', 'ndvi_daily_smooth.csv')
GEOCODE_CSV = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'ndvi_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MUNI    = None
TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33
COL = {'High': '#E74C3C', 'Low': '#3498DB', 'Mid': '#B0B0B0'}

# ---------------------------------------------------------------- データ
conn = sqlite3.connect(FIELD_DB)
quest = pd.read_sql("""SELECT field_id, year, yield FROM Questionaire
                       WHERE yield IS NOT NULL AND field_id IS NOT NULL
                         AND year BETWEEN 2015 AND 2018""", conn)
conn.close()
quest['field_id'] = quest['field_id'].astype(int)
quest['year']     = quest['year'].astype(int)

ndvi = pd.read_csv(NDVI_CSV)
ndvi['field_id'] = ndvi['field_id'].astype(int)
ndvi['year']     = ndvi['year'].astype(int)
ndvi_keys = ndvi.groupby(['field_id', 'year']).size().rename('n_days').reset_index()
print(f'NDVI: {len(ndvi_keys)} 圃場×年 / 収量: {len(quest)} 件')

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
geo = geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id')

df = (quest.merge(ndvi_keys, on=['field_id', 'year'], how='inner')
           .merge(geo, on='field_id', how='left').dropna(subset=['muniCd']))
df['muniCd'] = df['muniCd'].astype(int)

cand = (df.groupby(['muniCd', 'city'])
          .agg(n_field_years=('field_id', 'size'), n_fields=('field_id', 'nunique'))
          .reset_index().sort_values(['n_field_years', 'n_fields'], ascending=False)
          .reset_index(drop=True))
cand.to_csv(os.path.join(OUT_DIR, '18_muni_candidates.csv'), index=False, encoding='utf-8-sig')
print('\n市町村別 NDVI+収量あり（上位10）:')
print(cand.head(10).to_string(index=False))

if TARGET_MUNI is None:
    muni_cd, city = int(cand.loc[0, 'muniCd']), str(cand.loc[0, 'city'])
else:
    muni_cd = TARGET_MUNI
    city = str(cand.loc[cand['muniCd'] == muni_cd, 'city'].iloc[0])
print(f'\n★ 対象: {city} ({muni_cd})')

mdf = df[df['muniCd'] == muni_cd].copy().reset_index(drop=True)
mdf['group'] = 'Mid'
for yr in TARGET_YEARS:
    m = mdf['year'] == yr
    if m.sum() < 6:
        continue
    lo = mdf.loc[m, 'yield'].quantile(GROUP_QUANTILE)
    hi = mdf.loc[m, 'yield'].quantile(1 - GROUP_QUANTILE)
    mdf.loc[m & (mdf['yield'] <= lo), 'group'] = 'Low'
    mdf.loc[m & (mdf['yield'] >= hi), 'group'] = 'High'

print(f'  圃場数 {mdf["field_id"].nunique()} / 圃場×年 {len(mdf)}')
print(mdf.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())
n = mdf['group'].value_counts()

tag = f'18_{muni_cd}_{city}'
mdf.sort_values(['year', 'yield'], ascending=[True, False]).to_csv(
    os.path.join(OUT_DIR, f'{tag}_fields.csv'), index=False, encoding='utf-8-sig')

ts = ndvi.merge(mdf[['field_id', 'year', 'group', 'yield']], on=['field_id', 'year'], how='inner')


def legend_handles():
    return [Line2D([0], [0], color=COL[g], lw=2.5, label=f'{g}（{n.get(g, 0)}件）')
            for g in ['High', 'Mid', 'Low']]


# ---------------------------------------------------------------- 図1: 全部1枚
fig, ax = plt.subplots(figsize=(12, 6))
for g in ['Mid', 'High', 'Low']:   # Midを下に描く
    for (fid, yr), s in ts[ts['group'] == g].groupby(['field_id', 'year']):
        ax.plot(s['DAP'], s['ndvi_smooth'], color=COL[g],
                alpha=0.25 if g == 'Mid' else 0.45, lw=0.9)
for g in ['High', 'Low']:   # 群平均（太線）
    mean = ts[ts['group'] == g].groupby('DAP')['ndvi_smooth'].mean()
    ax.plot(mean.index, mean.values, color=COL[g], lw=3.5)
ax.set_xlabel('播種後日数 DAP'); ax.set_ylabel('NDVI（平滑化）')
ax.set_title(f'{city} 全圃場×年のNDVI時系列（太線=群平均）')
ax.grid(ls='--', alpha=0.4)
ax.legend(handles=legend_handles())
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{tag}_ndvi_all.png'), dpi=150)
plt.close()

# ---------------------------------------------------------------- 図2: 年別
yrs = [y for y in TARGET_YEARS if (ts['year'] == y).any()]
fig, axes = plt.subplots(1, len(yrs), figsize=(4.5 * len(yrs), 4.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, yr in zip(axes, yrs):
    sub = ts[ts['year'] == yr]
    for g in ['Mid', 'High', 'Low']:
        for fid, s in sub[sub['group'] == g].groupby('field_id'):
            ax.plot(s['DAP'], s['ndvi_smooth'], color=COL[g],
                    alpha=0.5 if g == 'Mid' else 0.8, lw=1.1)
    ax.set_title(f'{yr}年'); ax.set_xlabel('DAP'); ax.grid(ls='--', alpha=0.4)
axes[0].set_ylabel('NDVI（平滑化）')
axes[-1].legend(handles=legend_handles(), fontsize=8)
plt.suptitle(f'{city} 年別NDVI時系列', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{tag}_ndvi_by_year.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\n出力: {OUT_DIR}')
