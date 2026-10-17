"""
21_soilmoisture_intra_plot.py
=============================
土壌水分(VWC)データが揃っている圃場×年が最多の市町村を自動選択し、
全圃場×年のVWC時系列を多収(High)/低収(Low)で色分けしてプロットする。

  - DAP = 観測日 - seed_date（日数のまま。%正規化はしない）
  - 生育期間内（0 <= DAP <= harvest_dap）の観測のみ
  - グループは 18/19 と同じ: 年ごとの収量 上位33% = High, 下位33% = Low, 他 = Mid

出力: outputs/soil_analysis/
    21_<muniCd>_<city>_vwc_all.png / _by_year.png / _summary.csv
"""
import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
GEOCODE = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR = os.path.join(_ROOT, 'outputs', 'soil_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MUNI_CD = 23212   # 名古屋市
GROUP_Q = 0.33
COL = {'High': '#E74C3C', 'Low': '#3498DB', 'Mid': '#B0B0B0'}

conn = sqlite3.connect(FIELD_DB)
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
q = pd.read_sql("""SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
                   WHERE yield IS NOT NULL AND field_id IS NOT NULL
                     AND seed_date IS NOT NULL AND harvest_date IS NOT NULL""", conn)
conn.close()

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce')
sm = sm.dropna(subset=['field_id']); sm['field_id'] = sm['field_id'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce')
sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['date', 'VWC'])
q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
q['seed_date'] = pd.to_datetime(q['seed_date'], errors='coerce')
q['harvest_date'] = pd.to_datetime(q['harvest_date'], errors='coerce')
q = q.dropna(subset=['seed_date', 'harvest_date'])
q['harvest_dap'] = (q['harvest_date'] - q['seed_date']).dt.days
q = q[q['harvest_dap'] > 0]

geo = pd.read_csv(GEOCODE, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo = geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id')

# 圃場×年（VWC・収量が揃う）
fy = sm[['field_id', 'year']].drop_duplicates()
fy['year'] = fy['year'].astype(int)
fy = fy.merge(q, on=['field_id', 'year']).merge(geo, on='field_id')

cnt = fy.groupby(['muniCd', 'city']).size().sort_values(ascending=False)
print('市町村別 VWC×収量 圃場×年数（上位5）:\n', cnt.head(5).to_string())
muni = TARGET_MUNI_CD if TARGET_MUNI_CD else cnt.index[0][0]
reg = fy[fy['muniCd'] == muni].copy().reset_index(drop=True)
CITY = reg['city'].iloc[0]
print(f'\n選択: {CITY} (muniCd={int(muni)}) {len(reg)}件 / 年別 {reg.groupby("year").size().to_dict()}')

reg['group'] = 'Mid'
for yr in sorted(reg['year'].unique()):
    m = reg['year'] == yr
    if m.sum() < 3:
        continue
    lo, hi = reg.loc[m, 'yield'].quantile(GROUP_Q), reg.loc[m, 'yield'].quantile(1 - GROUP_Q)
    reg.loc[m & (reg['yield'] <= lo), 'group'] = 'Low'
    reg.loc[m & (reg['yield'] >= hi), 'group'] = 'High'

series = {}
for _, r in reg.iterrows():
    fid, yr = int(r['field_id']), int(r['year'])
    o = sm[(sm['field_id'] == fid) & (sm['year'] == yr)].copy()
    o['DAP'] = (o['date'] - r['seed_date']).dt.days
    o = o[(o['DAP'] >= 0) & (o['DAP'] <= r['harvest_dap'])]
    o = o.groupby('DAP', as_index=False)['VWC'].mean().sort_values('DAP')
    if len(o) >= 2:
        series[(fid, yr)] = (o, r['group'], r['yield'])

n = pd.Series([v[1] for v in series.values()]).value_counts()
print(f'プロット {len(series)}件 / グループ {n.to_dict()}')
print('観測点数/件: 平均%.1f 最小%d 最大%d' % (
    np.mean([len(v[0]) for v in series.values()]),
    min(len(v[0]) for v in series.values()), max(len(v[0]) for v in series.values())))

pd.DataFrame([{'field_id': k[0], 'year': k[1], 'group': v[1], 'yield': v[2], 'n_obs': len(v[0]),
               'vwc_mean': v[0]['VWC'].mean()} for k, v in series.items()]
             ).to_csv(os.path.join(OUT_DIR, f'21_{int(muni)}_{CITY}_vwc_summary.csv'),
                      index=False, encoding='utf-8-sig')


def handles():
    return [Line2D([0], [0], color=COL[g], lw=2.5, label=f'{g}（{n.get(g, 0)}件）')
            for g in ['High', 'Mid', 'Low']]


def draw(ax, keys):
    for g in ['Mid', 'High', 'Low']:
        for k in keys:
            o, gg, _ = series[k]
            if gg != g:
                continue
            ax.plot(o['DAP'], o['VWC'], '-o', color=COL[g], lw=0.9, ms=2,
                    alpha=0.35 if g == 'Mid' else 0.7)


fig, ax = plt.subplots(figsize=(12, 6))
draw(ax, list(series))
ax.set_xlabel('播種後日数 DAP'); ax.set_ylabel('VWC（土壌水分）')
ax.set_title(f'{CITY} 土壌水分時系列（観測値、赤=High / 青=Low / 灰=Mid）')
ax.grid(ls='--', alpha=0.4); ax.legend(handles=handles())
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'21_{int(muni)}_{CITY}_vwc_all.png'), dpi=150)
plt.close()

yrs = sorted({k[1] for k in series})
fig, axes = plt.subplots(1, len(yrs), figsize=(5 * len(yrs), 4.5), sharey=True)
axes = np.atleast_1d(axes)
for a, yr in zip(axes, yrs):
    draw(a, [k for k in series if k[1] == yr])
    a.set_title(f'{yr}年'); a.set_xlabel('DAP'); a.grid(ls='--', alpha=0.4)
axes[0].set_ylabel('VWC（土壌水分）')
axes[-1].legend(handles=handles(), fontsize=8)
plt.suptitle(f'{CITY} 年別 土壌水分時系列', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'21_{int(muni)}_{CITY}_vwc_by_year.png'), dpi=150, bbox_inches='tight')
plt.close()
print('出力:', OUT_DIR)
