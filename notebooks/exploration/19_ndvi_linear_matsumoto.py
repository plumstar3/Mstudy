"""
19_ndvi_linear_matsumoto.py
===========================
松本市のNDVI時系列を、スプライン補間・SGフィルタ・端点アンカー(0.15)を使わず、
生の観測値（hls_50m_ndvi_timeseries.csv）を線形補間のみで結んでプロットする。

  - DAP = 観測日 - seed_date（06_save_ndvi_daily.py と同じ定義）
  - 生育期間内（0 <= DAP <= harvest_dap）の観測のみ使用
  - 補間は最初〜最後の観測の間のみ（外挿しない）
  - High/Low/Mid は 18_ndvi_intra_plot.py の松本市グループ定義と同じ

出力: outputs/ndvi_intra_analysis/
    19_20202_matsumoto_linear_all.png      全圃場×年を1枚（観測点マーカー付き）
    19_20202_matsumoto_linear_by_year.png  年別パネル
    19_20202_matsumoto_linear_daily.csv    線形補間した日次値
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

_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_CSV   = os.path.join(_ROOT, 'data', 'raw', 'hls_50m_ndvi_timeseries.csv')
FIELD_DB  = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
GEOCODE   = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR   = os.path.join(_ROOT, 'outputs', 'ndvi_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

MUNI_CD, CITY = 20202, '松本市'
TARGET_YEARS  = [2015, 2016, 2017, 2018]
GROUP_Q       = 0.33
COL = {'High': '#E74C3C', 'Low': '#3498DB', 'Mid': '#B0B0B0'}

# ---------------------------------------------------------------- データ
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
                   WHERE yield IS NOT NULL AND field_id IS NOT NULL
                     AND year BETWEEN 2015 AND 2018""", conn)
conn.close()
q['field_id'] = q['field_id'].astype(int)
q['year']     = q['year'].astype(int)
q['seed_date']    = pd.to_datetime(q['seed_date'], errors='coerce')
q['harvest_date'] = pd.to_datetime(q['harvest_date'], errors='coerce')

geo = pd.read_csv(GEOCODE, encoding='utf-8-sig')
geo['muniCd'] = pd.to_numeric(geo['muniCd'], errors='coerce')
fids = set(geo.loc[geo['muniCd'] == MUNI_CD, 'field_id'].astype(int))
q = q[q['field_id'].isin(fids)].copy().reset_index(drop=True)

# グループ（18と同じ定義: 年ごと上位/下位33%）
q['group'] = 'Mid'
for yr in TARGET_YEARS:
    m = q['year'] == yr
    if m.sum() < 6:
        continue
    lo, hi = q.loc[m, 'yield'].quantile(GROUP_Q), q.loc[m, 'yield'].quantile(1 - GROUP_Q)
    q.loc[m & (q['yield'] <= lo), 'group'] = 'Low'
    q.loc[m & (q['yield'] >= hi), 'group'] = 'High'

raw = pd.read_csv(RAW_CSV, parse_dates=['date'])
raw['field_id'] = raw['field_id'].astype(int)

# ---------------------------------------------------------------- 線形補間
series, daily_rows, skipped = {}, [], []
for _, r in q.iterrows():
    fid, yr = int(r['field_id']), int(r['year'])
    if pd.isna(r['seed_date']) or pd.isna(r['harvest_date']):
        skipped.append((fid, yr, '播種/収穫日なし')); continue
    hd = (r['harvest_date'] - r['seed_date']).days
    o = raw[raw['field_id'] == fid].copy()
    o['DAP'] = (o['date'] - r['seed_date']).dt.days
    o = o[(o['DAP'] >= 0) & (o['DAP'] <= hd)]
    o = o.groupby('DAP', as_index=False)['ndvi'].mean().sort_values('DAP')
    if len(o) < 2:
        skipped.append((fid, yr, f'期間内観測{len(o)}点')); continue
    d = np.arange(int(o['DAP'].min()), int(o['DAP'].max()) + 1)
    v = np.interp(d, o['DAP'], o['ndvi'])
    series[(fid, yr)] = (o, d, v, r['group'])
    daily_rows += [{'field_id': fid, 'year': yr, 'group': r['group'], 'DAP': a, 'ndvi_linear': b}
                   for a, b in zip(d, v)]

print(f'{CITY}: 圃場×年 {len(q)} 件 → プロット {len(series)} 件 / 除外 {len(skipped)} 件')
for s in skipped:
    print('  除外:', s)
n_obs = pd.Series({k: len(v[0]) for k, v in series.items()})
print(f'実観測点数/件: 平均{n_obs.mean():.1f} 最小{n_obs.min()} 最大{n_obs.max()}')
n = pd.Series([v[3] for v in series.values()]).value_counts()
print('グループ件数:', n.to_dict())

pd.DataFrame(daily_rows).to_csv(os.path.join(OUT_DIR, '19_20202_matsumoto_linear_daily.csv'),
                                index=False, encoding='utf-8-sig')


def handles():
    return [Line2D([0], [0], color=COL[g], lw=2.5, label=f'{g}（{n.get(g, 0)}件）')
            for g in ['High', 'Mid', 'Low']]


def draw(ax, keys):
    for g in ['Mid', 'High', 'Low']:
        for k in keys:
            o, d, v, gg = series[k]
            if gg != g:
                continue
            ax.plot(d, v, color=COL[g], lw=1.0, alpha=0.45 if g == 'Mid' else 0.7)
            ax.plot(o['DAP'], o['ndvi'], 'o', color=COL[g], ms=3,
                    alpha=0.5 if g == 'Mid' else 0.9)


# ---------------------------------------------------------------- 図1
fig, ax = plt.subplots(figsize=(12, 6))
draw(ax, list(series))
ax.set_xlabel('播種後日数 DAP'); ax.set_ylabel('NDVI（観測値 + 線形補間）')
ax.set_title(f'{CITY} NDVI時系列（生観測値の線形補間、●=実観測）')
ax.grid(ls='--', alpha=0.4); ax.legend(handles=handles())
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '19_20202_matsumoto_linear_all.png'), dpi=150)
plt.close()

# ---------------------------------------------------------------- 図2 年別
yrs = [y for y in TARGET_YEARS if any(k[1] == y for k in series)]
fig, axes = plt.subplots(1, len(yrs), figsize=(4.5 * len(yrs), 4.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, yr in zip(axes, yrs):
    draw(ax, [k for k in series if k[1] == yr])
    ax.set_title(f'{yr}年'); ax.set_xlabel('DAP'); ax.grid(ls='--', alpha=0.4)
axes[0].set_ylabel('NDVI（観測値 + 線形補間）')
axes[-1].legend(handles=handles(), fontsize=8)
plt.suptitle(f'{CITY} 年別NDVI（線形補間）', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '19_20202_matsumoto_linear_by_year.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'出力: {OUT_DIR}')
