"""
22_soilmoisture_all_plot.py
===========================
地域を限定せず、土壌水分(VWC)と収量が揃う全圃場×年について
VWC時系列を多収(High)/低収(Low)で色分けしてプロットする（全体 + 年別）。

  - DAP = 観測日 - seed_date（日数）、0 <= DAP <= harvest_dap
  - グループ: 年ごとの収量 上位33% = High, 下位33% = Low, 他 = Mid（21と同じ）
  - 図には個別線に加え、High/Low のDAP別（5日ビン）平均線を重ねる

出力: outputs/soil_analysis/22_all_vwc_all.png / 22_all_vwc_by_year.png / 22_all_vwc_summary.csv
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
OUT_DIR = os.path.join(_ROOT, 'outputs', 'soil_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_Q = 0.33
BIN = 5
COL = {'High': '#E74C3C', 'Low': '#3498DB', 'Mid': '#B0B0B0'}
DARK = {'High': '#A93226', 'Low': '#1B4F72'}

conn = sqlite3.connect(FIELD_DB)
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
q = pd.read_sql("""SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
                   WHERE yield IS NOT NULL AND field_id IS NOT NULL
                     AND seed_date IS NOT NULL AND harvest_date IS NOT NULL""", conn)
conn.close()

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce')
sm = sm.dropna(subset=['field_id']); sm['field_id'] = sm['field_id'].astype(int)
sm['year'] = sm['year'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce')
sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['date', 'VWC'])
q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
q['seed_date'] = pd.to_datetime(q['seed_date'], errors='coerce')
q['harvest_date'] = pd.to_datetime(q['harvest_date'], errors='coerce')
q = q.dropna(subset=['seed_date', 'harvest_date'])
q['harvest_dap'] = (q['harvest_date'] - q['seed_date']).dt.days
q = q[q['harvest_dap'] > 0]

reg = sm[['field_id', 'year']].drop_duplicates().merge(q, on=['field_id', 'year']).reset_index(drop=True)
print(f'VWC×収量 圃場×年: {len(reg)} 件 / 年別 {reg.groupby("year").size().to_dict()}')

reg['group'] = 'Mid'
for yr in sorted(reg['year'].unique()):
    m = reg['year'] == yr
    if m.sum() < 3:
        continue
    lo, hi = reg.loc[m, 'yield'].quantile(GROUP_Q), reg.loc[m, 'yield'].quantile(1 - GROUP_Q)
    reg.loc[m & (reg['yield'] <= lo), 'group'] = 'Low'
    reg.loc[m & (reg['yield'] >= hi), 'group'] = 'High'

# 観測を一括でDAP化
d = sm.merge(reg[['field_id', 'year', 'seed_date', 'harvest_dap', 'group', 'yield']], on=['field_id', 'year'])
d['DAP'] = (d['date'] - d['seed_date']).dt.days
d = d[(d['DAP'] >= 0) & (d['DAP'] <= d['harvest_dap'])]
d = d.groupby(['field_id', 'year', 'group', 'yield', 'DAP'], as_index=False)['VWC'].mean()
cnt = d.groupby(['field_id', 'year']).size()
d = d.merge(cnt[cnt >= 2].rename('n').reset_index(), on=['field_id', 'year'])
d = d.sort_values(['field_id', 'year', 'DAP'])

keys = d[['field_id', 'year', 'group']].drop_duplicates()
n = keys['group'].value_counts()
print(f'プロット {len(keys)}件 / グループ {n.to_dict()}')
print(keys.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())

d.groupby(['field_id', 'year', 'group', 'yield']).agg(
    n_obs=('VWC', 'size'), vwc_mean=('VWC', 'mean')).reset_index().to_csv(
    os.path.join(OUT_DIR, '22_all_vwc_summary.csv'), index=False, encoding='utf-8-sig')


def handles(nn):
    return [Line2D([0], [0], color=COL[g], lw=2.5, label=f'{g}（{nn.get(g, 0)}件）')
            for g in ['High', 'Mid', 'Low']] + \
           [Line2D([0], [0], color='k', lw=3, label='High/Low 5日ビン平均')]


def draw(ax, sub):
    for g in ['Mid', 'High', 'Low']:
        for _, s in sub[sub['group'] == g].groupby(['field_id', 'year']):
            ax.plot(s['DAP'], s['VWC'], color=COL[g], lw=0.7,
                    alpha=0.18 if g == 'Mid' else 0.35)
    sub = sub.assign(b=(sub['DAP'] // BIN) * BIN + BIN / 2)
    for g in ['High', 'Low']:
        m = sub[sub['group'] == g].groupby('b')['VWC'].agg(['mean', 'size'])
        m = m[m['size'] >= 5]
        ax.plot(m.index, m['mean'], color=DARK[g], lw=2.8)


fig, ax = plt.subplots(figsize=(13, 6.5))
draw(ax, d)
ax.set_xlabel('播種後日数 DAP'); ax.set_ylabel('VWC（土壌水分）')
ax.set_title(f'全地域 土壌水分時系列（{len(keys)}圃場×年、赤=High / 青=Low / 灰=Mid）')
ax.grid(ls='--', alpha=0.4); ax.legend(handles=handles(n))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '22_all_vwc_all.png'), dpi=150)
plt.close()

yrs = sorted(keys['year'].unique())
fig, axes = plt.subplots(1, len(yrs), figsize=(6 * len(yrs), 5), sharey=True)
axes = np.atleast_1d(axes)
for a, yr in zip(axes, yrs):
    draw(a, d[d['year'] == yr])
    a.set_title(f'{yr}年（{int((keys["year"] == yr).sum())}件）')
    a.set_xlabel('DAP'); a.grid(ls='--', alpha=0.4)
axes[0].set_ylabel('VWC（土壌水分）')
axes[-1].legend(handles=handles(n), fontsize=8)
plt.suptitle('全地域 年別 土壌水分時系列', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '22_all_vwc_by_year.png'), dpi=150, bbox_inches='tight')
plt.close()
print('出力:', OUT_DIR)
