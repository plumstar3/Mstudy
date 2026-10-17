"""
30_vwc_pooled_individual_plot.py
=================================
29 の分析で使った High / Low 圃場の VWC 推移を、圃場ごとの個別線で描画する。
  左: High 圃場, 右: Low 圃場（細線=各圃場、太線=5%ビン平均）, 下段: 重ね合わせ(平均のみ)

[出力] outputs/yield_pooled_regional/30_vwc_individual.png, 30_vwc_individual_by_year.png
"""
import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'yield_pooled_regional')

GROUP_QUANTILE, MIN_RECORDS, MIN_PER_YEAR, BIN_W = 0.33, 12, 6, 5
COL = {'High': '#E74C3C', 'Low': '#3498DB'}
YEAR_COL = {2016: '#1b9e77', 2017: '#d95f02', 2018: '#7570b3'}

# ── 29 と同じデータ・グループ定義 ────────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018""", conn)
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
conn.close()
q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
q['seed_date'] = pd.to_datetime(q['seed_date'])
q['growing_days'] = (pd.to_datetime(q['harvest_date']) - q['seed_date']).dt.days
q = q[q['growing_days'] > 0]
sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce')
sm = sm.dropna(subset=['field_id']); sm['field_id'] = sm['field_id'].astype(int)
sm['year'] = sm['year'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce')
sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['date', 'VWC'])

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd'] = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
q = q.merge(geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id'), on='field_id', how='left')
ndvi = pd.read_csv(NDVI_CSV, usecols=['field_id', 'year'])
key = set(zip(ndvi['field_id'].astype(int), ndvi['year'].astype(int)))
q = q[[(f, y) in key for f, y in zip(q['field_id'], q['year'])]].reset_index(drop=True)

cnt = q.groupby('muniCd').size()
q = q[q['muniCd'].isin(cnt[cnt >= MIN_RECORDS].index)].copy()
q['group'] = 'Mid'
for (muni, yr), g in q.groupby(['muniCd', 'year']):
    if len(g) < MIN_PER_YEAR:
        q.loc[g.index, 'group'] = 'Skip'
        continue
    lo, hi = g['yield'].quantile(GROUP_QUANTILE), g['yield'].quantile(1 - GROUP_QUANTILE)
    q.loc[g.index[g['yield'] <= lo], 'group'] = 'Low'
    q.loc[g.index[g['yield'] >= hi], 'group'] = 'High'
sel = q[q['group'].isin(['High', 'Low'])]
df = sm.merge(sel[['field_id', 'year', 'group', 'seed_date', 'growing_days', 'muniCd', 'city']],
              on=['field_id', 'year'], how='inner')
df['DAP_pct'] = (df['date'] - df['seed_date']).dt.days / df['growing_days'] * 100
df = df[(df['DAP_pct'] >= 0) & (df['DAP_pct'] <= 100)]
df['fy'] = df['field_id'].astype(str) + '_' + df['year'].astype(str)
n = df.drop_duplicates('fy').groupby('group').size()
print(n.to_dict())

edges = np.arange(0, 100 + BIN_W, BIN_W); ctr = edges[:-1] + BIN_W / 2
def bin_mean(d):
    b = np.minimum((d['DAP_pct'] // BIN_W).astype(int), len(ctr) - 1)
    fb = d.assign(bin=b).groupby(['fy', 'bin'])['VWC'].mean().reset_index()
    return fb.groupby('bin')['VWC'].agg(['mean', 'sem', 'count']).reindex(range(len(ctr)))

# ── 図1: 個別線 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
for ax, g in zip(axes[:2], ['High', 'Low']):
    d = df[df['group'] == g]
    for _, s in d.groupby('fy'):
        s = s.sort_values('DAP_pct')
        ax.plot(s['DAP_pct'], s['VWC'], color=COL[g], alpha=0.25, lw=0.8)
    m = bin_mean(d)
    ax.plot(ctr, m['mean'], color='black', lw=2.5, label='5%ビン平均')
    ax.set_title(f'{g}圃場（n={n[g]}）'); ax.set_xlabel('相対生育ステージ DAP%')
    ax.grid(True, ls='--', alpha=0.4); ax.legend()
axes[0].set_ylabel('VWC')
ax = axes[2]
for g in ['High', 'Low']:
    m = bin_mean(df[df['group'] == g]); ok = m['count'] >= 5
    ax.plot(ctr[ok], m['mean'][ok], color=COL[g], lw=2.5, label=f'{g} (n={n[g]})')
    ax.fill_between(ctr[ok], (m['mean'] - m['sem'])[ok], (m['mean'] + m['sem'])[ok], color=COL[g], alpha=0.2)
ax.set_title('平均の重ね合わせ（±SE, ビン内 n>=5）'); ax.set_xlabel('相対生育ステージ DAP%')
ax.grid(True, ls='--', alpha=0.4); ax.legend()
fig.suptitle('土壌水分(VWC)推移: 地域内上下33%プール 圃場別（細線=各圃場）', fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '30_vwc_individual.png'), dpi=170); plt.close()

# ── 図2: 年別に色分け ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, g in zip(axes, ['High', 'Low']):
    d = df[df['group'] == g]
    for _, s in d.groupby('fy'):
        s = s.sort_values('DAP_pct')
        ax.plot(s['DAP_pct'], s['VWC'], color=YEAR_COL[int(s['year'].iloc[0])], alpha=0.35, lw=0.8)
    for yr, c in YEAR_COL.items():
        ax.plot([], [], color=c, label=f'{yr} (n={d[d.year == yr]["fy"].nunique()})')
    ax.set_title(f'{g}圃場 年別'); ax.set_xlabel('相対生育ステージ DAP%')
    ax.grid(True, ls='--', alpha=0.4); ax.legend()
axes[0].set_ylabel('VWC')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '30_vwc_individual_by_year.png'), dpi=170); plt.close()
print('完了')
