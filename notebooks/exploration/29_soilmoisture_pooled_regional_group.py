"""
29_soilmoisture_pooled_regional_group.py
=========================================
25 と同じグループ定義（市町村×年の上位/下位33% → 全地域プール）で土壌水分(VWC)を比較する。

[解析]  (11 と同じ DAP% 5%ビン方式 + 08 と同じ3期間平均 / VWC は生値)
  A. DAP% 5%ビンごとに圃場平均VWC を作り(同一圃場×ビンの複数日は平均して擬似反復を回避)、
     High vs Low を MWU（各群 n>=5）。FDR補正も併記。
  B. 3期間平均（early 0-33 / mid 33-67 / late 67-100）の MWU, rank-biserial, cell内Wilcoxon

[出力]  outputs/yield_pooled_regional/
    29_vwc_ts_pooled.png, 29_vwc_bin_stats.csv, 29_vwc_period_stats.csv
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
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
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_QUANTILE = 0.33
MIN_RECORDS    = 12
MIN_PER_YEAR   = 6
MIN_PER_GROUP  = 5
ALPHA          = 0.05
BIN_W          = 5
BIN_EDGES      = np.arange(0, 100 + BIN_W, BIN_W)
BIN_CENTERS    = BIN_EDGES[:-1] + BIN_W / 2
PERIODS = {'early(0-33)': (0, 33), 'mid(33-67)': (33, 67), 'late(67-100)': (67, 100.0001)}
COL_HIGH, COL_LOW, COL_SIG = '#E74C3C', '#3498DB', '#AAAAAA'

# ── データ ───────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
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

# ── グループ定義（25と同一） ──────────────────────────────────
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
sel = q[q['group'].isin(['High', 'Low'])].copy()
sel['cell'] = sel['muniCd'].astype(str) + '_' + sel['year'].astype(str)

df = sm.merge(sel[['field_id', 'year', 'group', 'cell', 'seed_date', 'growing_days']],
              on=['field_id', 'year'], how='inner')
df['DAP_pct'] = (df['date'] - df['seed_date']).dt.days / df['growing_days'] * 100
df = df[(df['DAP_pct'] >= 0) & (df['DAP_pct'] <= 100)]
fy = df.drop_duplicates(['field_id', 'year'])
print(f'VWCあり圃場×年: High={(fy.group == "High").sum()} / Low={(fy.group == "Low").sum()} '
      f'/ 市町村 {fy["cell"].str.split("_").str[0].nunique()}  (年別: {fy.groupby("year").size().to_dict()})')

# ── A. DAP%ビン ──────────────────────────────────────────────
df['bin'] = np.minimum(((df['DAP_pct']) // BIN_W).astype(int), len(BIN_CENTERS) - 1)
fb = df.groupby(['field_id', 'year', 'group', 'cell', 'bin'], as_index=False)['VWC'].mean()
rows = []
for i, ctr in enumerate(BIN_CENTERS):
    s = fb[fb['bin'] == i]
    h, l = s.loc[s.group == 'High', 'VWC'], s.loc[s.group == 'Low', 'VWC']
    p = mannwhitneyu(h, l, alternative='two-sided')[1] if len(h) >= MIN_PER_GROUP and len(l) >= MIN_PER_GROUP else np.nan
    rows.append({'DAP_pct_center': ctr, 'n_high': len(h), 'n_low': len(l),
                 'high_mean': h.mean() if len(h) else np.nan, 'high_se': h.sem() if len(h) > 1 else np.nan,
                 'low_mean': l.mean() if len(l) else np.nan, 'low_se': l.sem() if len(l) > 1 else np.nan,
                 'mwu_p': p})
bs = pd.DataFrame(rows)
v = bs['mwu_p'].notna()
bs.loc[v, 'mwu_q'] = multipletests(bs.loc[v, 'mwu_p'], method='fdr_bh')[1]
bs['sig'] = np.where(bs['mwu_p'].isna(), 'n.a.', np.where(bs['mwu_p'] < 0.01, '**', np.where(bs['mwu_p'] < ALPHA, '*', 'ns')))
bs.to_csv(os.path.join(OUT_DIR, '29_vwc_bin_stats.csv'), index=False, encoding='utf-8-sig')
print(f'[ビン別] 検定可能 {v.sum()} / 補正なし p<0.05: {(bs.mwu_p < ALPHA).sum()} / FDR<0.05: {(bs.mwu_q < ALPHA).sum()} '
      f'/ 最小p={bs.mwu_p.min():.4f}')
print(bs[bs['sig'].isin(['*', '**'])][['DAP_pct_center', 'n_high', 'n_low', 'high_mean', 'low_mean', 'mwu_p', 'mwu_q']].round(4).to_string(index=False))

fig, (ax, axn) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
ok = bs['n_high'] >= MIN_PER_GROUP
okl = bs['n_low'] >= MIN_PER_GROUP
for t, r in bs.iterrows():
    if r['sig'] in ('*', '**'):
        ax.axvspan(r['DAP_pct_center'] - BIN_W / 2, r['DAP_pct_center'] + BIN_W / 2, color=COL_SIG, alpha=0.35, zorder=0)
x = bs['DAP_pct_center']
ax.plot(x[ok], bs.loc[ok, 'high_mean'], color=COL_HIGH, lw=2, label='High')
ax.fill_between(x[ok], (bs.high_mean - bs.high_se)[ok], (bs.high_mean + bs.high_se)[ok], color=COL_HIGH, alpha=0.2)
ax.plot(x[okl], bs.loc[okl, 'low_mean'], color=COL_LOW, lw=2, label='Low')
ax.fill_between(x[okl], (bs.low_mean - bs.low_se)[okl], (bs.low_mean + bs.low_se)[okl], color=COL_LOW, alpha=0.2)
ax.set_ylabel('VWC'); ax.legend(); ax.grid(True, ls='--', alpha=0.4)
ax.set_title('土壌水分(VWC)推移: 地域内上下33%プール High vs Low（灰帯=補正なし p<0.05, ±SE）')
axn.plot(x, bs['n_high'], color=COL_HIGH); axn.plot(x, bs['n_low'], color=COL_LOW)
axn.axhline(MIN_PER_GROUP, ls=':', color='gray'); axn.set_ylabel('n圃場'); axn.set_xlabel('相対生育ステージ DAP%')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '29_vwc_ts_pooled.png'), dpi=200); plt.close()

# ── B. 3期間平均 ─────────────────────────────────────────────
prow = []
for pn, (lo, hi) in PERIODS.items():
    s = df[(df['DAP_pct'] >= lo) & (df['DAP_pct'] < hi)].groupby(
        ['field_id', 'year', 'group', 'cell'], as_index=False)['VWC'].mean()
    h, l = s.loc[s.group == 'High', 'VWC'], s.loc[s.group == 'Low', 'VWC']
    if len(h) < MIN_PER_GROUP or len(l) < MIN_PER_GROUP:
        prow.append({'period': pn, 'n_high': len(h), 'n_low': len(l)}); continue
    u, pv = mannwhitneyu(h, l, alternative='two-sided')
    d = []
    for _, g in s.groupby('cell'):
        gh, gl = g.loc[g.group == 'High', 'VWC'], g.loc[g.group == 'Low', 'VWC']
        if len(gh) and len(gl):
            d.append(gh.mean() - gl.mean())
    d = np.array(d)
    try:
        pc = wilcoxon(d)[1] if len(d) >= 6 and np.any(d != 0) else np.nan
    except Exception:
        pc = np.nan
    prow.append({'period': pn, 'n_high': len(h), 'n_low': len(l), 'high_mean': h.mean(), 'low_mean': l.mean(),
                 'mwu_p': pv, 'rank_biserial': 2 * u / (len(h) * len(l)) - 1,
                 'n_cells': len(d), 'cell_pos_ratio': (d > 0).mean() if len(d) else np.nan, 'cell_wilcoxon_p': pc})
ps = pd.DataFrame(prow)
ps['sig'] = np.where(ps['mwu_p'].isna(), 'n.a.', np.where(ps['mwu_p'] < 0.01, '**', np.where(ps['mwu_p'] < ALPHA, '*', 'ns')))
ps.to_csv(os.path.join(OUT_DIR, '29_vwc_period_stats.csv'), index=False, encoding='utf-8-sig')
print('\n[3期間平均]'); print(ps.round(4).to_string(index=False))
print('完了')
