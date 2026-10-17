"""
27_ndvi_pooled_regional_group.py
=================================
25 と同じグループ定義（市町村×年の上位/下位33% → 全地域プール）で NDVI を比較する。

[解析]  (07: 時系列, 08: 3期間平均 と同じ方法 / NDVI は生値)
  A. NDVI時系列（相対生育ステージ 0〜100% に正規化）
     High/Low の平均 ± 95%CI、各時刻 MWU + FDR補正 → 有意区間を灰帯で表示
  B. NDVI 3期間平均（early 0-33 / mid 33-67 / late 67-100）: MWU
     補助: 市町村×年セル内の (High平均 - Low平均) の Wilcoxon, rank-biserial 効果量

[出力]  outputs/yield_pooled_regional/
    27_ndvi_ts_pooled.png
    27_ndvi_period_stats.csv
    27_ndvi_period_bar.png
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
N_RESAMPLE     = 100
ALPHA          = 0.05
PERIODS = {'early(0-33)': (0, 33), 'mid(33-67)': (33, 67), 'late(67-100)': (67, 100.0001)}
COL_HIGH, COL_LOW, COL_SIG = '#E74C3C', '#3498DB', '#AAAAAA'
RELATIVE_PTS = np.linspace(0, 100, N_RESAMPLE)

# ── データ ───────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
conn.close()
q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
q['growing_days'] = (pd.to_datetime(q['harvest_date']) - pd.to_datetime(q['seed_date'])).dt.days
q = q[q['growing_days'] > 0]

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd'] = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
q = q.merge(geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id'), on='field_id', how='left')

ndvi = pd.read_csv(NDVI_CSV)
ndvi['field_id'] = ndvi['field_id'].astype(int); ndvi['year'] = ndvi['year'].astype(int)
key = set(zip(ndvi['field_id'], ndvi['year']))
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
sel = q[q['group'].isin(['High', 'Low'])].reset_index(drop=True)
sel['cell'] = sel['muniCd'].astype(str) + '_' + sel['year'].astype(str)
print(f'High={(sel.group == "High").sum()} / Low={(sel.group == "Low").sum()}')

# ── NDVI 相対ステージ正規化 + 3期間平均 ───────────────────────
nd = {k: g for k, g in ndvi.groupby(['field_id', 'year'])}
ts = np.full((len(sel), N_RESAMPLE), np.nan)
per_rows = []
for i, r in sel.iterrows():
    nf = nd.get((int(r['field_id']), int(r['year'])))
    row = {}
    if nf is not None and len(nf) >= 2:
        rel = nf['DAP'].values / r['growing_days'] * 100
        v = nf['ndvi_smooth'].values.astype(float)
        o = np.argsort(rel)
        ts[i] = np.interp(RELATIVE_PTS, rel[o], v[o])
        for pn, (lo, hi) in PERIODS.items():
            m = (rel >= lo) & (rel < hi)
            row[f'ndvi_{pn}'] = float(np.nanmean(v[m])) if m.sum() else np.nan
    per_rows.append(row)
sel = pd.concat([sel, pd.DataFrame(per_rows)], axis=1)
mh, ml = (sel['group'] == 'High').values, (sel['group'] == 'Low').values

# ── A. 時系列 ────────────────────────────────────────────────
mu_h, mu_l = np.nanmean(ts[mh], 0), np.nanmean(ts[ml], 0)
ci_h = 1.96 * np.nanstd(ts[mh], 0) / np.sqrt(mh.sum())
ci_l = 1.96 * np.nanstd(ts[ml], 0) / np.sqrt(ml.sum())
p = np.ones(N_RESAMPLE)
for t in range(N_RESAMPLE):
    h, l = ts[mh, t], ts[ml, t]
    h, l = h[~np.isnan(h)], l[~np.isnan(l)]
    if len(h) >= 3 and len(l) >= 3:
        p[t] = mannwhitneyu(h, l, alternative='two-sided')[1]
pc = multipletests(p, method='fdr_bh')[1]
sig = pc < ALPHA
print(f'[NDVI時系列] 有意時刻(FDR): {sig.mean() * 100:.0f}%   補正前p<0.05: {(p < ALPHA).mean() * 100:.0f}%   最小p={p.min():.4f}')

fig, ax = plt.subplots(figsize=(10, 5))
start = None
for t in range(N_RESAMPLE + 1):
    on = t < N_RESAMPLE and sig[t]
    if on and start is None:
        start = t
    elif not on and start is not None:
        ax.axvspan(RELATIVE_PTS[start], RELATIVE_PTS[t - 1], color=COL_SIG, alpha=0.3, zorder=0)
        start = None
ax.plot(RELATIVE_PTS, mu_h, color=COL_HIGH, lw=2, label=f'High (n={mh.sum()})')
ax.fill_between(RELATIVE_PTS, mu_h - ci_h, mu_h + ci_h, color=COL_HIGH, alpha=0.2)
ax.plot(RELATIVE_PTS, mu_l, color=COL_LOW, lw=2, label=f'Low (n={ml.sum()})')
ax.fill_between(RELATIVE_PTS, mu_l - ci_l, mu_l + ci_l, color=COL_LOW, alpha=0.2)
ax.set_xlabel('相対生育ステージ (%)'); ax.set_ylabel('NDVI（SG平滑化済み）')
ax.set_title(f'NDVI時系列: 地域内上下33%プール High vs Low（灰帯=FDR<0.05, 有意時刻 {sig.mean() * 100:.0f}%）')
ax.legend(); ax.grid(True, ls='--', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '27_ndvi_ts_pooled.png'), dpi=200); plt.close()

# ── B. 3期間平均 ─────────────────────────────────────────────
rows = []
for pn in PERIODS:
    c = f'ndvi_{pn}'
    h, l = sel.loc[mh, c].dropna(), sel.loc[ml, c].dropna()
    u, pv = mannwhitneyu(h, l, alternative='two-sided')
    d = []
    for _, g in sel.groupby('cell'):
        gh, gl = g.loc[g['group'] == 'High', c].dropna(), g.loc[g['group'] == 'Low', c].dropna()
        if len(gh) and len(gl):
            d.append(gh.mean() - gl.mean())
    d = np.array(d)
    try:
        pcell = wilcoxon(d)[1] if len(d) >= 6 and np.any(d != 0) else np.nan
    except Exception:
        pcell = np.nan
    rows.append({'period': pn, 'n_high': len(h), 'n_low': len(l),
                 'high_mean': h.mean(), 'low_mean': l.mean(),
                 'high_se': h.sem(), 'low_se': l.sem(),
                 'mwu_p': pv, 'rank_biserial': 2 * u / (len(h) * len(l)) - 1,
                 'n_cells': len(d), 'cell_pos_ratio': (d > 0).mean(), 'cell_wilcoxon_p': pcell})
st = pd.DataFrame(rows)
st['mwu_q'] = multipletests(st['mwu_p'], method='fdr_bh')[1]
st['sig'] = np.where(st['mwu_p'] < 0.01, '**', np.where(st['mwu_p'] < ALPHA, '*', 'ns'))
st.to_csv(os.path.join(OUT_DIR, '27_ndvi_period_stats.csv'), index=False, encoding='utf-8-sig')
print(st.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, (_, r) in zip(axes, st.iterrows()):
    ax.bar(['High', 'Low'], [r['high_mean'], r['low_mean']], yerr=[r['high_se'], r['low_se']],
           capsize=5, color=[COL_HIGH, COL_LOW], alpha=0.85, width=0.5)
    ax.set_title(f'NDVI {r["period"]}', fontsize=10); ax.set_ylim(bottom=0)
    ax.grid(axis='y', ls='--', alpha=0.5)
    ax.text(0.97, 0.97, f'{r["sig"]} p={r["mwu_p"]:.3f}\nn={int(r["n_high"])}/{int(r["n_low"])}',
            ha='right', va='top', transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8))
plt.suptitle('NDVI 3期間平均: High vs Low（平均±SE）')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '27_ndvi_period_bar.png'), dpi=200); plt.close()
print('完了')
