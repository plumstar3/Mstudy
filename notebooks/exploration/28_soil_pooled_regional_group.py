"""
28_soil_pooled_regional_group.py
=================================
25 と同じグループ定義（市町村×年の上位/下位33% → 全地域プール）で土壌特徴量を比較する。

[採取タイミング] 09 と同じ: 播種前後(5〜7月) / 収穫前後(10〜12月) を別々に解析（8〜9月は除外）
[検定] 09 と同じ: Mann-Whitney U（生値）、判定は補正なし p (*:<0.05, **:<0.01)
       補助: rank-biserial 効果量, FDR q値（タイミング内）, 市町村×年セル内 Wilcoxon

[出力]  outputs/yield_pooled_regional/
    28_soil_presow.png, 28_soil_postharv.png, 28_soil_stats.csv
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
TIMINGS = {'presow': [5, 6, 7], 'postharv': [10, 11, 12]}
TIMING_LABEL = {'presow': '播種前後(5-7月)', 'postharv': '収穫前後(10-12月)'}
SOIL_COLS = ['pH', 'CaO', 'MgO', 'K2O', 'P2O2', 'N_able', 'N_per', 'C_per', 'CEC',
             'H3PO4', 'B_able', 'Mo_able', 'Mn_c']
COL_HIGH, COL_LOW = '#E74C3C', '#3498DB'

# ── データ ───────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""
    SELECT field_id, year, yield FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
solid = pd.read_sql(f"""SELECT field_id, year, date, {', '.join(SOIL_COLS)} FROM Solid
                        WHERE field_id IS NOT NULL AND date IS NOT NULL""", conn)
conn.close()
for d in (q, solid):
    d['field_id'] = d['field_id'].astype(int); d['year'] = d['year'].astype(int)
solid['month'] = pd.to_datetime(solid['date']).dt.month
for c in SOIL_COLS:
    solid[c] = pd.to_numeric(solid[c], errors='coerce')

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
sel = q[q['group'].isin(['High', 'Low'])].reset_index(drop=True)
sel['cell'] = sel['muniCd'].astype(str) + '_' + sel['year'].astype(str)
print(f'High={(sel.group == "High").sum()} / Low={(sel.group == "Low").sum()}')

# ── タイミング別に解析 ───────────────────────────────────────
all_rows = []
for tm, months in TIMINGS.items():
    s = solid[solid['month'].isin(months)].groupby(['field_id', 'year'], as_index=False)[SOIL_COLS].mean()
    d = sel.merge(s, on=['field_id', 'year'], how='inner')
    mh, ml = (d['group'] == 'High').values, (d['group'] == 'Low').values
    print(f'\n=== {TIMING_LABEL[tm]}: High={mh.sum()} / Low={ml.sum()} / 市町村 {d["muniCd"].nunique()} ===')
    rows = []
    for c in SOIL_COLS:
        h, l = d.loc[mh, c].dropna(), d.loc[ml, c].dropna()
        if len(h) < MIN_PER_GROUP or len(l) < MIN_PER_GROUP or pd.concat([h, l]).nunique() < 2:
            rows.append({'timing': tm, 'variable': c, 'n_high': len(h), 'n_low': len(l), 'mwu_p': np.nan})
            continue
        u, p = mannwhitneyu(h, l, alternative='two-sided')
        diffs = []
        for _, g in d.groupby('cell'):
            gh, gl = g.loc[g['group'] == 'High', c].dropna(), g.loc[g['group'] == 'Low', c].dropna()
            if len(gh) and len(gl):
                diffs.append(gh.mean() - gl.mean())
        diffs = np.array(diffs)
        try:
            pcell = wilcoxon(diffs)[1] if len(diffs) >= 6 and np.any(diffs != 0) else np.nan
        except Exception:
            pcell = np.nan
        rows.append({'timing': tm, 'variable': c, 'n_high': len(h), 'n_low': len(l),
                     'high_median': h.median(), 'low_median': l.median(),
                     'high_mean': h.mean(), 'low_mean': l.mean(),
                     'mwu_p': p, 'rank_biserial': 2 * u / (len(h) * len(l)) - 1,
                     'n_cells': len(diffs), 'cell_wilcoxon_p': pcell})
    st = pd.DataFrame(rows)
    v = st['mwu_p'].notna()
    st.loc[v, 'mwu_q'] = multipletests(st.loc[v, 'mwu_p'], method='fdr_bh')[1]
    st['sig'] = np.where(st['mwu_p'].isna(), 'n.a.',
                         np.where(st['mwu_p'] < 0.01, '**', np.where(st['mwu_p'] < ALPHA, '*', 'ns')))
    print(st[['variable', 'n_high', 'n_low', 'high_median', 'low_median', 'rank_biserial',
              'mwu_p', 'mwu_q', 'cell_wilcoxon_p', 'sig']].round(4).to_string(index=False))
    print(f'有意(p<{ALPHA}, 補正なし): {(st["mwu_p"] < ALPHA).sum()} / {v.sum()}   (FDR<{ALPHA}: {(st["mwu_q"] < ALPHA).sum()})')
    all_rows.append(st)

    # 箱ひげ + ジッター
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.flatten()
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, SOIL_COLS):
        h, l = d.loc[mh, c].dropna(), d.loc[ml, c].dropna()
        ax.boxplot([h, l], positions=[1, 2], widths=0.5, showfliers=False)
        for x0, vv, col in [(1, h, COL_HIGH), (2, l, COL_LOW)]:
            ax.scatter(rng.uniform(x0 - .15, x0 + .15, len(vv)), vv, s=18, color=col, alpha=0.7, zorder=3)
        r = st[st['variable'] == c].iloc[0]
        ptxt = 'n.a.' if np.isnan(r['mwu_p']) else f'{r["sig"]} p={r["mwu_p"]:.3f}'
        ax.set_xticks([1, 2]); ax.set_xticklabels([f'High\n(n={len(h)})', f'Low\n(n={len(l)})'], fontsize=8)
        ax.set_title(f'{c}  {ptxt}', fontsize=9); ax.grid(axis='y', ls='--', alpha=0.4)
    for ax in axes[len(SOIL_COLS):]:
        ax.set_visible(False)
    fig.suptitle(f'土壌特徴量 {TIMING_LABEL[tm]}: 地域内上下33%プール High vs Low', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'28_soil_{tm}.png'), dpi=150, bbox_inches='tight'); plt.close()

pd.concat(all_rows).to_csv(os.path.join(OUT_DIR, '28_soil_stats.csv'), index=False, encoding='utf-8-sig')
print('完了')
