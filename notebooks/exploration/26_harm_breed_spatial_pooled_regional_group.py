"""
26_harm_breed_spatial_pooled_regional_group.py
===============================================
25 と同じグループ定義（市町村×年の上位/下位33% → 全地域プール）で、
病害・品種・栽植（条間/株間）特徴量を比較する。

[グループ定義] 25_weather_pooled_regional_group.py と同一
[解析]
  - 病害二値(sick/weed/wet/bug), 気象フラグ(6種): 被害率の比較
        プールFisher + 市町村×年で層別した Cochran-Mantel-Haenszel(CMH)検定
  - 病害順序(unripen/lay/loss): プールMWU + Fisher(>=2を被害あり) + cell内Wilcoxon
  - 栽植(between_lines/between_stocks): プールMWU (年別z-score) + cell内Wilcoxon
  - 品種(breed): breed x group の χ²/Fisher, 品種ごとの High 比率（記述）
  - 全検定をまとめて FDR(BH) 補正

[出力]  outputs/yield_pooled_regional/
    26_harm_stats.csv
    26_harm_binary_flag.png
    26_harm_ordinal_cultivar.png
    26_breed.png
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable
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
ALPHA          = 0.05
BREED_MIN_COUNT = 5

HARM_BINARY_COLS    = ['sick', 'weed', 'wet', 'bug']
HARM_BINARY_UNKNOWN = {'sick': [3], 'bug': [4]}
HARM_ORDINAL_COLS   = ['unripen', 'lay', 'loss']
HARM_FLAG_COLS      = ['heavy_rain', 'drought', 'typhoon', 'gale', 'long_rain', 'few_solar']
HARM_ALL            = HARM_BINARY_COLS + HARM_ORDINAL_COLS + HARM_FLAG_COLS
CULTIVAR_COLS       = ['between_lines', 'between_stocks']
COL_HIGH, COL_LOW   = '#E74C3C', '#3498DB'

# ── データ読み込み ───────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date, breed, between_lines, between_stocks
    FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
harm = pd.read_sql(f"SELECT field_id, year, {', '.join(HARM_ALL)} FROM harm WHERE field_id IS NOT NULL", conn)
conn.close()

q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
for c in CULTIVAR_COLS:
    q[c] = pd.to_numeric(q[c], errors='coerce')

harm['field_id'] = harm['field_id'].astype(int); harm['year'] = harm['year'].astype(int)
for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS:
    harm[c] = pd.to_numeric(harm[c], errors='coerce')
for c in HARM_FLAG_COLS:
    harm[c] = harm[c].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    harm[c] = pd.to_numeric(harm[c], errors='coerce').fillna(0).astype(int)
agg = {c: 'mean' for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS}
agg.update({c: 'max' for c in HARM_FLAG_COLS})
harm = harm.groupby(['field_id', 'year'])[HARM_ALL].agg(agg).reset_index()
for c, bad in HARM_BINARY_UNKNOWN.items():
    harm.loc[harm[c].isin(bad), c] = np.nan

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
sel = q[q['group'].isin(['High', 'Low'])].merge(harm, on=['field_id', 'year'], how='left').reset_index(drop=True)
sel['cell'] = sel['muniCd'].astype(str) + '_' + sel['year'].astype(str)
mh, ml = (sel['group'] == 'High').values, (sel['group'] == 'Low').values
print(f'High={mh.sum()} / Low={ml.sum()}  (市町村 {sel["muniCd"].nunique()})')

# ── 検定ユーティリティ ───────────────────────────────────────
def cell_paired_p(col, values=None):
    """市町村×年セル内の (High平均-Low平均) に対する Wilcoxon。"""
    v = sel[col] if values is None else values
    d = []
    for _, g in sel.assign(_v=v).groupby('cell'):
        h = g.loc[g['group'] == 'High', '_v'].dropna(); l = g.loc[g['group'] == 'Low', '_v'].dropna()
        if len(h) and len(l):
            d.append(h.mean() - l.mean())
    d = np.array(d)
    try:
        p = wilcoxon(d)[1] if len(d) >= 6 and np.any(d != 0) else np.nan
    except Exception:
        p = np.nan
    return p, (d > 0).mean() if len(d) else np.nan

def cmh_p(flag):
    """セル層別 CMH。flag: 0/1 Series (NaN可)。"""
    tabs = []
    for _, g in sel.assign(_f=flag).groupby('cell'):
        g = g.dropna(subset=['_f'])
        h, l = g.loc[g['group'] == 'High', '_f'], g.loc[g['group'] == 'Low', '_f']
        if len(h) == 0 or len(l) == 0:
            continue
        t = np.array([[(h == 1).sum(), (h == 0).sum()], [(l == 1).sum(), (l == 0).sum()]], float)
        if t[:, 0].sum() == 0 or t[:, 1].sum() == 0:      # 全員同じ → 情報なし
            continue
        tabs.append(t)
    if len(tabs) < 2:
        return np.nan, np.nan
    try:
        st = StratifiedTable(tabs)
        return st.test_null_odds(correction=True).pvalue, st.oddsratio_pooled
    except Exception:
        return np.nan, np.nan

rows = []
def add(**kw):
    rows.append(kw)

# 二値・フラグ
for c in HARM_BINARY_COLS + HARM_FLAG_COLS:
    if c in HARM_BINARY_COLS:        # 1=なし, >=2=あり
        f = (sel[c] >= 2).astype(float).where(sel[c].notna())
    else:
        f = sel[c].astype(float)
    h, l = f[mh].dropna(), f[ml].dropna()
    if len(h) < 5 or len(l) < 5:
        continue
    tab = [[int((h == 1).sum()), int((h == 0).sum())], [int((l == 1).sum()), int((l == 0).sum())]]
    p = fisher_exact(tab)[1]
    p_cmh, or_cmh = cmh_p(f)
    add(variable=c, type='binary/flag', n_high=len(h), n_low=len(l),
        high_stat=h.mean() * 100, low_stat=l.mean() * 100, p=p, p_strat=p_cmh, strat_effect=or_cmh,
        strat_note='CMH common OR')

# 順序: MWU + Fisher(>=2を被害ありとして二値化)  (08と同じ)
for c in HARM_ORDINAL_COLS:
    h, l = sel.loc[mh, c].dropna(), sel.loc[ml, c].dropna()
    if len(h) < 5 or len(l) < 5:
        continue
    p = mannwhitneyu(h, l, alternative='two-sided')[1]
    f = (sel[c] >= 2).astype(float).where(sel[c].notna())
    p_cmh, or_cmh = cmh_p(f)
    add(variable=c, type='ordinal(MWU)', n_high=len(h), n_low=len(l),
        high_stat=h.mean(), low_stat=l.mean(), p=p, p_strat=p_cmh, strat_effect=or_cmh,
        strat_note='CMH(>=2 vs 1) common OR')
    hb, lb = f[mh].dropna(), f[ml].dropna()
    tab = [[int(hb.sum()), int((hb == 0).sum())], [int(lb.sum()), int((lb == 0).sum())]]
    add(variable=c + '_ge2', type='ordinal(Fisher>=2)', n_high=len(hb), n_low=len(lb),
        high_stat=hb.mean() * 100, low_stat=lb.mean() * 100, p=fisher_exact(tab)[1],
        p_strat=p_cmh, strat_effect=or_cmh, strat_note='CMH common OR')

# 栽植: 生値でMWU（08と同じ）。cell内Wilcoxonは補助
for c in CULTIVAR_COLS:
    h, l = sel.loc[mh, c].dropna(), sel.loc[ml, c].dropna()
    if len(h) < 5 or len(l) < 5:
        continue
    p = mannwhitneyu(h, l, alternative='two-sided')[1]
    p_cell, pos = cell_paired_p(c)
    add(variable=c, type='cultivar', n_high=len(h), n_low=len(l),
        high_stat=h.mean(), low_stat=l.mean(), p=p, p_strat=p_cell, strat_effect=pos,
        strat_note='cell Wilcoxon / 正のセル割合')

# 品種: 記述統計のみ（検定なし, 08と同じ）
sel['breed'] = sel['breed'].astype(str).str.strip()
bc = sel[sel['breed'].notna() & ~sel['breed'].isin(['', 'None', 'nan'])]['breed'].value_counts()
keep = bc[bc >= BREED_MIN_COUNT].index
bsub = sel[sel['breed'].isin(keep)]
ct = pd.crosstab(bsub['breed'], bsub['group'])

st = pd.DataFrame(rows)
# 判定は 07/08 と同じく補正なしの p 値（*: p<0.05, **: p<0.01）。FDR q 値は参考列
st['q'] = multipletests(st['p'], method='fdr_bh')[1]
v = st['p_strat'].notna()
st.loc[v, 'q_strat'] = multipletests(st.loc[v, 'p_strat'], method='fdr_bh')[1]
st['sig'] = np.where(st['p'] < 0.01, '**', np.where(st['p'] < ALPHA, '*', 'ns'))
st.to_csv(os.path.join(OUT_DIR, '26_harm_stats.csv'), index=False, encoding='utf-8-sig')
print(st.round(4).to_string(index=False))
print(f'\n有意(p<{ALPHA}, 補正なし): {(st["p"] < ALPHA).sum()} / {len(st)}   (FDR<{ALPHA}: {(st["q"] < ALPHA).sum()})')

# ── 図 ───────────────────────────────────────────────────────
def pstr(r):
    return f'{r["sig"]} p={r["p"]:.3f}'

# 1) 二値+フラグ: 被害率
b = st[st['type'].isin(['binary/flag', 'ordinal(Fisher>=2)'])].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(b)); w = 0.38
ax.bar(x - w / 2, b['high_stat'], w, color=COL_HIGH, label='High', alpha=0.85)
ax.bar(x + w / 2, b['low_stat'], w, color=COL_LOW, label='Low', alpha=0.85)
for i, r in b.iterrows():
    ax.text(i, max(r['high_stat'], r['low_stat']) + 1, f'{r["sig"]}\np={r["p"]:.2f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(b['variable'], rotation=30, ha='right')
ax.set_ylabel('該当率 (%)'); ax.legend(); ax.grid(axis='y', ls='--', alpha=0.5)
ax.set_title('病害・気象フラグの該当率: 地域内上下33%プール High vs Low')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '26_harm_binary_flag.png'), dpi=200); plt.close()

# 2) 順序 + 栽植
fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
for ax, c in zip(axes, HARM_ORDINAL_COLS + CULTIVAR_COLS):
    r = st[(st['variable'] == c) & (st['type'] != 'ordinal(Fisher>=2)')]
    h, l = sel.loc[mh, c].dropna(), sel.loc[ml, c].dropna()
    ax.bar(['High', 'Low'], [h.mean(), l.mean()], yerr=[h.sem(), l.sem()], capsize=5,
           color=[COL_HIGH, COL_LOW], alpha=0.85, width=0.5)
    ax.set_title(c + (' (cm)' if c in CULTIVAR_COLS else ' (1-4)'), fontsize=10)
    ax.grid(axis='y', ls='--', alpha=0.5)
    if len(r):
        ax.text(0.97, 0.97, pstr(r.iloc[0]), ha='right', va='top', transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
plt.suptitle('病害順序変数・栽植: High vs Low（平均±SE）')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, '26_harm_ordinal_cultivar.png'), dpi=200); plt.close()

# 3) 品種
if ct.shape[0] >= 2:
    ct2 = ct.copy(); ct2['n'] = ct2.sum(axis=1); ct2['High_%'] = ct2['High'] / ct2['n'] * 100
    ct2 = ct2.sort_values('n', ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(ct2) + 3), 4.5))
    ax.bar(ct2.index, ct2['High_%'], color='#9B59B6', alpha=0.8)
    ax.axhline(50, ls='--', color='gray')
    for i, (n, hp) in enumerate(zip(ct2['n'], ct2['High_%'])):
        ax.text(i, hp + 1, f'n={n}', ha='center', fontsize=8)
    ax.set_ylabel('High の割合 (%) [High+Low中]'); ax.set_ylim(0, 105)
    ax.set_title('品種別 High 比率（記述統計のみ・検定なし）')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '26_breed.png'), dpi=200); plt.close()
print('完了')
