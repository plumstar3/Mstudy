"""
17_spacing_intra_analysis.py
============================
空間特徴量（between_lines=条間, between_stocks=株間）の「同一地域内」
多収 vs 低収圃場比較。

[対象地域] 条間・株間が数値で揃っている圃場×年の件数が最多の市町村（TARGET_MUNI で手動指定可）
[グループ] 市町村内で年ごとに収量上位33%→High、下位33%→Low（全年プール）
[検定]    Mann-Whitney U（連続値として）。値の分布を High/Mid/Low で色分けした積み上げ棒グラフも出力。

[出力] outputs/spacing_intra_analysis/
    17_muni_candidates.csv, 17_{muniCd}_{city}_stats.csv,
    17_{muniCd}_{city}_distribution.png, 17_{muniCd}_{city}_boxplot.png
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
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
GEOCODE_CSV = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'spacing_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MUNI    = 19209
TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33
MIN_PER_GROUP  = 3

COLS   = ['between_lines', 'between_stocks']
LABELS = {'between_lines': '条間 (cm)', 'between_stocks': '株間 (cm)'}
COLORS = {'High': '#E74C3C', 'Mid': '#BDC3C7', 'Low': '#3498DB'}

# ---------------------------------------------------------------- データ読み込み
conn = sqlite3.connect(FIELD_DB)
quest = pd.read_sql("""
    SELECT field_id, year, yield, between_lines, between_stocks
    FROM Questionaire
    WHERE yield IS NOT NULL AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
conn.close()
quest['field_id'] = quest['field_id'].astype(int)
quest['year']     = quest['year'].astype(int)
n_raw = len(quest)
for c in COLS:
    quest[c] = pd.to_numeric(quest[c], errors='coerce')
print(f'収量あり {n_raw} 件。数値変換後の欠損: ' + ', '.join(f'{c}={quest[c].isna().sum()}' for c in COLS))

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
geo = geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id')

df = quest.merge(geo, on='field_id', how='left').dropna(subset=['muniCd']).copy()
df['muniCd'] = df['muniCd'].astype(int)
df['sp_full'] = df[COLS].notna().all(axis=1)

cand = (df.groupby(['muniCd', 'city'])
          .agg(n_yield=('yield', 'size'),
               n_lines=('between_lines', 'count'),
               n_stocks=('between_stocks', 'count'),
               n_full=('sp_full', 'sum'))
          .reset_index().sort_values('n_full', ascending=False).reset_index(drop=True))
cand.to_csv(os.path.join(OUT_DIR, '17_muni_candidates.csv'), index=False, encoding='utf-8-sig')
print('\n市町村別 件数（上位10）:')
print(cand.head(10).to_string(index=False))

if TARGET_MUNI is None:
    muni_cd, city = int(cand.loc[0, 'muniCd']), str(cand.loc[0, 'city'])
else:
    muni_cd = TARGET_MUNI
    city = str(cand.loc[cand['muniCd'] == muni_cd, 'city'].iloc[0])
print(f'\n★ 対象: {city} ({muni_cd})')

mdf = df[df['muniCd'] == muni_cd].copy().reset_index(drop=True)
print(f'  収量あり {len(mdf)} 件 / 条間あり {mdf["between_lines"].notna().sum()} / 株間あり {mdf["between_stocks"].notna().sum()}')

# ---------------------------------------------------------------- グループ定義
mdf['group'] = 'Mid'
for yr in TARGET_YEARS:
    m = mdf['year'] == yr
    sub = mdf[m]
    if len(sub) < 6:
        continue
    lo, hi = sub['yield'].quantile(GROUP_QUANTILE), sub['yield'].quantile(1 - GROUP_QUANTILE)
    mdf.loc[m & (mdf['yield'] <= lo), 'group'] = 'Low'
    mdf.loc[m & (mdf['yield'] >= hi), 'group'] = 'High'
print('\n年別グループ件数:')
print(mdf.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())

# ---------------------------------------------------------------- 検定
rows = []
for c in COLS:
    h = mdf.loc[mdf['group'] == 'High', c].dropna()
    l = mdf.loc[mdf['group'] == 'Low',  c].dropna()
    p = np.nan
    if len(h) >= MIN_PER_GROUP and len(l) >= MIN_PER_GROUP and pd.concat([h, l]).nunique() > 1:
        try:
            p = mannwhitneyu(h, l, alternative='two-sided')[1]
        except Exception:
            pass
    rows.append({'feature': c, 'n_high': len(h), 'n_low': len(l),
                 'high_mean': h.mean(), 'low_mean': l.mean(),
                 'high_median': h.median(), 'low_median': l.median(), 'mwu_p': p})
    print(f'\n[{c}] High n={len(h)} mean={h.mean():.1f} med={h.median():.1f} | '
          f'Low n={len(l)} mean={l.mean():.1f} med={l.median():.1f} | MWU p={p:.4f}')
    print('  値の分布（High/Mid/Low）:')
    print(pd.crosstab(mdf[c], mdf['group']).reindex(columns=['High', 'Mid', 'Low'], fill_value=0).to_string())
stats = pd.DataFrame(rows)
tag = f'17_{muni_cd}_{city}'
stats.to_csv(os.path.join(OUT_DIR, f'{tag}_stats.csv'), index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 図: 分布（積み上げ）
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, c in zip(axes, COLS):
    ct = (pd.crosstab(mdf[c], mdf['group'])
            .reindex(columns=['High', 'Mid', 'Low'], fill_value=0).sort_index())
    xs = np.arange(len(ct))
    bottom = np.zeros(len(ct))
    for g in ['High', 'Mid', 'Low']:
        ax.bar(xs, ct[g].values, bottom=bottom, color=COLORS[g], label=g, edgecolor='white')
        bottom += ct[g].values
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{v:g}' for v in ct.index], rotation=45)
    ax.set_xlabel(LABELS[c]); ax.set_ylabel('件数')
    ax.set_title(f'{city} {LABELS[c]} の分布（N={int(ct.values.sum())}）')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{tag}_distribution.png'), dpi=150)
plt.close()

# ---------------------------------------------------------------- 図: 箱ひげ
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, c in zip(axes, COLS):
    h = mdf.loc[mdf['group'] == 'High', c].dropna()
    l = mdf.loc[mdf['group'] == 'Low',  c].dropna()
    ax.boxplot([h, l], labels=[f'High\n(n={len(h)})', f'Low\n(n={len(l)})'], showmeans=True)
    p = stats.loc[stats['feature'] == c, 'mwu_p'].iloc[0]
    ax.set_title(f'{LABELS[c]}  MWU p={p:.3f}' if not np.isnan(p) else f'{LABELS[c]}  MWU n.a.')
    ax.set_ylabel(LABELS[c]); ax.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{tag}_boxplot.png'), dpi=150)
plt.close()
print(f'\n出力: {OUT_DIR}')
