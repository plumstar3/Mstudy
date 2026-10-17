"""
23_regional_vs_global_group.py
===============================
市町村内（地域内）で多収/低収に分類された圃場が、
全体（全国プール）で見ても多収/低収なのかを確認する。

[グループ定義]
  地域内: 08_yield_intra_analysis.py と同じ
          市町村 × 年 の収量 上位33% → High / 下位33% → Low（その年6件以上）
  全体  : 07_yield_group_analysis.py と同じ
          NDVI保有の全圃場 × 年 の収量 上位33% → High / 下位33% → Low / 他 Mid
  対象市町村: NDVI保有レコード数 >= MIN_RECORDS（08と同じ）

[出力]  outputs/yield_intra_analysis/
    23_regional_vs_global.csv          (圃場×年ごとの地域内/全体グループ・全体パーセンタイル)
    23_regional_vs_global_summary.csv  (市町村 × 地域内グループ別の内訳)
    23_regional_vs_global.png          (内訳の積み上げ棒 + 全体パーセンタイル分布)
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
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'yield_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33
MIN_RECORDS    = 12

# ── データ読み込み（08と同じ母集団） ─────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""
    SELECT field_id, year, yield FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
conn.close()
q['field_id'] = q['field_id'].astype(int)
q['year']     = q['year'].astype(int)

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
q = q.merge(geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id'),
            on='field_id', how='left')

ndvi = pd.read_csv(NDVI_CSV, usecols=['field_id', 'year'])
ndvi_key = set(zip(ndvi['field_id'].astype(int), ndvi['year'].astype(int)))
q = q[[(f, y) in ndvi_key for f, y in zip(q['field_id'], q['year'])]].reset_index(drop=True)
print(f'対象レコード（NDVI保有）: {len(q)} 件')

# ── 全体グループ（年ごと全国プール）と全体パーセンタイル ───────
q['global_pct']   = q.groupby('year')['yield'].rank(pct=True) * 100   # 年内パーセンタイル
q['global_group'] = 'Mid'
for yr, g in q.groupby('year'):
    lo = g['yield'].quantile(GROUP_QUANTILE)
    hi = g['yield'].quantile(1 - GROUP_QUANTILE)
    q.loc[g.index[g['yield'] <= lo], 'global_group'] = 'Low'
    q.loc[g.index[g['yield'] >= hi], 'global_group'] = 'High'

# ── 地域内グループ ───────────────────────────────────────────
cnt = q.groupby('muniCd').size()
target = cnt[cnt >= MIN_RECORDS].index
q = q[q['muniCd'].isin(target)].copy()
q['regional_group'] = 'Mid'
for (muni, yr), g in q.groupby(['muniCd', 'year']):
    if len(g) < 6:
        continue
    lo = g['yield'].quantile(GROUP_QUANTILE)
    hi = g['yield'].quantile(1 - GROUP_QUANTILE)
    q.loc[g.index[g['yield'] <= lo], 'regional_group'] = 'Low'
    q.loc[g.index[g['yield'] >= hi], 'regional_group'] = 'High'

reg = q[q['regional_group'].isin(['High', 'Low'])].copy()
reg.to_csv(os.path.join(OUT_DIR, '23_regional_vs_global.csv'), index=False, encoding='utf-8-sig')
print(f'解析対象市町村: {len(target)} / 地域内 High+Low: {len(reg)} 件')

# ── 集計 ─────────────────────────────────────────────────────
def breakdown(df):
    n = len(df)
    return pd.Series({
        'n': n,
        'global_High_%': (df['global_group'] == 'High').mean() * 100,
        'global_Mid_%':  (df['global_group'] == 'Mid').mean() * 100,
        'global_Low_%':  (df['global_group'] == 'Low').mean() * 100,
        'mean_global_pct': df['global_pct'].mean(),
        'median_global_pct': df['global_pct'].median(),
    })

overall = reg.groupby('regional_group').apply(breakdown)
print('\n=== 全市町村合計 ===')
print(overall.round(1).to_string())

summ = reg.groupby(['muniCd', 'city', 'regional_group']).apply(breakdown).reset_index()
summ.to_csv(os.path.join(OUT_DIR, '23_regional_vs_global_summary.csv'),
            index=False, encoding='utf-8-sig')
print('\n=== 市町村別 ===')
print(summ.round(1).to_string(index=False))

# ── 図 ───────────────────────────────────────────────────────
COL = {'High': '#E74C3C', 'Mid': '#BDC3C7', 'Low': '#3498DB'}
munis = (reg.groupby(['muniCd', 'city']).size().sort_values(ascending=False).index.tolist())
labels = [f'{c}\n(n={len(reg[reg.muniCd == m])})' for m, c in munis]

fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='white',
                         gridspec_kw={'width_ratios': [1, 1, 1.4]})
fig.suptitle('地域内の多収/低収圃場は、全体（全国プール）でも多収/低収か？', fontsize=14, fontweight='bold')

for ax, rg in zip(axes[:2], ['High', 'Low']):
    bottoms = np.zeros(len(munis))
    for gg in ['High', 'Mid', 'Low']:
        vals = []
        for m, c in munis:
            d = reg[(reg.muniCd == m) & (reg.regional_group == rg)]
            vals.append((d['global_group'] == gg).mean() * 100 if len(d) else 0)
        ax.bar(range(len(munis)), vals, bottom=bottoms, color=COL[gg], label=f'全体 {gg}')
        bottoms += np.array(vals)
    ax.set_xticks(range(len(munis)))
    ax.set_xticklabels([c for _, c in munis], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('割合 (%)')
    ov = overall.loc[rg]
    ax.set_title(f'地域内 {rg} 圃場の全体での内訳\n'
                 f'(全体High {ov["global_High_%"]:.0f}% / Mid {ov["global_Mid_%"]:.0f}% / '
                 f'Low {ov["global_Low_%"]:.0f}%)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')

# 全体パーセンタイル分布（市町村別）
ax = axes[2]
for i, (m, c) in enumerate(munis):
    for rg, off in [('High', -0.18), ('Low', 0.18)]:
        d = reg[(reg.muniCd == m) & (reg.regional_group == rg)]['global_pct']
        ax.scatter(np.full(len(d), i + off) + np.random.default_rng(i).uniform(-0.08, 0.08, len(d)),
                   d, s=14, alpha=0.6, color=COL[rg])
        if len(d):
            ax.hlines(d.median(), i + off - 0.15, i + off + 0.15, color='black', lw=1.5)
ax.axhline(67, ls='--', color='gray', lw=0.8); ax.axhline(33, ls='--', color='gray', lw=0.8)
ax.set_xticks(range(len(munis)))
ax.set_xticklabels([c for _, c in munis], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('全体での収量パーセンタイル（年内, %）')
ax.set_title('全体パーセンタイル分布（赤=地域内High / 青=地域内Low / 黒線=中央値）', fontsize=10)
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
out_png = os.path.join(OUT_DIR, '23_regional_vs_global.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n図 -> {out_png}\n完了')
