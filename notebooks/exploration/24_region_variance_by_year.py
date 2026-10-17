"""
24_region_variance_by_year.py
==============================
「同じ地域の圃場は、全体の中で同じ位置（多収寄り/低収寄り）にまとまっているか」を
年ごとに分けて確認する。

[手順]（年ごとに独立に実施。年効果は年を分けることで除去）
  1. その年の NDVI保有全圃場を全体プールとし、収量の上位33%/下位33%の境界を求める
  2. 1年あたり MIN_N 件以上ある市町村のみを対象とする
  3. 指標
     - η²   : 収量分散のうち地域差で説明される割合（一元配置分散分析）
     - ICC(1): 同地域の圃場同士の収量の類似度（不均衡データ対応）
     - Kruskal-Wallis検定: 地域間で収量分布に差があるか
     - 一致度: 地域ごとに「最頻の全体グループ(High/Mid/Low)に属する圃場の割合」
  4. 地域別の箱ひげ図（中央値順）＋全体の上位/下位境界線

[出力]  outputs/yield_intra_analysis/
    24_region_variance_by_year.png
    24_region_variance_summary.csv   (年ごとの η² / ICC / KW p値)
    24_region_concordance.csv        (年×市町村の一致度)
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kruskal
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

TARGET_YEARS = [2015, 2016, 2017, 2018]
Q            = 0.33
MIN_N        = 4      # その年に地域内で最低この件数ある市町村のみ対象

# ── データ ───────────────────────────────────────────────────
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
key = set(zip(ndvi['field_id'].astype(int), ndvi['year'].astype(int)))
q = q[[(f, y) in key for f, y in zip(q['field_id'], q['year'])]].dropna(subset=['muniCd']).reset_index(drop=True)
print(f'対象レコード（NDVI保有）: {len(q)} 件')


def anova_icc(groups):
    """一元配置分散分析から η² と ICC(1)（不均衡データ対応）を算出"""
    allv = np.concatenate(groups)
    N, k = len(allv), len(groups)
    gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    eta2 = ssb / (ssb + ssw)
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return eta2, icc


COL = {'High': '#E74C3C', 'Mid': '#BDC3C7', 'Low': '#3498DB'}
summary, conc_rows = [], []
fig, axes = plt.subplots(2, 2, figsize=(22, 13), facecolor='white')
fig.suptitle('年ごとの地域別収量分布（全体の上位/下位33%境界との比較）', fontsize=14, fontweight='bold')

for ax, yr in zip(axes.flatten(), TARGET_YEARS):
    d = q[q['year'] == yr].copy()
    hi, lo = d['yield'].quantile(1 - Q), d['yield'].quantile(Q)
    d['gg'] = np.where(d['yield'] >= hi, 'High', np.where(d['yield'] <= lo, 'Low', 'Mid'))

    cnt = d.groupby('muniCd').size()
    sel = cnt[cnt >= MIN_N].index
    ds = d[d['muniCd'].isin(sel)].copy()
    groups = [g['yield'].values for _, g in ds.groupby('muniCd')]
    eta2, icc = anova_icc(groups)
    kw_p = kruskal(*groups).pvalue
    summary.append({'year': yr, 'n_total_year': len(d), 'n_regions': len(groups),
                    'n_fields_in_regions': len(ds), 'eta2': eta2, 'ICC1': icc,
                    'kruskal_p': kw_p, 'global_hi_thr': hi, 'global_lo_thr': lo})
    print(f'[{yr}] 地域数={len(groups)} 圃場={len(ds)}  η²={eta2:.3f}  ICC={icc:.3f}  KW p={kw_p:.4g}')

    order = ds.groupby(['muniCd', 'city'])['yield'].median().sort_values().index.tolist()
    for i, (m, c) in enumerate(order):
        g = ds[ds['muniCd'] == m]
        # 最頻グループへの一致度
        vc = g['gg'].value_counts()
        conc_rows.append({'year': yr, 'muniCd': int(m), 'city': c, 'n': len(g),
                          'modal_group': vc.index[0], 'concordance_%': vc.iloc[0] / len(g) * 100,
                          'High': int((g.gg == 'High').sum()), 'Mid': int((g.gg == 'Mid').sum()),
                          'Low': int((g.gg == 'Low').sum())})
        ax.boxplot(g['yield'].values, positions=[i], widths=0.6, showfliers=False,
                   medianprops=dict(color='black', lw=1.8))
        for gg in ['High', 'Mid', 'Low']:
            yy = g.loc[g.gg == gg, 'yield']
            ax.scatter(i + np.random.default_rng(i).uniform(-0.2, 0.2, len(yy)), yy,
                       s=22, color=COL[gg], edgecolors='gray', lw=0.3, zorder=3)
    ax.axhline(hi, ls='--', color=COL['High'], lw=1.2, label=f'全体 上位33%境界 ({hi:.0f})')
    ax.axhline(lo, ls='--', color=COL['Low'],  lw=1.2, label=f'全体 下位33%境界 ({lo:.0f})')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f'{c}\n(n={int((ds.muniCd == m).sum())})' for m, c in order],
                       rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('収量 (kg/10a)')
    ax.set_title(f'{yr}年  η²={eta2:.2f}  ICC={icc:.2f}  Kruskal-Wallis p={kw_p:.3g}', fontsize=11)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
png = os.path.join(OUT_DIR, '24_region_variance_by_year.png')
fig.savefig(png, dpi=150, bbox_inches='tight')
plt.close(fig)

sdf = pd.DataFrame(summary)
cdf = pd.DataFrame(conc_rows)
sdf.to_csv(os.path.join(OUT_DIR, '24_region_variance_summary.csv'), index=False, encoding='utf-8-sig')
cdf.to_csv(os.path.join(OUT_DIR, '24_region_concordance.csv'), index=False, encoding='utf-8-sig')

print('\n=== 一致度（最頻の全体グループに属する割合, 年別・件数加重平均） ===')
for yr, g in cdf.groupby('year'):
    w = (g['concordance_%'] * g['n']).sum() / g['n'].sum()
    print(f'  {yr}: {w:.1f}%（ランダムなら約{(1/3)*100+ 0:.0f}〜50%程度）')
print('\n=== 地域別 一致度 ===')
print(cdf.round(1).to_string(index=False))
print(f'\n図 -> {png}\n完了')
