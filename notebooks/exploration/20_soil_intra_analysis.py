"""
20_soil_intra_analysis.py
=========================
土壌特徴量の「同一地域内」多収 vs 低収圃場比較（16_disease_intra_analysis.py と同じ方針）。

[対象地域] 収量と土壌データが揃っている圃場×年の件数が最多の市町村（TARGET_MUNI で手動指定可）
           ※ 採取時期を揃えるため、収穫前後(10〜12月)の採取のみを対象にカウント
[グループ] 市町村内で年ごとに収量上位33%→High、下位33%→Low、他はMid
[検定]    各土壌項目で Mann-Whitney U（High vs Low、全年プール）。標本が小さいため低検出力に注意。
[図]      全点をHigh/Mid/Lowで色分けしたドットプロット（Midは参考表示、検定には使わない）

[出力] outputs/soil_intra_analysis/
    20_muni_candidates.csv, 20_{muniCd}_{city}_stats.csv, 20_{muniCd}_{city}_dotplot.png
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
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'soil_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MUNI      = None
TARGET_YEARS     = [2015, 2016, 2017, 2018]
GROUP_QUANTILE   = 0.33
POST_HARV_MONTHS = [10, 11, 12]
MIN_PER_GROUP    = 3

SOIL_COLS = ['pH', 'CaO', 'MgO', 'K2O', 'P2O2', 'N_able', 'N_per', 'C_per', 'CEC',
             'H3PO4', 'B_able', 'Mo_able', 'Mn_c']
SOIL_LABELS = {
    'pH': 'pH', 'CaO': '石灰 CaO', 'MgO': '苦土 MgO', 'K2O': '加里 K2O', 'P2O2': 'リン酸 P2O2',
    'N_able': '有効態窒素', 'N_per': '全窒素 N(%)', 'C_per': '全炭素 C(%)', 'CEC': 'CEC',
    'H3PO4': 'リン酸吸収', 'B_able': '有効態B', 'Mo_able': '有効態Mo', 'Mn_c': '交換性Mn',
}
COL = {'High': '#E74C3C', 'Low': '#3498DB', 'Mid': '#B0B0B0'}

# ---------------------------------------------------------------- データ
conn = sqlite3.connect(FIELD_DB)
quest = pd.read_sql("""SELECT field_id, year, yield FROM Questionaire
                       WHERE yield IS NOT NULL AND field_id IS NOT NULL
                         AND year BETWEEN 2015 AND 2018""", conn)
solid = pd.read_sql(f"""SELECT field_id, year, date, {', '.join(SOIL_COLS)} FROM Solid
                        WHERE field_id IS NOT NULL AND date IS NOT NULL""", conn)
conn.close()
for d in (quest, solid):
    d['field_id'] = d['field_id'].astype(int)
    d['year']     = d['year'].astype(int)
solid['month'] = pd.to_datetime(solid['date']).dt.month
for c in SOIL_COLS:
    solid[c] = pd.to_numeric(solid[c], errors='coerce')
solid = solid[solid['month'].isin(POST_HARV_MONTHS)]
solid = solid.groupby(['field_id', 'year'], as_index=False)[SOIL_COLS].mean()   # 同一圃場×年の複数採取は平均
solid['n_soil'] = solid[SOIL_COLS].notna().sum(axis=1)

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
geo = geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id')

df = (quest.merge(solid, on=['field_id', 'year'], how='inner')
           .merge(geo, on='field_id', how='left').dropna(subset=['muniCd']))
df['muniCd'] = df['muniCd'].astype(int)
print(f'収穫前後の土壌×収量: {len(df)} 件')

cand = (df.groupby(['muniCd', 'city'])
          .agg(n_field_years=('field_id', 'size'), n_fields=('field_id', 'nunique'),
               mean_n_items=('n_soil', 'mean'))
          .reset_index().sort_values(['n_field_years', 'mean_n_items'], ascending=False)
          .reset_index(drop=True))
cand.to_csv(os.path.join(OUT_DIR, '20_muni_candidates.csv'), index=False, encoding='utf-8-sig')
print('\n市町村別 土壌データ件数（上位10）:')
print(cand.head(10).to_string(index=False))

if TARGET_MUNI is None:
    muni_cd, city = int(cand.loc[0, 'muniCd']), str(cand.loc[0, 'city'])
else:
    muni_cd = TARGET_MUNI
    city = str(cand.loc[cand['muniCd'] == muni_cd, 'city'].iloc[0])
print(f'\n★ 対象: {city} ({muni_cd})')

mdf = df[df['muniCd'] == muni_cd].copy().reset_index(drop=True)

# グループ定義は、土壌がなくても市町村内の全収量データで行う方が本来は妥当だが、
# 16・17と同様「分析に使うデータの範囲内」で年ごとに分位を取る。
mdf['group'] = 'Mid'
for yr in TARGET_YEARS:
    m = mdf['year'] == yr
    if m.sum() < 6:
        continue
    lo, hi = mdf.loc[m, 'yield'].quantile(GROUP_QUANTILE), mdf.loc[m, 'yield'].quantile(1 - GROUP_QUANTILE)
    mdf.loc[m & (mdf['yield'] <= lo), 'group'] = 'Low'
    mdf.loc[m & (mdf['yield'] >= hi), 'group'] = 'High'
print('\n年別グループ件数:')
print(mdf.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())
n = mdf['group'].value_counts()

# ---------------------------------------------------------------- 検定・図
rows = []
ncol = 5
nrow = int(np.ceil(len(SOIL_COLS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.2 * nrow))
axes = np.array(axes).flatten()
rng = np.random.default_rng(0)
for ax, col in zip(axes, SOIL_COLS):
    vals = {g: mdf.loc[mdf['group'] == g, col].dropna() for g in ['High', 'Mid', 'Low']}
    h, l = vals['High'], vals['Low']
    p = np.nan
    if len(h) >= MIN_PER_GROUP and len(l) >= MIN_PER_GROUP and pd.concat([h, l]).nunique() > 1:
        try:
            p = mannwhitneyu(h, l, alternative='two-sided')[1]
        except Exception:
            pass
    sig = 'n.a.' if np.isnan(p) else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    for g, x0 in [('High', 1), ('Mid', 2), ('Low', 3)]:
        v = vals[g]
        ax.scatter(rng.uniform(x0 - .22, x0 + .22, len(v)), v, color=COL[g], s=40,
                   alpha=0.5 if g == 'Mid' else 0.85, edgecolors='white', lw=0.5, zorder=3)
        if len(v):
            ax.hlines(v.median(), x0 - .3, x0 + .3, colors=COL[g], lw=2, zorder=4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f'High\n(n={len(h)})', f'Mid\n(n={len(vals["Mid"])})', f'Low\n(n={len(l)})'], fontsize=8)
    ax.set_xlim(0.4, 3.6)
    p_str = 'n.a.' if np.isnan(p) else f'{p:.3f}'
    ax.set_title(f'{SOIL_LABELS[col]}   MWU {sig} p={p_str}', fontsize=9)
    ax.grid(axis='y', ls='--', alpha=0.4)
    rows.append({'variable': col, 'n_high': len(h), 'n_low': len(l),
                 'high_median': h.median(), 'low_median': l.median(),
                 'high_mean': h.mean(), 'low_mean': l.mean(), 'mwu_p': p, 'sig': sig})
    print(f'  {col:<8}: n={len(h)}/{len(l)}  High中央値={h.median():8.3f} Low中央値={l.median():8.3f}  {sig} p={p_str}')
for ax in axes[len(SOIL_COLS):]:
    ax.set_visible(False)
fig.suptitle(f'{city} 土壌特徴量（収穫前後採取）High / Mid / Low  ※検定はHigh vs Low、横線=中央値', fontsize=12)
plt.tight_layout()
tag = f'20_{muni_cd}_{city}'
plt.savefig(os.path.join(OUT_DIR, f'{tag}_dotplot.png'), dpi=150, bbox_inches='tight')
plt.close()
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'{tag}_stats.csv'), index=False, encoding='utf-8-sig')
print(f'\n出力: {OUT_DIR}')
