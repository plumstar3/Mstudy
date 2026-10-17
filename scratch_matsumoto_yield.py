import sqlite3, pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt, matplotlib.font_manager as fm

for _fn in ['IPAexGothic','IPAGothic','MS Gothic','Yu Gothic','Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn; break
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# ── データ読み込み ─────────────────────────────────────────────────
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce')
matsumoto_ids   = geo[geo['muniCd'] == 20202]['field_id'].unique()

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
q = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date
    FROM Questionaire
    WHERE yield IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
conn.close()

q = q.dropna(subset=['field_id'])
q['field_id'] = q['field_id'].astype(int)
q['year']     = q['year'].astype(int)

df = q[q['field_id'].isin(matsumoto_ids)].copy()
print(f'松本市 圃場×年: {len(df)} 件 / 圃場数: {df["field_id"].nunique()}')
print(df.groupby('year')['yield'].describe().round(1))

# ── 年ごとの High/Low 定義（上下33%）─────────────────────────────
QUANTILE = 0.33
df['group'] = 'Mid'
for yr in [2015, 2016, 2017, 2018]:
    mask = df['year'] == yr
    sub  = df[mask]
    if len(sub) < 6: continue
    lo = sub['yield'].quantile(QUANTILE)
    hi = sub['yield'].quantile(1 - QUANTILE)
    df.loc[mask & (df['yield'] <= lo), 'group'] = 'Low'
    df.loc[mask & (df['yield'] >= hi), 'group'] = 'High'

COL = {'High': '#E74C3C', 'Mid': '#95A5A6', 'Low': '#3498DB'}

# ── 図1: 年別収量の箱ひげ図 + ストリッププロット（High/Low色分け）
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
years = sorted(df['year'].unique())
all_yields = df['yield'].values

for i, yr in enumerate(years):
    ax = axes[i]
    sub = df[df['year'] == yr].sort_values('yield')
    # 箱ひげ
    bp = ax.boxplot(sub['yield'], patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    boxprops=dict(facecolor='#EEEEEE', alpha=0.6), widths=0.5)
    # 各圃場の点（High/Low/Mid で色分け）
    for _, row in sub.iterrows():
        ax.scatter(1 + np.random.uniform(-0.15, 0.15), row['yield'],
                   color=COL[row['group']], s=60, zorder=5, alpha=0.85,
                   edgecolors='white', lw=0.5)
    # 閾値ライン
    lo_thr = sub['yield'].quantile(QUANTILE)
    hi_thr = sub['yield'].quantile(1 - QUANTILE)
    ax.axhline(hi_thr, color=COL['High'], lw=1.2, ls='--', alpha=0.7)
    ax.axhline(lo_thr, color=COL['Low'],  lw=1.2, ls='--', alpha=0.7)

    n_h = (sub['group']=='High').sum()
    n_l = (sub['group']=='Low').sum()
    ax.set_title(f'{yr}年\nHigh={n_h} / Low={n_l}', fontsize=10)
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.grid(axis='y', ls='--', alpha=0.4)

axes[0].set_ylabel('収量 (kg/10a)', fontsize=10)

# 凡例
from matplotlib.patches import Patch
handles = [Patch(color=COL['High'], label='High群（上位33%）'),
           Patch(color=COL['Mid'],  label='Mid（中間）'),
           Patch(color=COL['Low'],  label='Low群（下位33%）')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle('松本市 — 全圃場の収量（年別）\n点: 各圃場、破線: High/Low 閾値', fontsize=12)
plt.tight_layout(rect=[0, 0.06, 1, 1])
out1 = 'outputs/yield_intra_analysis/matsumoto_yield_by_year.png'
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f'保存: {out1}')

# ── 図2: 全期間の収量ヒストグラム（High/Low/Mid 色分け積み上げ）
fig2, ax2 = plt.subplots(figsize=(8, 5))
bins = np.linspace(df['yield'].min() - 10, df['yield'].max() + 10, 18)
for grp, color in [('Low','#3498DB'), ('Mid','#95A5A6'), ('High','#E74C3C')]:
    vals = df[df['group']==grp]['yield']
    ax2.hist(vals, bins=bins, color=color, alpha=0.75, label=grp, edgecolor='white')
ax2.axvline(df['yield'].median(), color='black', lw=1.5, ls='--', label=f'全体中央値 {df["yield"].median():.0f}')
ax2.set_xlabel('収量 (kg/10a)', fontsize=11)
ax2.set_ylabel('圃場数', fontsize=11)
ax2.set_title('松本市 — 収量の分布（2015〜2018年合算）', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(axis='y', ls='--', alpha=0.4)
out2 = 'outputs/yield_intra_analysis/matsumoto_yield_hist.png'
plt.tight_layout()
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f'保存: {out2}')
