"""
yield_distribution_cdf_matched.py
=====================================================================
目的:
  Questionaireに対応する78市町村のMAFFデータ（2014-2018年）と
  現場収量（Questionaire 2015-2018年）の分布を比較する。

出力:
  outputs/yield_pred_v3/yield_distribution_cdf.png  (上書き更新)
"""
import sqlite3, os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

os.chdir(r'c:\Users\amilu\Projects\vsCodeFile\Mstudy')
OUT_DIR = os.path.join('outputs', 'yield_pred_v3')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Questionaireデータ（2015-2018） ──────────────────────────────────────
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
quest = pd.read_sql('''
    SELECT field_id, year, CAST(yield AS REAL) AS yield
    FROM Questionaire
    WHERE year BETWEEN 2015 AND 2018
      AND yield IS NOT NULL AND CAST(yield AS REAL) > 0''', conn)
conn.close()
quest = quest.dropna(subset=['field_id'])
quest['field_id'] = quest['field_id'].astype(int)
quest['yield']    = pd.to_numeric(quest['yield'], errors='coerce')
quest = quest.dropna(subset=['yield'])

# ─── 2. 市町村コードを付与 ─────────────────────────────────────────────────────
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce')
geo = geo[['field_id', 'muniCd']].drop_duplicates('field_id')

quest = quest.merge(geo, on='field_id', how='left')
matched_muniCds = quest['muniCd'].dropna().unique()
print(f'Questionaireに対応する市町村数: {len(matched_muniCds)}')
print(f'Questionaireサンプル数: {len(quest)}')

# ─── 3. MAFFデータ（対応78市町村 × 2014-2018年） ─────────────────────────────
maff_raw = pd.read_csv('data/raw/summary-soy-2010-2018.csv', encoding='cp932')
maff_raw.columns = ['year', 'city', 'maff_yield', 'muniCd']
maff_raw['maff_yield'] = pd.to_numeric(maff_raw['maff_yield'], errors='coerce')
maff_raw['muniCd']     = pd.to_numeric(maff_raw['muniCd'],     errors='coerce')
maff_raw = maff_raw.dropna(subset=['maff_yield', 'muniCd'])

# 対応市町村かつ2014-2018年に絞る
maff_matched = maff_raw[
    (maff_raw['muniCd'].isin(matched_muniCds)) &
    (maff_raw['year'].between(2014, 2018))
].copy()
print(f'MAFFサンプル数（78市町村 × 2014-2018年）: {len(maff_matched)}')

# ─── 4. 統計サマリー ──────────────────────────────────────────────────────────
y_field = quest['yield'].dropna().values
y_maff  = maff_matched['maff_yield'].values

print()
print('=== 分布統計サマリー ===')
print(f'                  現場（Questionaire）  MAFF（78市町村）')
print(f'  サンプル数:         {len(y_field):>6}              {len(y_maff):>6}')
print(f'  平均 (kg/10a):   {y_field.mean():>8.1f}          {y_maff.mean():>8.1f}')
print(f'  中央値 (kg/10a): {np.median(y_field):>8.1f}          {np.median(y_maff):>8.1f}')
print(f'  標準偏差:        {y_field.std():>8.1f}          {y_maff.std():>8.1f}')
print(f'  最小:            {y_field.min():>8.1f}          {y_maff.min():>8.1f}')
print(f'  最大:            {y_field.max():>8.1f}          {y_maff.max():>8.1f}')
print(f'  平均の差:        {y_field.mean() - y_maff.mean():>+8.1f} kg/10a')

# ─── 5. 可視化（PDF + CDF + 箱ひげ図） ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    '収量分布比較\n現場データ（Questionaire 2015-2018） vs MAFF市町村平均（78市町村 2014-2018）',
    fontsize=12, fontweight='bold'
)

COLOR_FIELD = '#2980b9'   # 青：現場
COLOR_MAFF  = '#e74c3c'   # 赤：MAFF

x_range = np.linspace(0, 600, 1000)

# ── 左: PDF（確率密度関数）──────────────────────────────────────────────────
ax = axes[0]
kde_field = gaussian_kde(y_field, bw_method=0.3)
kde_maff  = gaussian_kde(y_maff,  bw_method=0.3)
ax.fill_between(x_range, kde_field(x_range), alpha=0.3, color=COLOR_FIELD)
ax.plot(x_range, kde_field(x_range), color=COLOR_FIELD, linewidth=2,
        label=f'現場 (n={len(y_field)}, 平均{y_field.mean():.0f}kg)')
ax.fill_between(x_range, kde_maff(x_range),  alpha=0.3, color=COLOR_MAFF)
ax.plot(x_range, kde_maff(x_range),  color=COLOR_MAFF,  linewidth=2,
        label=f'MAFF (n={len(y_maff)}, 平均{y_maff.mean():.0f}kg)')
ax.axvline(y_field.mean(), color=COLOR_FIELD, linestyle='--', linewidth=1.2, alpha=0.8)
ax.axvline(y_maff.mean(),  color=COLOR_MAFF,  linestyle='--', linewidth=1.2, alpha=0.8)
ax.set_xlabel('収量 (kg/10a)', fontsize=11)
ax.set_ylabel('確率密度', fontsize=11)
ax.set_title('PDF（確率密度関数）', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 600)
ax.grid(alpha=0.3)

# ── 中央: CDF（累積分布関数）─────────────────────────────────────────────────
ax = axes[1]
for y, color, label in [
    (y_field, COLOR_FIELD, f'現場 (n={len(y_field)})'),
    (y_maff,  COLOR_MAFF,  f'MAFF 78市町村 (n={len(y_maff)})'),
]:
    sorted_y = np.sort(y)
    cdf = np.arange(1, len(sorted_y)+1) / len(sorted_y)
    ax.plot(sorted_y, cdf, color=color, linewidth=2, label=label)

# 中央値のマーカー
for y, color in [(y_field, COLOR_FIELD), (y_maff, COLOR_MAFF)]:
    med = np.median(y)
    ax.axvline(med, color=color, linestyle=':', alpha=0.7, linewidth=1.0)

ax.set_xlabel('収量 (kg/10a)', fontsize=11)
ax.set_ylabel('累積確率', fontsize=11)
ax.set_title('CDF（累積分布関数）', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 600)
ax.grid(alpha=0.3)

# ── 右: 箱ひげ図（年別の分布） ────────────────────────────────────────────────
ax = axes[2]
years = [2015, 2016, 2017, 2018]

# 現場: 年別
field_by_year = [quest[quest['year'] == y]['yield'].dropna().values for y in years]
bp1 = ax.boxplot(field_by_year, positions=[y - 0.3 for y in years],
                  widths=0.25, patch_artist=True,
                  boxprops=dict(facecolor=COLOR_FIELD, alpha=0.6),
                  medianprops=dict(color='white', linewidth=2),
                  whiskerprops=dict(color=COLOR_FIELD),
                  capprops=dict(color=COLOR_FIELD),
                  flierprops=dict(marker='o', color=COLOR_FIELD, alpha=0.3, markersize=3))

# MAFF（2014は2015のみ前年として参照のため2015-2018で年別表示）
maff_by_year = [maff_matched[maff_matched['year'] == y]['maff_yield'].values for y in years]
bp2 = ax.boxplot(maff_by_year, positions=[y + 0.3 for y in years],
                  widths=0.25, patch_artist=True,
                  boxprops=dict(facecolor=COLOR_MAFF, alpha=0.6),
                  medianprops=dict(color='white', linewidth=2),
                  whiskerprops=dict(color=COLOR_MAFF),
                  capprops=dict(color=COLOR_MAFF),
                  flierprops=dict(marker='o', color=COLOR_MAFF, alpha=0.3, markersize=3))

ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel('年度', fontsize=11)
ax.set_ylabel('収量 (kg/10a)', fontsize=11)
ax.set_title('年別箱ひげ図', fontsize=12)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_FIELD, alpha=0.6, label='現場（Questionaire）'),
    Patch(facecolor=COLOR_MAFF,  alpha=0.6, label='MAFF（78市町村）')
]
ax.legend(handles=legend_elements, fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'yield_distribution_cdf.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\n保存完了: {save_path}')
