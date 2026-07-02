"""
analyze_data_distribution.py
各年のデータ分布図の出力と、lat/lon が一致する field_id（連続記録圃場）の確認
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 日本語フォント設定（Windows 標準フォントを優先して探す） ───────────────
import matplotlib.font_manager as fm

_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'IPAexGothic', 'Noto Sans CJK JP']
_found = None
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _found = _fn
        break
if _found:
    plt.rcParams['font.family'] = _found
else:
    # フォールバック: システムに存在するフォントパスを直接指定
    _candidates = [
        r'C:\Windows\Fonts\YuGothM.ttc',
        r'C:\Windows\Fonts\meiryo.ttc',
        r'C:\Windows\Fonts\msgothic.ttc',
    ]
    for _p in _candidates:
        if os.path.exists(_p):
            fm.fontManager.addfont(_p)
            _prop = fm.FontProperties(fname=_p)
            plt.rcParams['font.family'] = _prop.get_name()
            break

plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from itertools import combinations

FIELD_DB   = 'data/processed/FieldData_fieldid.db'
OUT_DIR    = 'outputs/data_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ── データ読み込み ──────────────────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
df = pd.read_sql('''
    SELECT field_id, year, yield, lat, lon, breed
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018
    ORDER BY year, field_id
''', conn)
conn.close()

df['field_id'] = df['field_id'].astype(int)
df['year']     = df['year'].astype(int)
df['yield']    = pd.to_numeric(df['yield'],  errors='coerce')
df['lat']      = pd.to_numeric(df['lat'],    errors='coerce')
df['lon']      = pd.to_numeric(df['lon'],    errors='coerce')
df['breed']    = df['breed'].astype(str).str.strip().replace({'None': 'unknown', 'nan': 'unknown'})
# カンマ区切り → 最初の品種
df['breed'] = df['breed'].str.split(',').str[0].str.strip()

YEARS  = [2015, 2016, 2017, 2018]
COLORS = {2015: '#4C72B0', 2016: '#55A868', 2017: '#C44E52', 2018: '#8172B2'}

# ════════════════════════════════════════════════════════════════════════════
# Figure 1: 年別 収量分布（ヒストグラム + KDE + 箱ひげ）
# ════════════════════════════════════════════════════════════════════════════
from scipy.stats import gaussian_kde

fig = plt.figure(figsize=(16, 12), facecolor='#f9f9f9')
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# 上段: 年ごとのヒストグラム
for i, yr in enumerate(YEARS):
    ax = fig.add_subplot(gs[0, i])
    sub = df[df['year'] == yr]['yield'].dropna()
    ax.hist(sub, bins=25, color=COLORS[yr], edgecolor='white', linewidth=0.6, alpha=0.85)
    # KDE
    kde = gaussian_kde(sub, bw_method=0.3)
    xs  = np.linspace(sub.min() - 20, sub.max() + 20, 200)
    ax2 = ax.twinx()
    ax2.plot(xs, kde(xs), color='#222222', lw=2)
    ax2.set_yticks([])
    ax.set_title(f'{yr}年  N={len(sub)}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Yield (kg/10a)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.axvline(sub.mean(), color='red', lw=1.5, linestyle='--', label=f'mean={sub.mean():.1f}')
    ax.axvline(sub.median(), color='orange', lw=1.5, linestyle=':', label=f'med={sub.median():.1f}')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, alpha=0.25)

# 中段: 全年オーバーレイ ヒストグラム（透過）
ax_all = fig.add_subplot(gs[1, :2])
for yr in YEARS:
    sub = df[df['year'] == yr]['yield'].dropna()
    ax_all.hist(sub, bins=30, alpha=0.45, label=str(yr),
                color=COLORS[yr], edgecolor='white', linewidth=0.5)
ax_all.set_title('全年オーバーレイ（収量ヒストグラム）', fontsize=11, fontweight='bold')
ax_all.set_xlabel('Yield (kg/10a)', fontsize=9)
ax_all.set_ylabel('Count', fontsize=9)
ax_all.legend(title='Year', fontsize=9)
ax_all.set_facecolor('#fdfdfd')
ax_all.grid(True, alpha=0.25)

# 中段: 箱ひげ図
ax_box = fig.add_subplot(gs[1, 2:])
data_by_year = [df[df['year'] == yr]['yield'].dropna().values for yr in YEARS]
bp = ax_box.boxplot(data_by_year, labels=[str(y) for y in YEARS],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', lw=2))
for patch, yr in zip(bp['boxes'], YEARS):
    patch.set_facecolor(COLORS[yr])
    patch.set_alpha(0.7)
ax_box.set_title('年別 収量箱ひげ図', fontsize=11, fontweight='bold')
ax_box.set_xlabel('Year', fontsize=9)
ax_box.set_ylabel('Yield (kg/10a)', fontsize=9)
ax_box.set_facecolor('#fdfdfd')
ax_box.grid(True, alpha=0.25, axis='y')

# 下段: 年別記述統計テーブル
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.axis('off')
stats_rows = []
for yr in YEARS:
    sub = df[df['year'] == yr]['yield'].dropna()
    stats_rows.append([
        str(yr), str(len(sub)),
        f'{sub.mean():.1f}', f'{sub.std():.1f}',
        f'{sub.min():.1f}', f'{sub.quantile(0.25):.1f}',
        f'{sub.median():.1f}', f'{sub.quantile(0.75):.1f}',
        f'{sub.max():.1f}'
    ])
col_labels = ['Year', 'N', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max']
tbl = ax_tbl.table(
    cellText=stats_rows, colLabels=col_labels,
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r > 0:
        yr_val = YEARS[r - 1]
        cell.set_facecolor(COLORS[yr_val] + '30')
ax_tbl.set_title('年別 収量記述統計', fontsize=11, fontweight='bold', pad=12)

fig.suptitle('年別 収量データ分布分析 (2015–2018)', fontsize=14, fontweight='bold', y=1.01)
path1 = os.path.join(OUT_DIR, 'yield_distribution_by_year.png')
fig.savefig(path1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Figure 1 saved: {path1}')


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: 年別 地理分布（lat/lon 散布図）
# ════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5), facecolor='#f9f9f9')
for ax, yr in zip(axes2, YEARS):
    sub = df[df['year'] == yr].dropna(subset=['lat', 'lon'])
    sc  = ax.scatter(sub['lon'], sub['lat'], c=sub['yield'],
                     cmap='RdYlGn', s=35, alpha=0.75,
                     vmin=df['yield'].quantile(0.05),
                     vmax=df['yield'].quantile(0.95),
                     edgecolors='white', linewidths=0.3)
    ax.set_title(f'{yr}年  N={len(sub)}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)
    plt.colorbar(sc, ax=ax, label='Yield', shrink=0.8)
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, alpha=0.3)

fig2.suptitle('年別 圃場位置と収量（地理分布）', fontsize=13, fontweight='bold')
fig2.tight_layout()
path2 = os.path.join(OUT_DIR, 'geo_distribution_by_year.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Figure 2 saved: {path2}')


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: 年別 品種構成（棒グラフ）
# ════════════════════════════════════════════════════════════════════════════
breed_year = pd.crosstab(df['year'], df['breed'])
fig3, ax3 = plt.subplots(figsize=(14, 5), facecolor='#f9f9f9')
breed_year.plot(kind='bar', ax=ax3, stacked=True, colormap='tab20', edgecolor='white', linewidth=0.4)
ax3.set_title('年別 品種構成', fontsize=12, fontweight='bold')
ax3.set_xlabel('Year', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.legend(title='Breed', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.set_facecolor('#fdfdfd')
ax3.grid(True, alpha=0.25, axis='y')
fig3.tight_layout()
path3 = os.path.join(OUT_DIR, 'breed_composition_by_year.png')
fig3.savefig(path3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f'Figure 3 saved: {path3}')


# ════════════════════════════════════════════════════════════════════════════
# 連続記録圃場の確認: lat/lon が一致する field_id ペア
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('連続記録圃場の確認: 異なる field_id で lat/lon が一致するもの')
print('='*60)

# 丸め精度を変えながら確認（緯度経度が完全一致 or 近接）
ROUND_DIGITS = 4   # 約11m の精度

df_geo = df.dropna(subset=['lat', 'lon']).copy()
df_geo['lat_r'] = df_geo['lat'].round(ROUND_DIGITS)
df_geo['lon_r'] = df_geo['lon'].round(ROUND_DIGITS)

# (lat_r, lon_r) で groupby し、複数 field_id がある組を抽出
geo_grp = df_geo.groupby(['lat_r', 'lon_r'])['field_id'].apply(set).reset_index()
geo_grp.columns = ['lat_r', 'lon_r', 'field_ids']
geo_grp['n_field_ids'] = geo_grp['field_ids'].apply(len)
geo_grp['n_unique_years'] = geo_grp.apply(
    lambda row: df_geo[
        (df_geo['lat_r'] == row['lat_r']) & (df_geo['lon_r'] == row['lon_r'])
    ]['year'].nunique(), axis=1
)

multi_fid = geo_grp[geo_grp['n_field_ids'] > 1].copy()
print(f'\nround({ROUND_DIGITS}桁) で lat/lon が一致し、複数 field_id が存在する地点数: {len(multi_fid)}')

if len(multi_fid) > 0:
    # 詳細表示
    records = []
    for _, row in multi_fid.iterrows():
        fids = sorted(row['field_ids'])
        sub  = df_geo[
            (df_geo['lat_r'] == row['lat_r']) & (df_geo['lon_r'] == row['lon_r'])
        ][['field_id', 'year', 'yield', 'breed']].sort_values('year')
        for _, r in sub.iterrows():
            records.append({
                'lat': row['lat_r'], 'lon': row['lon_r'],
                'n_fids': row['n_field_ids'],
                'field_id': int(r['field_id']), 'year': int(r['year']),
                'yield': round(float(r['yield']), 1), 'breed': r['breed']
            })
    rec_df = pd.DataFrame(records)
    print(rec_df.to_string(index=False))
    print(f'\n連続記録が存在する地点数: {len(multi_fid)}')
    print(f'そのうち年をまたぐ記録がある地点: {(multi_fid["n_unique_years"] > 1).sum()}')
else:
    print('  → 同一 lat/lon に複数 field_id は存在しない（連続記録なし）')

# ── 丸め精度を緩めて (3桁 ≈ 111m) で確認 ──
ROUND_DIGITS2 = 3
df_geo['lat_r3'] = df_geo['lat'].round(ROUND_DIGITS2)
df_geo['lon_r3'] = df_geo['lon'].round(ROUND_DIGITS2)
geo_grp3 = df_geo.groupby(['lat_r3', 'lon_r3'])['field_id'].apply(set).reset_index()
geo_grp3.columns = ['lat_r3', 'lon_r3', 'fid_set']
geo_grp3['n_field_ids'] = geo_grp3['fid_set'].apply(len)
multi3 = geo_grp3[geo_grp3['n_field_ids'] > 1]
print(f'\nround(3桁, ~111m) で複数 field_id が同一地点: {len(multi3)} 地点  ({multi3["n_field_ids"].sum()} field_id)')

# ── Figure 4: 連続記録サマリ図 ──
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), facecolor='#f9f9f9')

# 左: field_id 出現年数の分布
year_count = df.groupby('field_id')['year'].nunique()
axes4[0].hist(year_count.values, bins=[0.5, 1.5, 2.5, 3.5, 4.5],
              color='#4C72B0', edgecolor='white', rwidth=0.8)
axes4[0].set_title('圃場あたりの記録年数分布', fontsize=11, fontweight='bold')
axes4[0].set_xlabel('記録年数 (unique years per field_id)', fontsize=9)
axes4[0].set_ylabel('圃場数', fontsize=9)
axes4[0].set_xticks([1, 2, 3, 4])
axes4[0].set_facecolor('#fdfdfd')
axes4[0].grid(True, alpha=0.25, axis='y')
for cnt in [1, 2, 3, 4]:
    n = (year_count == cnt).sum()
    axes4[0].text(cnt, n + 0.5, str(n), ha='center', va='bottom', fontsize=10)

# 右: 各年のサンプル数
n_by_year = df.groupby('year')['field_id'].count()
bars = axes4[1].bar([str(y) for y in YEARS],
                    [n_by_year[yr] for yr in YEARS],
                    color=[COLORS[y] for y in YEARS],
                    edgecolor='white', linewidth=0.5)
axes4[1].set_title('年別サンプル数', fontsize=11, fontweight='bold')
axes4[1].set_xlabel('Year', fontsize=9)
axes4[1].set_ylabel('サンプル数', fontsize=9)
axes4[1].set_facecolor('#fdfdfd')
axes4[1].grid(True, alpha=0.25, axis='y')
for bar, yr in zip(bars, YEARS):
    axes4[1].text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 1, str(n_by_year[yr]),
                  ha='center', va='bottom', fontsize=11, fontweight='bold')

fig4.suptitle('圃場の連続記録状況', fontsize=12, fontweight='bold')
fig4.tight_layout()
path4 = os.path.join(OUT_DIR, 'field_continuity.png')
fig4.savefig(path4, dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f'\nFigure 4 saved: {path4}')

print('\n=== 圃場別 記録年数サマリ ===')
print(year_count.value_counts().sort_index().rename('圃場数').to_string())
print(f'\n総 field_id 数 : {df["field_id"].nunique()}')
print(f'1年のみ記録   : {(year_count == 1).sum()} 圃場')
print(f'2年以上記録   : {(year_count >= 2).sum()} 圃場')
print(f'全4年記録     : {(year_count == 4).sum()} 圃場')
