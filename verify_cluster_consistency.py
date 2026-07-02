"""
verify_cluster_consistency.py
===========================================================
【目的】
  クラスタ内で「高収→高収」「低収→低収」と年をまたいで
  正しくつながっているかを確認する。

  特に「多収/低収が混在している地域」で、
  ・高収クラスタ → 年をまたいでも偏差がプラス圏で推移しているか
  ・低収クラスタ → 年をまたいでも偏差がマイナス圏で推移しているか
  を視覚化する。

【出力図】
  1. cluster_consistency_all.png
     全クラスタを「収量偏差の推移」で表示（高収/低収クラスタを色分け）

  2. cluster_consistency_mixed_areas.png
     「混在地域（同一エリア内に複数クラスタが存在する地域）」のみ
     ズームイン。エリアごとに複数クラスタを並べて表示し、
     「別クラスタに分離されているが年をまたいでも水準が維持されているか」を確認。

  3. cluster_linking_check.csv
     各クラスタの各年の偏差値を一覧化した CSV
     （全偏差がプラスなら「一貫多収」、全マイナスなら「一貫低収」）
"""

import sqlite3, os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR  = 'outputs/data_analysis'
FIELD_DB = 'data/processed/FieldData_fieldid.db'
YEARS    = [2015, 2016, 2017, 2018]
DIST_THR = 300
DEV_THR  = 70

# ── データ読み込み ─────────────────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
df = pd.read_sql('''SELECT field_id, year, yield, lat, lon
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018
    ORDER BY year, field_id''', conn)
conn.close()
df['field_id'] = df['field_id'].astype(int)
df['year']     = df['year'].astype(int)
df['yield']    = pd.to_numeric(df['yield'], errors='coerce')
df['lat']      = pd.to_numeric(df['lat'],   errors='coerce')
df['lon']      = pd.to_numeric(df['lon'],   errors='coerce')
df = df.dropna(subset=['lat', 'lon', 'yield']).reset_index(drop=True)

year_means = df.groupby('year')['yield'].mean()
df['yield_dev'] = df.apply(lambda r: r['yield'] - year_means[r['year']], axis=1)

lats    = df['lat'].values
lons    = df['lon'].values
years_a = df['year'].values
devs    = df['yield_dev'].values
N       = len(df)

# ── Haversine & クラスタリング ─────────────────────────────────────────────────
def haversine_matrix(lats, lons):
    R = 6371000.0
    lat_r = np.radians(lats); lon_r = np.radians(lons)
    dlat = lat_r[:,None]-lat_r[None,:]; dlon = lon_r[:,None]-lon_r[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:,None])*np.cos(lat_r[None,:])*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))

print('距離行列計算中...')
D = haversine_matrix(lats, lons)

parent = list(range(N))
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(x,y): parent[find(x)]=find(y)

mask = (years_a[:,None]!=years_a[None,:]) & (D<=DIST_THR) & \
       (np.abs(devs[:,None]-devs[None,:])<=DEV_THR)
for r,c in zip(*np.where(np.tril(mask,k=-1))): union(r,c)

cluster_id = np.array([find(i) for i in range(N)])

# 年またぎクラスタのみ抽出
clusters = defaultdict(list)
for i in range(N): clusters[cluster_id[i]].append(i)

multi_clusters = {}  # root -> members
for root, members in clusters.items():
    if len(set(years_a[m] for m in members)) > 1:
        multi_clusters[root] = members

print(f'年またぎクラスタ数: {len(multi_clusters)}')

# ── 各クラスタの統計を計算 ─────────────────────────────────────────────────────
cl_stats = []
for cl_idx, (root, members) in enumerate(multi_clusters.items()):
    sub = df.iloc[members]
    yr_devs = {}
    yr_yields = {}
    yr_fids = {}
    for yr in YEARS:
        s = sub[sub['year']==yr]
        if len(s)>0:
            yr_devs[yr]   = round(s['yield_dev'].mean(), 1)
            yr_yields[yr] = round(s['yield'].mean(), 1)
            yr_fids[yr]   = list(s['field_id'])
    
    valid_devs = [v for v in yr_devs.values()]
    all_positive = all(v > 0  for v in valid_devs)
    all_negative = all(v < 0  for v in valid_devs)
    n_pos = sum(1 for v in valid_devs if v > 0)
    n_neg = sum(1 for v in valid_devs if v < 0)
    
    # クラスタ判定
    if all_positive:   consistency = '一貫多収'
    elif all_negative: consistency = '一貫低収'
    else:              consistency = '混在(要確認)'
    
    cl_stats.append({
        'cl_idx'      : cl_idx,
        'root'        : root,
        'members'     : members,
        'n_years'     : len(yr_devs),
        'n_fields'    : len(members),
        'lat_mean'    : sub['lat'].mean(),
        'lon_mean'    : sub['lon'].mean(),
        'yr_devs'     : yr_devs,
        'yr_yields'   : yr_yields,
        'yr_fids'     : yr_fids,
        'dev_mean'    : np.mean(valid_devs),
        'consistency' : consistency,
        'n_pos_years' : n_pos,
        'n_neg_years' : n_neg,
    })

# ── CSV出力（クラスタ一貫性チェック） ──────────────────────────────────────────
csv_rows = []
for cs in cl_stats:
    row = {
        'cluster_idx'  : cs['cl_idx'],
        'n_years'      : cs['n_years'],
        'n_fields'     : cs['n_fields'],
        'consistency'  : cs['consistency'],
        'dev_mean'     : round(cs['dev_mean'],1),
        'lat_mean'     : round(cs['lat_mean'],5),
        'lon_mean'     : round(cs['lon_mean'],5),
    }
    for yr in YEARS:
        row[f'dev_{yr}']   = cs['yr_devs'].get(yr)
        row[f'yield_{yr}'] = cs['yr_yields'].get(yr)
    csv_rows.append(row)

csv_df = pd.DataFrame(csv_rows)
csv_path = f'{OUT_DIR}/cluster_linking_check.csv'
csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n一貫性チェック CSV: {csv_path}')
print(f'  一貫多収クラスタ: {sum(1 for cs in cl_stats if cs["consistency"]=="一貫多収")}')
print(f'  一貫低収クラスタ: {sum(1 for cs in cl_stats if cs["consistency"]=="一貫低収")}')
print(f'  混在(要確認)   : {sum(1 for cs in cl_stats if cs["consistency"]=="混在(要確認)")}')

# ── 図1: 全クラスタの偏差推移（高収クラスタ=赤系、低収クラスタ=青系） ──────────
print('\n全クラスタ偏差推移図を生成中...')

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#f9f9f9')

# 左: 収量偏差の推移
ax_l = axes[0]
ax_l.axhline(0, color='black', lw=2, alpha=0.6, zorder=3, label='年平均 (偏差=0)')
ax_l.axhspan(0, 400, alpha=0.04, color='red')
ax_l.axhspan(-400, 0, alpha=0.04, color='blue')

for cs in cl_stats:
    yrs = sorted(cs['yr_devs'].keys())
    devvals = [cs['yr_devs'][yr] for yr in yrs]
    
    if cs['consistency'] == '一貫多収':
        color, lw, alpha, zorder = '#e74c3c', 2.0, 0.75, 4
    elif cs['consistency'] == '一貫低収':
        color, lw, alpha, zorder = '#3498db', 2.0, 0.75, 4
    else:
        color, lw, alpha, zorder = '#f39c12', 2.5, 0.95, 5  # 混在は目立たせる

    ax_l.plot(yrs, devvals, '-o', color=color, lw=lw, alpha=alpha,
              markersize=6, zorder=zorder)
    # 各点に偏差値ラベル
    for yr, dv in zip(yrs, devvals):
        sign = '+' if dv >= 0 else ''
        ax_l.text(yr+0.05, dv, f'{sign}{dv:.0f}', fontsize=6,
                  color=color, alpha=0.8, va='center')

ax_l.set_xticks(YEARS)
ax_l.set_xlabel('年', fontsize=11)
ax_l.set_ylabel('収量偏差 (kg/10a) = 収量 - その年の全圃場平均', fontsize=10)
ax_l.set_title('全クラスタの年別収量偏差推移\n（0より上=多収・0より下=低収）',
               fontsize=11, fontweight='bold')
ax_l.set_facecolor('#fafafa')
ax_l.grid(True, alpha=0.25)

legend_elements = [
    Line2D([0],[0], color='#e74c3c', lw=2.5, label='一貫多収クラスタ（全年で偏差>0）'),
    Line2D([0],[0], color='#3498db', lw=2.5, label='一貫低収クラスタ（全年で偏差<0）'),
    Line2D([0],[0], color='#f39c12', lw=2.5, label='混在クラスタ（要確認: 年によって偏差が正負逆転）'),
    Line2D([0],[0], color='black',   lw=2.0, label='年平均 (偏差=0)'),
]
ax_l.legend(handles=legend_elements, fontsize=8, loc='upper right')

# 右: 一貫性の分類ごとの件数棒グラフ
ax_r = axes[1]
labels   = ['一貫多収\n（全年偏差>0）', '一貫低収\n（全年偏差<0）', '混在\n（偏差が正負逆転）']
counts   = [
    sum(1 for cs in cl_stats if cs['consistency']=='一貫多収'),
    sum(1 for cs in cl_stats if cs['consistency']=='一貫低収'),
    sum(1 for cs in cl_stats if cs['consistency']=='混在(要確認)'),
]
colors_b = ['#e74c3c', '#3498db', '#f39c12']
bars = ax_r.bar(labels, counts, color=colors_b, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, cnt in zip(bars, counts):
    ax_r.text(bar.get_x()+bar.get_width()/2, cnt+0.2, str(cnt),
              ha='center', va='bottom', fontsize=14, fontweight='bold')

ax_r.set_ylabel('クラスタ数', fontsize=11)
ax_r.set_title('クラスタの収量水準一貫性\n（年をまたいで同じ水準でつながっているか）',
               fontsize=11, fontweight='bold')
ax_r.set_facecolor('#fafafa')
ax_r.grid(True, alpha=0.3, axis='y')
ax_r.set_ylim(0, max(counts)+3)

fig.suptitle(f'年またぎクラスタの収量水準一貫性チェック\n'
             f'（距離≤{DIST_THR}m / 偏差差≤{DEV_THR}kg）', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/cluster_consistency_all.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  保存: {OUT_DIR}/cluster_consistency_all.png')

# ── 図2: 混在地域ズームイン──────────────────────────────────────────────────────
# 同じ地域（1度グリッド内）に複数クラスタがある地域を特定
print('\n混在地域の詳細確認図を生成中...')

cell_size = 0.8
cell_to_clusters = defaultdict(list)
for cs in cl_stats:
    cell = (round(cs['lat_mean']/cell_size)*cell_size,
            round(cs['lon_mean']/cell_size)*cell_size)
    cell_to_clusters[cell].append(cs)

# 複数クラスタが存在するセル（混在地域）を抽出し、クラスタ数の多い順に並べる
mixed_cells = [(cell, cls) for cell, cls in cell_to_clusters.items() if len(cls) >= 2]
mixed_cells.sort(key=lambda x: len(x[1]), reverse=True)
mixed_cells = mixed_cells[:8]   # 上位8エリア

print(f'  複数クラスタが存在する地域: {len(mixed_cells)} エリア（上位8を表示）')

ncols = 4
nrows = -(-len(mixed_cells) // ncols)
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 6.5),
                             facecolor='#f5f5f5')
axes2 = [ax for row in (axes2 if nrows>1 else [axes2]) for ax in (row if ncols>1 else [row])]

for ax_i, (cell, cls_list) in enumerate(mixed_cells):
    ax = axes2[ax_i]

    # このエリアの全偏差を集めてY軸範囲を決定
    all_devs_in_area = [
        dv
        for cs in cls_list
        for dv in cs['yr_devs'].values()
    ]
    y_margin = 60
    y_lo = min(all_devs_in_area) - y_margin
    y_hi = max(all_devs_in_area) + y_margin

    ax.axhline(0, color='black', lw=2, alpha=0.5, zorder=3)
    ax.axhspan(0,   y_hi, alpha=0.05, color='red')
    ax.axhspan(y_lo, 0,   alpha=0.05, color='blue')

    cmap20 = plt.cm.tab10
    consistent_ok  = 0
    consistent_ng  = 0

    for ci, cs in enumerate(cls_list):
        yrs     = sorted(cs['yr_devs'].keys())
        devvals = [cs['yr_devs'][yr] for yr in yrs]

        if cs['consistency'] == '一貫多収':
            color = '#e74c3c'
            marker = 'o'
            lw = 2.2
            consistent_ok += 1
        elif cs['consistency'] == '一貫低収':
            color = '#2980b9'
            marker = 'o'
            lw = 2.2
            consistent_ok += 1
        else:
            color = '#f39c12'
            marker = 'X'
            lw = 2.8
            consistent_ng += 1

        ax.plot(yrs, devvals, f'-{marker}', color=color, lw=lw, alpha=0.85,
                markersize=9, zorder=4+ci,
                label=f'cl{cs["cl_idx"]+1} ({cs["consistency"]}) n={cs["n_fields"]}')

        # 各点の偏差とfield_idを表示
        for yr, dv in zip(yrs, devvals):
            fids = cs['yr_fids'].get(yr, [])
            sign = '+' if dv >= 0 else ''
            fid_str = ','.join(str(f) for f in fids[:2])
            offset_pts = 22 if dv >= 0 else -22
            ax.annotate(f'{sign}{dv:.0f}\n(fid:{fid_str})',
                        xy=(yr, dv),
                        xytext=(0, offset_pts),
                        textcoords='offset points',
                        ha='center', va='bottom' if dv >= 0 else 'top',
                        fontsize=8, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.8))

    n_cls = len(cls_list)
    ok_str = f'OK:{consistent_ok}' if consistent_ok > 0 else ''
    ng_str = f' NG:{consistent_ng}' if consistent_ng > 0 else ''
    ax.set_title(f'エリア {ax_i+1} (lat~{cell[0]:.1f}, lon~{cell[1]:.1f})\n'
                 f'{n_cls}クラスタ {ok_str}{ng_str}',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=10)
    ax.set_ylabel('収量偏差 (kg/10a)', fontsize=10)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)

for i in range(len(mixed_cells), len(axes2)):
    axes2[i].set_visible(False)

legend_elements2 = [
    Line2D([0],[0], color='#e74c3c', lw=2.5, marker='o', label='一貫多収クラスタ（全年偏差>0）✓'),
    Line2D([0],[0], color='#2980b9', lw=2.5, marker='o', label='一貫低収クラスタ（全年偏差<0）✓'),
    Line2D([0],[0], color='#f39c12', lw=2.5, marker='X', markersize=9,
           label='混在クラスタ（年によって偏差が正負逆転 → 問題あり）✗'),
]
fig2.legend(handles=legend_elements2, loc='lower center', ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
fig2.suptitle('複数クラスタが混在する地域での一貫性確認\n'
              '（赤線=多収クラスタ・青線=低収クラスタ → 0の線をまたがないか確認）',
              fontsize=12, fontweight='bold')
fig2.tight_layout(rect=[0, 0.06, 1, 1])
fig2.savefig(f'{OUT_DIR}/cluster_consistency_mixed_areas.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'  保存: {OUT_DIR}/cluster_consistency_mixed_areas.png')

# ── 混在クラスタの詳細を表示 ────────────────────────────────────────────────────
mixed_cls = [cs for cs in cl_stats if cs['consistency']=='混在(要確認)']
if mixed_cls:
    print(f'\n【要確認】混在クラスタの詳細 ({len(mixed_cls)}件):')
    print(f'  （年をまたいで偏差の正負が逆転 = 多収と低収が同一クラスタに混入している可能性）')
    for cs in mixed_cls:
        dev_str = '  '.join(f'{yr}:{cs["yr_devs"].get(yr,"-"):+.0f}' if isinstance(cs["yr_devs"].get(yr), float) else f'{yr}:-'
                            for yr in YEARS if cs["yr_devs"].get(yr) is not None)
        print(f'  cl{cs["cl_idx"]+1}: {dev_str}  (n={cs["n_fields"]}圃場, {cs["n_years"]}年)')
else:
    print('\n混在クラスタなし: 全クラスタで高収/低収の一貫性が確認されました')

print('\n完了')
