"""
verify_clustering.py
===========================================================
【目的】
  距離 ≤ 300m で近接する全ペアについて、
  ・同一クラスタに入ったペア → 緑線（正しく統合）
  ・別クラスタに分かれたペア → 赤線（収量偏差が大きく分離）
  として地図上に可視化し、クラスタリングの妥当性を確認する。

  各圃場の点は「収量偏差の大きさ」で色付け：
    赤系  → 多収（その年の平均より大幅に高い）
    青系  → 低収（その年の平均より大幅に低い）
    白/灰  → 平均付近

  【確認のポイント】
  ・多収/低収が混在する密集地域 → 赤線で正しく分離されているか
  ・単色（多収だけ、または低収だけ）の密集地域 → 緑線で正しく統合されているか
"""

import sqlite3, os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 日本語フォント
_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR  = 'outputs/data_analysis'
FIELD_DB = 'data/processed/FieldData_fieldid.db'
YEARS    = [2015, 2016, 2017, 2018]
DIST_THR = 300   # m
DEV_THR  = 70    # kg/10a

# ── データ読み込み ─────────────────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
df = pd.read_sql('''
    SELECT field_id, year, yield, lat, lon
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018
    ORDER BY year, field_id
''', conn)
conn.close()
df['field_id'] = df['field_id'].astype(int)
df['year']     = df['year'].astype(int)
df['yield']    = pd.to_numeric(df['yield'], errors='coerce')
df['lat']      = pd.to_numeric(df['lat'],   errors='coerce')
df['lon']      = pd.to_numeric(df['lon'],   errors='coerce')
df = df.dropna(subset=['lat', 'lon', 'yield']).reset_index(drop=True)

year_means = df.groupby('year')['yield'].mean()
df['yield_dev'] = df.apply(lambda r: r['yield'] - year_means[r['year']], axis=1)

lats     = df['lat'].values
lons     = df['lon'].values
years_a  = df['year'].values
devs     = df['yield_dev'].values
N        = len(df)

# ── Haversine ─────────────────────────────────────────────────────────────────
def haversine_matrix(lats, lons):
    R = 6371000.0
    lat_r = np.radians(lats);  lon_r = np.radians(lons)
    dlat  = lat_r[:, None] - lat_r[None, :]
    dlon  = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

print('距離行列を計算中...')
D = haversine_matrix(lats, lons)

# ── Union-Find クラスタリング ──────────────────────────────────────────────────
parent = list(range(N))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    parent[find(x)] = find(y)

diff_year = years_a[:, None] != years_a[None, :]
near      = D <= DIST_THR
dev_close = np.abs(devs[:, None] - devs[None, :]) <= DEV_THR
mask_connect = diff_year & near & dev_close

rows_c, cols_c = np.where(np.tril(mask_connect, k=-1))
for r, c in zip(rows_c, cols_c):
    union(r, c)

cluster_id = np.array([find(i) for i in range(N)])

# ── 300m以内の全ペアを抽出 ────────────────────────────────────────────────────
# （同一年ペアも含めて確認、ただし可視化は異なる年のみ）
mask_near_diffyear = near & diff_year   # 異なる年かつ300m以内
rows_n, cols_n = np.where(np.tril(mask_near_diffyear, k=-1))

print(f'300m以内の異年ペア数: {len(rows_n)}')
same_cluster_pairs = [(r, c) for r, c in zip(rows_n, cols_n) if cluster_id[r] == cluster_id[c]]
diff_cluster_pairs = [(r, c) for r, c in zip(rows_n, cols_n) if cluster_id[r] != cluster_id[c]]
print(f'  同クラスタペア（緑線）: {len(same_cluster_pairs)}')
print(f'  別クラスタペア（赤線）: {len(diff_cluster_pairs)}')

# 赤線ペアの詳細を表示（どのくらいの偏差差があるか）
if diff_cluster_pairs:
    diffs = [abs(devs[r] - devs[c]) for r, c in diff_cluster_pairs]
    dists = [D[r, c] for r, c in diff_cluster_pairs]
    print(f'  別クラスタペアの偏差差 - 平均:{np.mean(diffs):.1f}  最大:{np.max(diffs):.1f}  最小:{np.min(diffs):.1f}')
    print(f'  別クラスタペアの距離   - 平均:{np.mean(dists):.1f}m  最大:{np.max(dists):.1f}m')

# ── カラーマップ（収量偏差 → 色） ─────────────────────────────────────────────
# 偏差の絶対値が大きいほど濃い色
# 正の偏差（多収）→ 赤系、負の偏差（低収）→ 青系、ゼロ付近 → 白
dev_max = max(abs(devs).max(), 1.0)
norm    = mcolors.TwoSlopeNorm(vmin=-dev_max, vcenter=0, vmax=dev_max)
cmap_pt = mcm.RdBu_r   # 赤（多収）- 白（平均）- 青（低収）

# ── 図1: 全国マップ ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# ---- 赤線（別クラスタ：偏差大きく分離）----
for r, c in diff_cluster_pairs:
    ax.plot([lons[r], lons[c]], [lats[r], lats[c]],
            color='#ff4444', alpha=0.55, linewidth=1.2, zorder=2)

# ---- 緑線（同クラスタ：偏差近く統合）----
for r, c in same_cluster_pairs:
    ax.plot([lons[r], lons[c]], [lats[r], lats[c]],
            color='#44ff88', alpha=0.45, linewidth=1.0, zorder=2)

# ---- 各圃場の点（収量偏差で色付け）----
sc = ax.scatter(df['lon'], df['lat'],
                c=devs, cmap=cmap_pt, norm=norm,
                s=40, zorder=4,
                edgecolors='white', linewidths=0.4, alpha=0.95)

# カラーバー
cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('収量偏差 (kg/10a)\n赤=多収 / 青=低収', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# 凡例
legend_elements = [
    Line2D([0], [0], color='#44ff88', lw=2,
           label=f'同クラスタ（距離≤{DIST_THR}m かつ 偏差差≤{DEV_THR}kg）\n→ 同じ収量水準の連続圃場'),
    Line2D([0], [0], color='#ff4444', lw=2,
           label=f'別クラスタ（距離≤{DIST_THR}m だが 偏差差>{DEV_THR}kg）\n→ 多収/低収の混在地域を正しく分離'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
          facecolor='#2d2d4e', edgecolor='white', labelcolor='white',
          framealpha=0.85)

ax.set_xlabel('経度', color='white', fontsize=10)
ax.set_ylabel('緯度', color='white', fontsize=10)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#555577')
ax.set_title(
    'クラスタリング妥当性マップ\n'
    '（緑線=同クラスタ統合・赤線=多収/低収分離 / 点の色=収量偏差）',
    color='white', fontsize=12, fontweight='bold'
)
ax.grid(True, alpha=0.15, color='#8888aa')

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/clustering_validity_map.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close(fig)
print(f'\n全国マップ保存: {OUT_DIR}/clustering_validity_map.png')

# ── 図2: 密集地域ズームイン（赤線が存在するエリアを自動検出） ─────────────────
if diff_cluster_pairs:
    # 赤線ペアが存在するエリアをクラスタリングして上位N地域を選ぶ
    red_lats = [(lats[r]+lats[c])/2 for r, c in diff_cluster_pairs]
    red_lons = [(lons[r]+lons[c])/2 for r, c in diff_cluster_pairs]

    # 赤線の中点を使って密集エリアを見つける（簡易: 格子で集計）
    from collections import Counter
    cell_size = 0.5   # 緯度経度0.5度グリッド
    cells = Counter()
    for la, lo in zip(red_lats, red_lons):
        cells[(round(la/cell_size)*cell_size, round(lo/cell_size)*cell_size)] += 1

    # 上位6エリアを取得
    top_cells = cells.most_common(6)
    print(f'\n赤線（多収/低収分離）が多い地域 TOP{len(top_cells)}:')

    ncols = 3
    nrows = -(-len(top_cells) // ncols)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 5.0),
                                facecolor='#f5f5f5')
    if nrows == 1:
        axes2 = list(axes2)
    else:
        axes2 = [ax for row in axes2 for ax in row]

    ZOOM = 0.35   # ズームの半幅（緯度経度度）

    for ax_i, ((cell_la, cell_lo), cnt) in enumerate(top_cells):
        ax = axes2[ax_i]

        lat_min = cell_la - ZOOM;  lat_max = cell_la + ZOOM
        lon_min = cell_lo - ZOOM;  lon_max = cell_lo + ZOOM

        # このエリア内の点を抽出
        in_area = (lats >= lat_min) & (lats <= lat_max) & \
                  (lons >= lon_min) & (lons <= lon_max)
        idx_in  = np.where(in_area)[0]

        if len(idx_in) < 2:
            ax.set_visible(False)
            continue

        # 背景（エリア外の全圃場を薄くプロット）
        ax.scatter(df['lon'], df['lat'], s=3, c='#cccccc', alpha=0.15, zorder=1)

        # 赤線（エリア内）
        n_red_in = 0
        for r, c in diff_cluster_pairs:
            if in_area[r] or in_area[c]:
                ax.plot([lons[r], lons[c]], [lats[r], lats[c]],
                        color='#e74c3c', alpha=0.75, linewidth=1.8, zorder=2)
                n_red_in += 1

        # 緑線（エリア内）
        n_grn_in = 0
        for r, c in same_cluster_pairs:
            if in_area[r] and in_area[c]:
                ax.plot([lons[r], lons[c]], [lats[r], lats[c]],
                        color='#27ae60', alpha=0.65, linewidth=1.5, zorder=2)
                n_grn_in += 1

        # 点（エリア内のみ色付け、エリア外はグレー）
        for i in idx_in:
            color_val = norm(devs[i])
            color     = cmap_pt(color_val)
            ax.scatter(lons[i], lats[i], s=100, color=color,
                       edgecolors='white', linewidths=0.7, zorder=4)
            # 収量偏差ラベル
            sign = '+' if devs[i] >= 0 else ''
            ax.text(lons[i], lats[i] + 0.005,
                    f'{sign}{devs[i]:.0f}',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#c0392b' if devs[i] >= 0 else '#2980b9',
                    fontweight='bold')

        print(f'  エリア (lat~={cell_la:.2f}, lon~={cell_lo:.2f}): '
              f'赤線{n_red_in}本 / 緑線{n_grn_in}本 / 圃場{len(idx_in)}件')

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_title(f'密集地域 {ax_i+1}\n'
                     f'(lat~={cell_la:.1f}, lon~={cell_lo:.1f}) '
                     f'赤:{n_red_in}本 緑:{n_grn_in}本',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('経度', fontsize=8)
        ax.set_ylabel('緯度', fontsize=8)
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # 余白非表示
    for i in range(len(top_cells), len(axes2)):
        axes2[i].set_visible(False)

    legend_elements2 = [
        Line2D([0], [0], color='#27ae60', lw=2, label='同クラスタ（同収量水準・統合）'),
        Line2D([0], [0], color='#e74c3c', lw=2, label='別クラスタ（多収/低収混在・分離）'),
        Patch(facecolor='#e74c3c', alpha=0.4, label='赤点：多収（偏差 > 0）'),
        Patch(facecolor='#3498db', alpha=0.4, label='青点：低収（偏差 < 0）'),
    ]
    fig2.legend(handles=legend_elements2, loc='lower center', ncol=4, fontsize=9,
                bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig2.suptitle(
        '多収/低収 混在地域のズームイン\n'
        '（赤線=収量偏差が大きくて正しく分離、点の数字=収量偏差 kg/10a）',
        fontsize=12, fontweight='bold'
    )
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    fig2.savefig(f'{OUT_DIR}/clustering_validity_zoom.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'\nズームイン図保存: {OUT_DIR}/clustering_validity_zoom.png')

# ── 図3: 偏差差のヒストグラム（赤線 vs 緑線の偏差差分布） ─────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 5), facecolor='#f9f9f9')

same_devdiffs = [abs(devs[r]-devs[c]) for r, c in same_cluster_pairs]
diff_devdiffs = [abs(devs[r]-devs[c]) for r, c in diff_cluster_pairs]

bins = np.linspace(0, max(max(same_devdiffs, default=0),
                           max(diff_devdiffs, default=0)) + 10, 40)

ax3.hist(same_devdiffs, bins=bins, color='#27ae60', alpha=0.7,
         label=f'同クラスタペア (n={len(same_devdiffs)})\n偏差差 ≤ {DEV_THR}kg で統合', edgecolor='white')
ax3.hist(diff_devdiffs, bins=bins, color='#e74c3c', alpha=0.7,
         label=f'別クラスタペア (n={len(diff_devdiffs)})\n偏差差 > {DEV_THR}kg で分離', edgecolor='white')
ax3.axvline(DEV_THR, color='black', linestyle='--', lw=2,
            label=f'閾値 DEV_THR = {DEV_THR} kg')
ax3.set_xlabel('2圃場間の収量偏差差 (kg/10a)', fontsize=11)
ax3.set_ylabel('ペア数', fontsize=11)
ax3.set_title('300m以内ペアの収量偏差差分布\n（緑=同クラスタに統合 / 赤=別クラスタに分離）',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_facecolor('#fdfdfd')
ax3.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(f'{OUT_DIR}/clustering_devdiff_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f'偏差差分布図保存: {OUT_DIR}/clustering_devdiff_distribution.png')

print('\n完了')
