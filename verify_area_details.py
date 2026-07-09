"""
verify_area_details.py
===========================================================
【目的】
  複数クラスタが混在する地域について、各エリアごとに

  [左パネル] ミニ地図
    ・各圃場を field_id ラベル付きで表示
    ・300m 以内のペアを線で結ぶ
        緑線 = 同クラスタ（距離 OK + 偏差差 OK）
        赤線 = 別クラスタ（距離 OK だが偏差差 > DEV_THR）
        灰線 = 300m超（クラスタリング対象外）は描かない
    ・スケールバーで 300m の感覚を表示

  [右パネル] 収量偏差推移（field_id 個別）
    ・各 field_id を別々の線でプロット（平均化しない）
    ・同クラスタは同系色、別クラスタは別色
    ・各点に field_id と偏差値ラベル

  → 「なぜ fid310 は fid292 と繋がらないか（距離超過 or 偏差差超過）」が
    左の地図と右のグラフを見比べることで一目でわかる。
"""

import sqlite3, os
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ── 日本語フォント ─────────────────────────────────────────────────────────────
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

COLORS_YR = {2015: '#4C72B0', 2016: '#2ca02c', 2017: '#d62728', 2018: '#9467bd'}

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
fids    = df['field_id'].values
N       = len(df)

# ── Haversine ─────────────────────────────────────────────────────────────────
def haversine_matrix(la, lo):
    R = 6371000.0
    lr = np.radians(la); lonr = np.radians(lo)
    dlat = lr[:,None]-lr[None,:]; dlon = lonr[:,None]-lonr[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

print('距離行列を計算中...')
D = haversine_matrix(lats, lons)

# ── Union-Find クラスタリング ──────────────────────────────────────────────────
parent = list(range(N))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y): parent[find(x)] = find(y)

mask_connect = (years_a[:,None] != years_a[None,:]) & \
               (D <= DIST_THR) & \
               (np.abs(devs[:,None] - devs[None,:]) <= DEV_THR)
for r, c in zip(*np.where(np.tril(mask_connect, k=-1))):
    union(r, c)

cluster_id = np.array([find(i) for i in range(N)])
df['cluster_id'] = cluster_id

# 年またぎクラスタのみ
clusters = defaultdict(list)
for i in range(N):
    clusters[cluster_id[i]].append(i)
multi_clusters = {root: mems for root, mems in clusters.items()
                  if len(set(years_a[m] for m in mems)) > 1}

print(f'年またぎクラスタ数: {len(multi_clusters)}')

# ── 300m以内の異年ペアを全抽出（地図描画用） ─────────────────────────────────
mask_near = (years_a[:,None] != years_a[None,:]) & (D <= DIST_THR)
near_pairs = [(r, c) for r, c in zip(*np.where(np.tril(mask_near, k=-1)))]

# ── 混在地域の検出（0.8度グリッド、複数クラスタが存在するセル） ────────────────
cell_size = 0.8
cell_to_clusters = defaultdict(list)
for root, mems in multi_clusters.items():
    sub = df.iloc[mems]
    lat_c = round(sub['lat'].mean() / cell_size) * cell_size
    lon_c = round(sub['lon'].mean() / cell_size) * cell_size
    cell_to_clusters[(lat_c, lon_c)].append({'root': root, 'members': mems})

mixed_cells = [(cell, cls) for cell, cls in cell_to_clusters.items() if len(cls) >= 2]
mixed_cells.sort(key=lambda x: len(x[1]), reverse=True)
mixed_cells = mixed_cells[:8]

print(f'複数クラスタが混在する地域: {len(mixed_cells)} エリア')

# ── クラスタに固定色を割り当てる ─────────────────────────────────────────────
# 各エリア内でクラスタごとに色を割り当て
CLUSTER_COLORS_BASE = [
    '#e74c3c',  # 赤
    '#2980b9',  # 青
    '#27ae60',  # 緑
    '#8e44ad',  # 紫
    '#e67e22',  # オレンジ
    '#16a085',  # ティール
    '#c0392b',  # 暗赤
    '#2c3e50',  # 暗青
]

# ── 緯度経度 → メートル換算係数 ────────────────────────────────────────────────
def latlon_to_m_per_deg(lat_center):
    """緯度1度あたりのメートル数、経度1度あたりのメートル数"""
    R = 6371000.0
    m_per_lat = R * math.pi / 180
    m_per_lon = R * math.cos(math.radians(lat_center)) * math.pi / 180
    return m_per_lat, m_per_lon

# ── 各エリアの図を生成 ─────────────────────────────────────────────────────────
for area_idx, (cell, cls_list) in enumerate(mixed_cells):
    cell_lat, cell_lon = cell
    m_per_lat, m_per_lon = latlon_to_m_per_deg(cell_lat)

    # このエリアに関わる全インデックスを収集
    all_members_idx = set()
    for cs in cls_list:
        all_members_idx.update(cs['members'])

    # 関連する field_id のみ抽出（エリアの中心から近い圃場を全て含む）
    # 「このエリアのクラスタに属する圃場」+ 「300m以内にある全圃場」
    seed_lats = lats[list(all_members_idx)]
    seed_lons = lons[list(all_members_idx)]
    lat_min_s = seed_lats.min() - 0.005
    lat_max_s = seed_lats.max() + 0.005
    lon_min_s = seed_lons.min() - 0.005
    lon_max_s = seed_lons.max() + 0.005
    # エリア範囲内の全圃場
    in_area_mask = ((lats >= lat_min_s) & (lats <= lat_max_s) &
                    (lons >= lon_min_s) & (lons <= lon_max_s))
    area_indices = np.where(in_area_mask)[0]

    if len(area_indices) < 2:
        continue

    # クラスタ → 色のマッピング
    root_to_color = {}
    for ci, cs in enumerate(cls_list):
        root_to_color[cs['root']] = CLUSTER_COLORS_BASE[ci % len(CLUSTER_COLORS_BASE)]
    # 非クラスタ（単独）は灰色
    GREY = '#aaaaaa'

    def get_color(idx):
        root = find(idx)
        return root_to_color.get(root, GREY)

    # ── 図を作成: 上段=地図、下段=偏差グラフ（縦2分割）─────────────────────────
    # エリア内の近接ペアを抽出
    area_set = set(area_indices.tolist())
    area_pairs = [(r, c) for r, c in near_pairs
                  if r in area_set or c in area_set]

    fig, (ax_map, ax_dev) = plt.subplots(1, 2, figsize=(18, 8),
                                          facecolor='#f7f7f7')
    fig.suptitle(
        f'エリア {area_idx+1}  '
        f'（lat~{cell_lat:.1f}, lon~{cell_lon:.1f}）\n'
        f'左: 地図（緑線=同クラスタ / 赤線=偏差差>{DEV_THR}kg で別クラスタ）  '
        f'右: 各 field_id の収量偏差推移（平均化なし）',
        fontsize=12, fontweight='bold'
    )

    # ────────────────────────────────────────────────────
    # 左パネル: ミニ地図
    # ────────────────────────────────────────────────────
    ax_map.set_facecolor('#eef2f7')

    # 全エリア内圃場の点と field_id ラベル
    for i in area_indices:
        color = get_color(i)
        yr    = int(years_a[i])
        fid   = int(fids[i])
        dv    = devs[i]

        # マーカー形状: 年で変える
        marker_by_yr = {2015: 'o', 2016: 's', 2017: '^', 2018: 'D'}
        mk = marker_by_yr.get(yr, 'o')

        ax_map.scatter(lons[i], lats[i], s=160, color=color, marker=mk,
                       edgecolors='white', linewidths=1.2, zorder=5)
        # field_id と収量偏差ラベル
        sign = '+' if dv >= 0 else ''
        ax_map.text(lons[i], lats[i] + 0.0008,
                    f'fid{fid}\n{yr}:{sign}{dv:.0f}',
                    ha='center', va='bottom', fontsize=7.5,
                    color=color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ペア間の接続線
    for r, c in area_pairs:
        same_cl = (cluster_id[r] == cluster_id[c])
        dev_diff = abs(devs[r] - devs[c])
        dist_m   = D[r, c]

        if same_cl:
            lc, lw, la, ls = '#27ae60', 2.0, 0.80, '-'    # 緑: 同クラスタ
        else:
            lc, lw, la, ls = '#e74c3c', 2.0, 0.80, '--'   # 赤破線: 別クラスタ

        ax_map.plot([lons[r], lons[c]], [lats[r], lats[c]],
                    color=lc, lw=lw, alpha=la, linestyle=ls, zorder=3)

        # 線の中点に「距離 / 偏差差」注釈
        mid_lon = (lons[r] + lons[c]) / 2
        mid_lat = (lats[r] + lats[c]) / 2
        reason = f'{dist_m:.0f}m\n|dev|={dev_diff:.0f}'
        ax_map.text(mid_lon, mid_lat, reason,
                    ha='center', va='center', fontsize=6.5,
                    color=lc,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                    zorder=6)

    # 300m スケールバー（右下）
    xlim = ax_map.get_xlim(); ylim = ax_map.get_ylim()
    # 再描画後のスケールは tight_layout 後に設定するため、まず仮のマージンで計算
    x_range = lon_max_s - lon_min_s
    y_range = lat_max_s - lat_min_s
    # 300m を経度差に変換
    scale_deg = 300 / m_per_lon
    sb_x0 = lon_max_s - scale_deg - x_range * 0.05
    sb_y  = lat_min_s + y_range * 0.04
    ax_map.plot([sb_x0, sb_x0 + scale_deg], [sb_y, sb_y],
                color='black', lw=3, solid_capstyle='butt', zorder=10)
    ax_map.text(sb_x0 + scale_deg / 2, sb_y + y_range * 0.015,
                '300 m', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_map.set_xlim(lon_min_s - x_range*0.05, lon_max_s + x_range*0.05)
    ax_map.set_ylim(lat_min_s - y_range*0.05, lat_max_s + y_range*0.05)
    ax_map.set_xlabel('経度', fontsize=10)
    ax_map.set_ylabel('緯度', fontsize=10)
    ax_map.set_title('地図（各 field_id の位置と距離・偏差差）', fontsize=11, fontweight='bold')
    ax_map.grid(True, alpha=0.3)

    # 年マーカー凡例
    yr_handles = [Line2D([0],[0], marker=mk, color='gray', markersize=8,
                         linestyle='None', label=f'{yr}年')
                  for yr, mk in {2015:'o',2016:'s',2017:'^',2018:'D'}.items()]
    line_handles = [
        Line2D([0],[0], color='#27ae60', lw=2, label=f'同クラスタ（偏差差≤{DEV_THR}kg）'),
        Line2D([0],[0], color='#e74c3c', lw=2, linestyle='--',
               label=f'別クラスタ（偏差差>{DEV_THR}kg）'),
    ]
    ax_map.legend(handles=yr_handles + line_handles, fontsize=8, loc='upper left',
                  framealpha=0.9, ncol=2)

    # クラスタ色の凡例
    for ci, cs in enumerate(cls_list):
        color = CLUSTER_COLORS_BASE[ci % len(CLUSTER_COLORS_BASE)]
        # クラスタの一貫性を判定
        mems = cs['members']
        sub  = df.iloc[mems]
        yr_devs = {yr: sub[sub['year']==yr]['yield_dev'].mean()
                   for yr in YEARS if len(sub[sub['year']==yr]) > 0}
        vals = list(yr_devs.values())
        if all(v > 0 for v in vals):   cons = '一貫多収'
        elif all(v < 0 for v in vals): cons = '一貫低収'
        else:                           cons = '混在'
        ax_map.scatter([], [], color=color, s=100,
                       label=f'クラスタ{ci+1} ({cons})', zorder=10)
    ax_map.legend(handles=yr_handles + line_handles +
                  [ax_map.scatter([], [], color=CLUSTER_COLORS_BASE[ci%len(CLUSTER_COLORS_BASE)],
                                  s=80, label=f'cl{ci+1}')
                   for ci in range(len(cls_list))],
                  fontsize=8, loc='upper left', framealpha=0.9, ncol=1)

    # ────────────────────────────────────────────────────
    # 右パネル: 各 field_id の収量偏差推移（平均化せず個別プロット）
    # ────────────────────────────────────────────────────
    ax_dev.axhline(0, color='black', lw=2, alpha=0.6, zorder=3)

    # このエリアに関連する全 field_id の偏差推移を描く
    # ① クラスタに属する圃場（クラスタ色）
    plotted_fids = set()
    for ci, cs in enumerate(cls_list):
        color = CLUSTER_COLORS_BASE[ci % len(CLUSTER_COLORS_BASE)]
        mems  = cs['members']
        sub   = df.iloc[mems]

        # 同クラスタ内の同年複数 field_id をわずかにジッター
        unique_fids_in_cluster = sub['field_id'].unique()
        for fid_val in sorted(unique_fids_in_cluster):
            fid_rows = sub[sub['field_id'] == fid_val].sort_values('year')
            if len(fid_rows) == 0:
                continue

            yrs_f  = fid_rows['year'].tolist()
            devs_f = fid_rows['yield_dev'].tolist()

            ax_dev.plot(yrs_f, devs_f, '-o', color=color, lw=2.0,
                        markersize=10, alpha=0.85, zorder=4)
            # 各点ラベル
            for yr_f, dv_f in zip(yrs_f, devs_f):
                sign = '+' if dv_f >= 0 else ''
                ax_dev.annotate(
                    f'fid{fid_val}\n{sign}{dv_f:.0f}',
                    xy=(yr_f, dv_f),
                    xytext=(0, 18 if dv_f >= 0 else -18),
                    textcoords='offset points',
                    ha='center', va='bottom' if dv_f >= 0 else 'top',
                    fontsize=8, color=color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.8)
                )
            plotted_fids.add(fid_val)

    # ② エリア内にいるがどのクラスタにも属さない圃場（灰色）
    for i in area_indices:
        fid_val = int(fids[i])
        if fid_val in plotted_fids:
            continue
        yr_f  = int(years_a[i])
        dv_f  = devs[i]
        ax_dev.scatter(yr_f, dv_f, s=80, color=GREY, marker='x',
                       linewidths=2, zorder=4, alpha=0.7)
        sign = '+' if dv_f >= 0 else ''
        ax_dev.annotate(f'fid{fid_val}\n{sign}{dv_f:.0f}',
                        xy=(yr_f, dv_f),
                        xytext=(0, 18 if dv_f >= 0 else -18),
                        textcoords='offset points',
                        ha='center', va='bottom' if dv_f >= 0 else 'top',
                        fontsize=8, color=GREY, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Y軸範囲を実データに合わせる
    all_devs_area = [devs[i] for i in area_indices]
    y_margin = 60
    y_lo = min(all_devs_area) - y_margin
    y_hi = max(all_devs_area) + y_margin
    ax_dev.axhspan(0,    y_hi, alpha=0.05, color='red')
    ax_dev.axhspan(y_lo, 0,    alpha=0.05, color='blue')
    ax_dev.set_ylim(y_lo, y_hi)
    ax_dev.set_xticks(YEARS)
    ax_dev.set_xticklabels([str(y) for y in YEARS], fontsize=11)
    ax_dev.set_xlabel('年', fontsize=11)
    ax_dev.set_ylabel('収量偏差 (kg/10a)\n= 収量 − その年の全圃場平均', fontsize=10)
    ax_dev.set_title('各 field_id の収量偏差推移\n（同色 = 同クラスタ / 平均化なし）',
                     fontsize=11, fontweight='bold')
    ax_dev.set_facecolor('#fafafa')
    ax_dev.grid(True, alpha=0.3)

    # クラスタ凡例
    dev_handles = [
        Line2D([0],[0], color=CLUSTER_COLORS_BASE[ci % len(CLUSTER_COLORS_BASE)],
               lw=2.5, marker='o', markersize=8,
               label=f'クラスタ{ci+1} (n={len(cs["members"])}圃場)')
        for ci, cs in enumerate(cls_list)
    ] + [Line2D([0],[0], color=GREY, lw=0, marker='x', markersize=8,
                markeredgewidth=2, label='クラスタ未所属')]
    ax_dev.legend(handles=dev_handles, fontsize=9, loc='best', framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f'{OUT_DIR}/area_detail_{area_idx+1:02d}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  エリア{area_idx+1} 保存: {fname}')

print('\n完了')
