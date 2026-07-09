"""
verify_best_area_zoom.py
===========================================================
【目的】
  「近接圃場ペアが最も密集している地点」を自動検出し、
  その地点のみを大きく拡大したプレゼン向け図を生成する。

  [左パネル] 拡大地図（約 1km × 1km 範囲）
    - 各圃場を field_id + 年度 + 収量偏差ラベル付きで表示
    - 300m 以内のペアを線で結ぶ
        緑実線  = 同クラスタ（距離 OK + 偏差差 ≤ 70kg）
        赤破線  = 別クラスタ（距離 OK だが偏差差 > 70kg）
    - 300m スケールバー付き
    - 線の中点に「距離 / 偏差差」を表示

  [右パネル] 収量偏差推移（field_id 個別・平均化なし）
    - 各 field_id を個別の線でプロット
    - 同クラスタ = 同系色
    - 各点に field_id・偏差・年ラベル
"""

import sqlite3, os, math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
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

lats_a  = df['lat'].values
lons_a  = df['lon'].values
years_a = df['year'].values
devs_a  = df['yield_dev'].values
fids_a  = df['field_id'].values
N       = len(df)

# ── Haversine ─────────────────────────────────────────────────────────────────
def haversine_matrix(la, lo):
    R = 6371000.0
    lr  = np.radians(la); lor = np.radians(lo)
    dlat = lr[:,None]-lr[None,:]; dlon = lor[:,None]-lor[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

print('距離行列を計算中...')
D = haversine_matrix(lats_a, lons_a)

# ── Union-Find クラスタリング ──────────────────────────────────────────────────
parent = list(range(N))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y): parent[find(x)] = find(y)

mask_connect = (years_a[:,None] != years_a[None,:]) & \
               (D <= DIST_THR) & \
               (np.abs(devs_a[:,None] - devs_a[None,:]) <= DEV_THR)
for r, c in zip(*np.where(np.tril(mask_connect, k=-1))):
    union(r, c)

cluster_id = np.array([find(i) for i in range(N)])

# 300m以内の異年ペア（同・別クラスタ問わず）
mask_near = (years_a[:,None] != years_a[None,:]) & (D <= DIST_THR)
ri, ci = np.where(np.tril(mask_near, k=-1))
near_pairs = list(zip(ri.tolist(), ci.tolist()))

print(f'300m以内の異年ペア: {len(near_pairs)}')

# ── 最良の「密集地点」を自動検出 ─────────────────────────────────────────────
# 赤線（別クラスタ）と緑線（同クラスタ）が両方ある
# かつ関係圃場が実際に近接（bbox が小さい）な地点を選ぶ

# 各ペアの中点座標を求め、その周辺 500m 以内のペア数をカウント
def n_pairs_near_point(clat, clon, pairs, threshold_m=600):
    """指定座標から threshold_m 以内のペアを返す"""
    result = []
    for r, c in pairs:
        mid_lat = (lats_a[r] + lats_a[c]) / 2
        mid_lon = (lons_a[r] + lons_a[c]) / 2
        dist = haversine_scalar(clat, clon, mid_lat, mid_lon)
        if dist <= threshold_m:
            result.append((r, c))
    return result

def haversine_scalar(la1, lo1, la2, lo2):
    R = 6371000.0
    dlat = math.radians(la2-la1); dlon = math.radians(lo2-lo1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# 全ペアの中点を候補点とし、その周囲のペア密度を評価
best_score = -1
best_center = None
best_pairs_in_area = []

for r, c in near_pairs:
    clat = (lats_a[r] + lats_a[c]) / 2
    clon = (lons_a[r] + lons_a[c]) / 2
    pairs_in = n_pairs_near_point(clat, clon, near_pairs, threshold_m=500)

    # スコア: 同クラスタペアと別クラスタペアの両方があれば高スコア
    n_same = sum(1 for rr, cc in pairs_in if cluster_id[rr] == cluster_id[cc])
    n_diff = sum(1 for rr, cc in pairs_in if cluster_id[rr] != cluster_id[cc])
    score  = min(n_same, n_diff) * 2 + len(pairs_in)  # 両方バランスよくある地点を優先

    if score > best_score:
        best_score = score
        best_center = (clat, clon)
        best_pairs_in_area = pairs_in

print(f'最良密集地点: lat={best_center[0]:.4f}, lon={best_center[1]:.4f}')
print(f'  周辺ペア数: {len(best_pairs_in_area)}  (score={best_score})')

# その地点の全関係圃場インデックスを収集
involved_indices = set()
for r, c in best_pairs_in_area:
    involved_indices.add(r)
    involved_indices.add(c)
involved_indices = sorted(involved_indices)

print(f'  関係圃場数: {len(involved_indices)}  field_ids: '
      f'{sorted(set(fids_a[i] for i in involved_indices))}')

# ── 地図の表示範囲を決定（関係圃場 + マージン） ──────────────────────────────
inv_lats = lats_a[involved_indices]
inv_lons = lons_a[involved_indices]
lat_center = inv_lats.mean()
lon_center = inv_lons.mean()

R = 6371000.0
m_per_lat = R * math.pi / 180
m_per_lon = R * math.cos(math.radians(lat_center)) * math.pi / 180

# bboxは実圃場の幅 + 30%マージン
lat_spread = max(inv_lats.max() - inv_lats.min(), 300/m_per_lat)
lon_spread = max(inv_lons.max() - inv_lons.min(), 300/m_per_lon)
margin_lat = lat_spread * 0.5
margin_lon = lon_spread * 0.5
lat_min = inv_lats.min() - margin_lat;  lat_max = inv_lats.max() + margin_lat
lon_min = inv_lons.min() - margin_lon;  lon_max = inv_lons.max() + margin_lon

# ── クラスタに色を割り当て ─────────────────────────────────────────────────────
unique_clusters = sorted(set(cluster_id[i] for i in involved_indices))
PALETTE = [
    '#e74c3c',  # 赤（多収系）
    '#2980b9',  # 青（低収系）
    '#27ae60',  # 緑
    '#8e44ad',  # 紫
    '#e67e22',  # オレンジ
    '#16a085',  # ティール
    '#c0392b',  # 暗赤
    '#1a6fa8',  # 暗青
]
cl_color = {cl: PALETTE[i % len(PALETTE)] for i, cl in enumerate(unique_clusters)}
GREY = '#999999'

# 年マーカー
MARKER_YR = {2015: 'o', 2016: 's', 2017: '^', 2018: 'D'}
SIZE_YR   = {2015: 200, 2016: 200, 2017: 220, 2018: 190}

# ── 図を作成（大きめのプレゼン向け） ─────────────────────────────────────────
fig = plt.figure(figsize=(22, 10), facecolor='white')
fig.patch.set_facecolor('#f8f9fa')

ax_map = fig.add_axes([0.03, 0.08, 0.44, 0.82])   # 左: 地図
ax_dev = fig.add_axes([0.54, 0.08, 0.43, 0.82])   # 右: 偏差推移

# ============================================================
# 左パネル: 拡大地図
# ============================================================
ax_map.set_facecolor('#e8eef5')
ax_map.set_xlim(lon_min, lon_max)
ax_map.set_ylim(lat_min, lat_max)

# ① 近接ペアの接続線（関係圃場のみ）
for r, c in best_pairs_in_area:
    same_cl  = (cluster_id[r] == cluster_id[c])
    dev_diff = abs(devs_a[r] - devs_a[c])
    dist_m   = D[r, c]

    if same_cl:
        lc, lw, ls, zo = '#27ae60', 2.5, '-',  3
    else:
        lc, lw, ls, zo = '#e74c3c', 2.5, '--', 3

    ax_map.plot([lons_a[r], lons_a[c]], [lats_a[r], lats_a[c]],
                color=lc, lw=lw, linestyle=ls, alpha=0.75, zorder=zo,
                solid_capstyle='round')



# ② 各圃場の点とラベル（関係圃場のみ）
# ラベルをクラスタ重心に対する相対位置で振り分ける（重なり防止）
lon_c = np.mean([lons_a[i] for i in involved_indices])
lat_c = np.mean([lats_a[i] for i in involved_indices])

for enum_i, i in enumerate(involved_indices):
    cl   = cluster_id[i]
    color = cl_color.get(cl, GREY)
    yr   = int(years_a[i])
    fid  = int(fids_a[i])
    dv   = devs_a[i]
    mk   = MARKER_YR.get(yr, 'o')
    sz   = SIZE_YR.get(yr, 180)

    ax_map.scatter(lons_a[i], lats_a[i], s=sz, c=color, marker=mk,
                   edgecolors='white', linewidths=1.5, zorder=7, alpha=0.95)

    # ラベル: field_id / 年 / 偏差値
    sign  = '+' if dv >= 0 else ''
    label = f'fid{fid} / {yr}年\n{sign}{dv:.0f}kg'

    # 重心に対して右側 → ラベルを右へ、左側 → ラベルを左へ
    # 上側 → さらに上へ、下側 → 下へ（矢印で繋ぐ）
    is_right = lons_a[i] >= lon_c
    is_upper = lats_a[i] >= lat_c

    x_off_pts = 30 if is_right else -30
    y_off_pts = 20 if is_upper else -20
    ha_str    = 'left' if is_right else 'right'
    va_str    = 'bottom' if is_upper else 'top'

    ax_map.annotate(
        label,
        xy=(lons_a[i], lats_a[i]),
        xytext=(x_off_pts, y_off_pts),
        textcoords='offset points',
        ha=ha_str, va=va_str,
        fontsize=8.5, color=color, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
        arrowprops=dict(arrowstyle='-', color=color, lw=0.8, alpha=0.7),
        zorder=8
    )

# ③ 300m スケールバー
scale_deg_lon = DIST_THR / m_per_lon
sb_lon0 = lon_max - scale_deg_lon - (lon_max - lon_min) * 0.03
sb_lat  = lat_min + (lat_max - lat_min) * 0.04
ax_map.annotate('', xy=(sb_lon0 + scale_deg_lon, sb_lat),
                xytext=(sb_lon0, sb_lat),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2.0))
ax_map.text(sb_lon0 + scale_deg_lon / 2, sb_lat + (lat_max-lat_min)*0.012,
            '300 m', ha='center', va='bottom', fontsize=10,
            fontweight='bold', color='black')

# 軸設定
ax_map.set_xlabel('経度', fontsize=12)
ax_map.set_ylabel('緯度', fontsize=12)
ax_map.set_title('拡大地図\n（各圃場の位置と年度間の結びつき）',
                 fontsize=13, fontweight='bold', pad=10)
ax_map.grid(True, alpha=0.25, color='white', linewidth=1.5)
ax_map.tick_params(labelsize=10)

# 凡例（線種）
line_handles = [
    Line2D([0],[0], color='#27ae60', lw=2.5, linestyle='-',
           label=f'同クラスタ（偏差差 ≤ {DEV_THR}kg → 年またぎ連続記録として採用）'),
    Line2D([0],[0], color='#e74c3c', lw=2.5, linestyle='--',
           label=f'別クラスタ（偏差差 > {DEV_THR}kg → 収量水準が異なり分離）'),
]
yr_handles = [
    Line2D([0],[0], marker=mk, color='gray', markersize=10, linestyle='None',
           label=f'{yr}年')
    for yr, mk in MARKER_YR.items()
]
ax_map.legend(handles=line_handles + yr_handles, fontsize=9,
              loc='upper left', framealpha=0.92, edgecolor='#cccccc')

# ============================================================
# 右パネル: 収量偏差推移（field_id 個別）
# ============================================================
ax_dev.set_facecolor('#fafafa')
ax_dev.axhline(0, color='black', lw=2.0, alpha=0.6, zorder=3)
ax_dev.axhspan(0, 999, alpha=0.05, color='#e74c3c', zorder=1)
ax_dev.axhspan(-999, 0, alpha=0.05, color='#2980b9', zorder=1)

# 各 field_id を個別にプロット
plotted_fids = {}  # fid -> color

for i in involved_indices:
    fid   = int(fids_a[i])
    cl    = cluster_id[i]
    color = cl_color.get(cl, GREY)

    if fid not in plotted_fids:
        plotted_fids[fid] = color

# field_id ごとに全年のデータを取得してプロット
for fid, color in sorted(plotted_fids.items()):
    fid_mask = df['field_id'] == fid
    fid_df   = df[fid_mask].sort_values('year')

    yrs  = fid_df['year'].tolist()
    dvs  = fid_df['yield_dev'].tolist()
    yldb = fid_df['yield'].tolist()

    ax_dev.plot(yrs, dvs, '-o', color=color, lw=2.5,
                markersize=12, alpha=0.90, zorder=4,
                markeredgecolor='white', markeredgewidth=1.5)

    # ラベル: fid, 収量偏差, 実収量
    # field_id のインデックスで左右を交互に振り分け（同年ラベル重なり防止）
    fid_idx   = sorted(plotted_fids.keys()).index(fid)
    is_right_dev = (fid_idx % 2 == 0)   # 偶数 fid → 右、奇数 → 左
    x_off_dev = 38 if is_right_dev else -38
    ha_dev    = 'left' if is_right_dev else 'right'

    for yr, dv, yl in zip(yrs, dvs, yldb):
        sign  = '+' if dv >= 0 else ''
        label = f'fid{fid}  {sign}{dv:.0f}\n({yl:.0f}kg)'
        y_off = 14 if dv >= 0 else -14
        ax_dev.annotate(
            label,
            xy=(yr, dv), xytext=(x_off_dev, y_off),
            textcoords='offset points',
            ha=ha_dev, va='bottom' if dv >= 0 else 'top',
            fontsize=8.5, color=color, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
            arrowprops=dict(arrowstyle='-', color=color, lw=0.8, alpha=0.7),
            zorder=5
        )

# Y軸範囲
all_dvs = [devs_a[i] for i in involved_indices]
y_mg = 70
y_lo = min(all_dvs) - y_mg
y_hi = max(all_dvs) + y_mg
ax_dev.set_ylim(y_lo, y_hi)

ax_dev.set_xticks(YEARS)
ax_dev.set_xticklabels([f'{y}年' for y in YEARS], fontsize=12)
ax_dev.set_xlabel('年度', fontsize=12)
ax_dev.set_ylabel('収量偏差 (kg/10a)\n= 実収量 − その年の全圃場平均', fontsize=11)
ax_dev.set_title('各 field_id の年度別収量偏差推移\n（同色 = 同クラスタ、平均化なし）',
                 fontsize=13, fontweight='bold', pad=10)
ax_dev.grid(True, alpha=0.3, axis='y')
ax_dev.tick_params(labelsize=10)

# クラスタ凡例（右グラフ）
unique_cls_in_area = sorted(set(cl_color.keys()))
cl_handles = []
for ci, cl in enumerate(unique_cls_in_area):
    color      = cl_color[cl]
    cl_members = [i for i in involved_indices if cluster_id[i] == cl]
    fids_in_cl = sorted(set(fids_a[i] for i in cl_members))
    cl_handles.append(
        Line2D([0],[0], color=color, lw=2.5, marker='o', markersize=10,
               markeredgecolor='white', markeredgewidth=1.5,
               label=f'クラスタ{ci+1}  fid={fids_in_cl}')
    )

ax_dev.legend(handles=cl_handles, fontsize=9, loc='best',
              framealpha=0.92, edgecolor='#cccccc')

# ── タイトルと保存 ─────────────────────────────────────────────────────────────
fig.suptitle(
    f'近接圃場の年またぎ紐づけ検証\n'
    f'（距離閾値: {DIST_THR}m ／ 収量偏差差閾値: {DEV_THR}kg/10a）\n'
    f'中心座標: lat={best_center[0]:.4f}, lon={best_center[1]:.4f}',
    fontsize=14, fontweight='bold', y=0.98
)

out_path = f'{OUT_DIR}/best_area_zoom.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
plt.close(fig)
print(f'\nプレゼン用拡大図: {out_path}')
