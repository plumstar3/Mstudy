"""
build_past_yield_features.py
===========================================================
【目的】
  Union-Find クラスタリングを使わず、各 (field_id, year) ごとに
  「その圃場から見て過去かつ近接かつ収量水準が近い圃場」を
  直接検索して過去収量特徴量を生成する。

【アルゴリズム（圃場中心アプローチ）】
  予測対象: (fid_t, year_t)
    ①  year_j < year_t  ← 過去年
    ②  dist(fid_t, fid_j) ≤ DIST_THR (300m)  ← 近接
    ③  |dev_t - dev_j|  ≤ DEV_THR  (70kg)  ← 収量水準が近い

  条件を満たす (fid_j, year_j) を集めてその収量の
  mean / max / min / std / count を特徴量として出力する。

【Union-Find との違い】
  - 推移性がないため「A↔B↔C だが A と C は閾値外」という
    過剰連結が発生しない。
  - 各圃場が独立に過去記録を検索するため、
    「この圃場の過去記録はこれ」が直接決まる。

【出力】
  outputs/data_analysis/past_yield_features_v2.csv
    field_id, year, past_yield_n, past_yield_mean, past_yield_max,
    past_yield_min, past_yield_std, past_dev_mean,
    past_fids      ← どの field_id が過去記録として採用されたか
"""

import sqlite3, os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ── 日本語フォント ─────────────────────────────────────────────────────────────
_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR  = 'outputs/data_analysis'
os.makedirs(OUT_DIR, exist_ok=True)
FIELD_DB = 'data/processed/FieldData_fieldid.db'

DIST_THR = 300   # m
DEV_THR  = 70    # kg/10a
YEARS    = [2015, 2016, 2017, 2018]

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

print('年別平均収量:')
for yr, m in year_means.items():
    print(f'  {yr}年: {m:.1f} kg/10a')

lats_a  = df['lat'].values
lons_a  = df['lon'].values
years_a = df['year'].values
devs_a  = df['yield_dev'].values
fids_a  = df['field_id'].values
N       = len(df)

# ── Haversine 距離行列 ─────────────────────────────────────────────────────────
print('\n距離行列を計算中...')

def haversine_matrix(la, lo):
    R = 6371000.0
    lr  = np.radians(la); lor = np.radians(lo)
    dlat = lr[:,None] - lr[None,:]; dlon = lor[:,None] - lor[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

D = haversine_matrix(lats_a, lons_a)

# ── 圃場中心アプローチ：各 (fid_t, year_t) の過去記録を検索 ─────────────────
print('各圃場の過去記録を検索中...')

feature_rows = []

for t_idx in range(N):
    fid_t  = int(fids_a[t_idx])
    year_t = int(years_a[t_idx])
    dev_t  = devs_a[t_idx]

    # 条件①②③を同時に評価
    # ①  year_j < year_t
    # ②  dist ≤ DIST_THR
    # ③  |dev_t - dev_j| ≤ DEV_THR
    past_mask = (
        (years_a < year_t) &                          # ① 過去年
        (D[t_idx] <= DIST_THR) &                      # ② 距離条件
        (np.abs(dev_t - devs_a) <= DEV_THR)           # ③ 収量水準条件
    )
    # 自分自身は除外（念のため）
    past_mask[t_idx] = False

    past_indices = np.where(past_mask)[0]

    if len(past_indices) == 0:
        feature_rows.append({
            'field_id'       : fid_t,
            'year'           : year_t,
            'past_yield_n'   : 0,
            'past_yield_mean': np.nan,
            'past_yield_max' : np.nan,
            'past_yield_min' : np.nan,
            'past_yield_std' : np.nan,
            'past_dev_mean'  : np.nan,
            'past_fids'      : '',
        })
    else:
        past_yields = df.iloc[past_indices]['yield'].values
        past_devs   = df.iloc[past_indices]['yield_dev'].values
        past_fids   = sorted(set(fids_a[j] for j in past_indices))

        feature_rows.append({
            'field_id'       : fid_t,
            'year'           : year_t,
            'past_yield_n'   : len(past_indices),
            'past_yield_mean': round(float(np.mean(past_yields)), 1),
            'past_yield_max' : round(float(np.max(past_yields)), 1),
            'past_yield_min' : round(float(np.min(past_yields)), 1),
            'past_yield_std' : round(float(np.std(past_yields)), 1),
            'past_dev_mean'  : round(float(np.mean(past_devs)), 1),
            'past_fids'      : ','.join(str(f) for f in past_fids),
        })

feat_df = pd.DataFrame(feature_rows)
feat_df.to_csv(f'{OUT_DIR}/past_yield_features_v2.csv', index=False, encoding='utf-8-sig')

# ── 結果サマリ ─────────────────────────────────────────────────────────────────
has_past = feat_df[feat_df['past_yield_n'] > 0]
print(f'\n【結果】')
print(f'  全サンプル: {len(feat_df)}')
print(f'  過去記録あり: {len(has_past)} ({100*len(has_past)/len(feat_df):.1f}%)')
print(f'  過去記録なし: {len(feat_df)-len(has_past)}')
print()
print('  年別（過去記録あり件数）:')
for yr in YEARS:
    yr_all  = feat_df[feat_df['year'] == yr]
    yr_has  = has_past[has_past['year'] == yr]
    n_mean  = yr_has['past_yield_n'].mean() if len(yr_has) > 0 else 0
    print(f'    {yr}年: {len(yr_has):3d}/{len(yr_all):3d} 件  '
          f'（1件あたり平均 {n_mean:.1f} 過去圃場）')

print(f'\n  CSV: {OUT_DIR}/past_yield_features_v2.csv')

# ── 可視化: 拡大エリアで Union-Find vs 圃場中心の比較 ──────────────────────────
# 先ほどの密集エリアで「どの圃場が過去記録として採用されたか」を示す
# ターゲット圃場リスト: 先ほどのエリアの 2017 年記録
TARGET_FIDS = [509, 522, 525, 527, 528, 531, 534, 535, 538, 541]  # 2017年
TARGET_YEAR = 2017

print(f'\n【{TARGET_YEAR}年ターゲット圃場ごとの過去記録】')
print(f'  (距離<={DIST_THR}m / 偏差差<={DEV_THR}kg)')
print(f'  {"field_id":>10} | {"dev_t":>7} | {"past_n":>6} | past_fids (dev_j)')

# 過去圃場のdevも表示
for fid_t in sorted(TARGET_FIDS):
    t_rows = feat_df[(feat_df['field_id'] == fid_t) & (feat_df['year'] == TARGET_YEAR)]
    if len(t_rows) == 0:
        continue
    r = t_rows.iloc[0]

    # dev_t を取得
    dev_t_val = df[(df['field_id'] == fid_t) & (df['year'] == TARGET_YEAR)]['yield_dev']
    if len(dev_t_val) == 0:
        continue
    dev_t_val = dev_t_val.values[0]

    # past_fids の詳細（各 fid の year と dev）
    if r['past_fids']:
        past_fid_list = [int(f) for f in r['past_fids'].split(',')]
        details = []
        for pf in past_fid_list:
            pf_rows = df[(df['field_id'] == pf) & (df['year'] < TARGET_YEAR)]
            for _, pr in pf_rows.iterrows():
                sign = '+' if pr['yield_dev'] >= 0 else ''
                details.append(f'fid{pf}({int(pr["year"])}:{sign}{pr["yield_dev"]:.0f})')
        detail_str = '  '.join(details)
    else:
        detail_str = '（なし）'

    sign_t = '+' if dev_t_val >= 0 else ''
    print(f'  fid{fid_t:>6} | {sign_t}{dev_t_val:>5.0f} | {int(r["past_yield_n"]):>6} | {detail_str}')

# ── 可視化図：2017年ターゲット圃場とその過去記録の対応 ──────────────────────
# エリア内の圃場を地図上に表示し、
# ターゲット(2017)→過去記録(2016) の対応を矢印で示す

# エリアに関係する全インデックスを取得
area_fids = set(TARGET_FIDS) | {511, 523, 526, 510}  # + 2016年記録
area_mask = np.isin(fids_a, list(area_fids))
area_idx  = np.where(area_mask)[0]

inv_lats = lats_a[area_idx]
inv_lons = lons_a[area_idx]

lat_center = inv_lats.mean();  lon_center = inv_lons.mean()
import math
R = 6371000.0
m_per_lat = R * math.pi / 180
m_per_lon = R * math.cos(math.radians(lat_center)) * math.pi / 180

lat_sp = max(inv_lats.max() - inv_lats.min(), 300/m_per_lat)
lon_sp = max(inv_lons.max() - inv_lons.min(), 300/m_per_lon)
lat_min = inv_lats.min() - lat_sp*0.5;  lat_max = inv_lats.max() + lat_sp*0.5
lon_min = inv_lons.min() - lon_sp*0.5;  lon_max = inv_lons.max() + lon_sp*0.5

MARKER_YR = {2015: 'o', 2016: 's', 2017: '^', 2018: 'D'}

# field_id → 固定色
COLORS = [
    '#e74c3c','#2980b9','#27ae60','#8e44ad','#e67e22',
    '#16a085','#c0392b','#1a6fa8','#f39c12','#7f8c8d',
    '#d35400','#2ecc71','#9b59b6','#3498db',
]
fid_list_sorted = sorted(area_fids)
fid_to_color    = {fid: COLORS[i % len(COLORS)] for i, fid in enumerate(fid_list_sorted)}

fig, (ax_map, ax_tbl) = plt.subplots(1, 2, figsize=(20, 9), facecolor='white')
fig.patch.set_facecolor('#f8f9fa')

ax_map.set_facecolor('#e8eef5')
ax_map.set_xlim(lon_min, lon_max)
ax_map.set_ylim(lat_min, lat_max)

# 全圃場を描画
for i in area_idx:
    fid  = int(fids_a[i])
    yr   = int(years_a[i])
    dv   = devs_a[i]
    color = fid_to_color.get(fid, '#aaaaaa')
    mk    = MARKER_YR.get(yr, 'o')
    sz    = 280 if yr == 2017 else 200

    ax_map.scatter(lons_a[i], lats_a[i], s=sz, c=color, marker=mk,
                   edgecolors='white', linewidths=1.5, zorder=5)

    is_right = lons_a[i] >= lon_center
    is_upper = lats_a[i] >= lat_center
    x_off = 28 if is_right else -28
    y_off = 18 if is_upper else -18
    ha_s  = 'left' if is_right else 'right'
    va_s  = 'bottom' if is_upper else 'top'

    sign = '+' if dv >= 0 else ''
    ax_map.annotate(
        f'fid{fid}/{yr}\n{sign}{dv:.0f}kg',
        xy=(lons_a[i], lats_a[i]),
        xytext=(x_off, y_off), textcoords='offset points',
        ha=ha_s, va=va_s, fontsize=8, color=color, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
        arrowprops=dict(arrowstyle='-', color=color, lw=0.7, alpha=0.6),
        zorder=8
    )

# 2017年→過去記録への矢印を描画
for fid_t in sorted(TARGET_FIDS):
    t_rows = feat_df[(feat_df['field_id'] == fid_t) & (feat_df['year'] == TARGET_YEAR)]
    if len(t_rows) == 0 or t_rows.iloc[0]['past_yield_n'] == 0:
        continue

    # ターゲットの座標
    t_df_row = df[(df['field_id'] == fid_t) & (df['year'] == TARGET_YEAR)]
    if len(t_df_row) == 0:
        continue
    t_lat = t_df_row['lat'].values[0]
    t_lon = t_df_row['lon'].values[0]
    t_color = fid_to_color.get(fid_t, '#aaaaaa')

    # 過去記録圃場への矢印
    past_fid_list_str = t_rows.iloc[0]['past_fids']
    if not past_fid_list_str:
        continue
    for pf in [int(f) for f in past_fid_list_str.split(',')]:
        pf_rows = df[(df['field_id'] == pf) & (df['year'] < TARGET_YEAR)]
        for _, pr in pf_rows.iterrows():
            ax_map.annotate(
                '', xy=(pr['lon'], pr['lat']),
                xytext=(t_lon, t_lat),
                arrowprops=dict(
                    arrowstyle='->', color=t_color, lw=1.8,
                    connectionstyle='arc3,rad=0.1',
                    alpha=0.7
                ),
                zorder=4
            )

# スケールバー
scale_deg = DIST_THR / m_per_lon
sb_lon0   = lon_max - scale_deg - (lon_max-lon_min)*0.03
sb_lat    = lat_min + (lat_max-lat_min)*0.04
ax_map.annotate('', xy=(sb_lon0+scale_deg, sb_lat), xytext=(sb_lon0, sb_lat),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2.0))
ax_map.text(sb_lon0+scale_deg/2, sb_lat+(lat_max-lat_min)*0.012,
            '300 m', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_map.set_xlabel('経度', fontsize=12)
ax_map.set_ylabel('緯度', fontsize=12)
ax_map.set_title(
    f'圃場中心アプローチ: 2017年ターゲット → 過去記録の対応\n'
    f'（矢印: ターゲット→過去記録圃場 / 同色=同fid）',
    fontsize=12, fontweight='bold'
)
ax_map.grid(True, alpha=0.25, color='white', linewidth=1.5)

yr_handles = [Line2D([0],[0], marker=mk, color='gray', markersize=9, linestyle='None',
                     label=f'{yr}年（{"ターゲット" if yr==2017 else "過去記録"}）')
              for yr, mk in MARKER_YR.items() if yr in [2016, 2017]]
ax_map.legend(handles=yr_handles, fontsize=9, loc='upper left', framealpha=0.9)

# ── 右パネル: 対応表 ─────────────────────────────────────────────────────────
ax_tbl.axis('off')
rows_data = [['2017年ターゲット\n(fid / 偏差)', '採用された過去記録\n(fid / 年 / 偏差)']]
for fid_t in sorted(TARGET_FIDS):
    t_rows = feat_df[(feat_df['field_id'] == fid_t) & (feat_df['year'] == TARGET_YEAR)]
    dev_t_val = df[(df['field_id'] == fid_t) & (df['year'] == TARGET_YEAR)]['yield_dev']
    if len(dev_t_val) == 0:
        continue
    dev_t_val = dev_t_val.values[0]
    sign_t = '+' if dev_t_val >= 0 else ''

    if len(t_rows) == 0 or t_rows.iloc[0]['past_yield_n'] == 0:
        past_str = '（なし）'
    else:
        past_fid_list = [int(f) for f in t_rows.iloc[0]['past_fids'].split(',')]
        parts = []
        for pf in past_fid_list:
            pf_rows = df[(df['field_id'] == pf) & (df['year'] < TARGET_YEAR)]
            for _, pr in pf_rows.iterrows():
                s = '+' if pr['yield_dev'] >= 0 else ''
                parts.append(f'fid{pf}({int(pr["year"])}) {s}{pr["yield_dev"]:.0f}kg')
        past_str = '\n'.join(parts)

    rows_data.append([f'fid{fid_t}\n{sign_t}{dev_t_val:.0f}kg', past_str])

tbl = ax_tbl.table(
    cellText=rows_data[1:], colLabels=rows_data[0],
    cellLoc='left', loc='center',
    colWidths=[0.28, 0.62]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 2.5)
# ヘッダ行を強調
for col in range(2):
    tbl[(0, col)].set_facecolor('#2c3e50')
    tbl[(0, col)].set_text_props(color='white', fontweight='bold')
# 行ごとに交互着色
for row in range(1, len(rows_data)):
    bg = '#f0f4f8' if row % 2 == 0 else 'white'
    for col in range(2):
        tbl[(row, col)].set_facecolor(bg)

ax_tbl.set_title(
    f'各2017年圃場の「過去記録」対応表\n'
    f'（距離≤{DIST_THR}m かつ |偏差差|≤{DEV_THR}kg）',
    fontsize=12, fontweight='bold', pad=15
)

fig.suptitle(
    '圃場中心アプローチによる過去記録紐づけ\n'
    '（Union-Find を使わず、各圃場が独立に過去記録を検索）',
    fontsize=13, fontweight='bold', y=0.99
)

out_path = f'{OUT_DIR}/past_links_field_centric.png'
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='#f8f9fa')
plt.close(fig)
print(f'\n  可視化図: {out_path}')
print('\n完了')
