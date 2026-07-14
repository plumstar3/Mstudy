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
    ②  dist(fid_t, fid_j) <= DIST_THR (300m)  ← 近接
    ③  |候補群の偏差中央値 - dev_j| <= DEV_THR (70kg)  ← 近傍内分布で外れ値除去
        (対象圃場の収量は一切参照しない: リークなし)

  候補の信頼性要件：候補圃場が 2 件未満の場合は NaN（分布計算不可）

  条件①②を満たす候補の中から外れ値を除いた圃場の収量の
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

OUT_DIR     = 'outputs/data_analysis'
os.makedirs(OUT_DIR, exist_ok=True)
FIELD_DB    = 'data/processed/FieldData_fieldid.db'
WEATHER_DB  = 'data/processed/weather_database_fieldid.db'
GDD_CSV     = 'outputs/gdd/gdd_daily.csv'

DIST_THR       = 300        # m  近接閾値
YEARS          = [2015, 2016, 2017, 2018]
WEATHER_COLS     = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
# 変数ごとに取る統計量を限定（135次元 → 39次元に削減）
WEATHER_STAT_MAP = {
    'TMP_mea': ['mean'],              # 平均気温: 平均のみ
    'TMP_max': ['mean', 'max'],       # 最高気温: 高温障害のため max も保持
    'TMP_min': ['mean', 'min'],       # 最低気温: 冷害・霜害のため min も保持
    'APCPRA':  ['mean', 'max'],       # 降水量: 豪雨ダメージのため max も保持
    'SSD':     ['mean'],              # 日照時間: 平均のみ
    'GSR':     ['mean'],              # 日射量: 平均のみ
    'WIND':    ['mean', 'max'],       # 風速: 台風・倒伏のため max も保持
    'SWE':     ['mean'],              # 積雪相当水量: 平均のみ
    'RH':      ['mean'],              # 湿度: 平均のみ
}
GDD_THRESHOLDS = [600, 1000]
HARM_COLS      = ['sick', 'wet', 'typhoon', 'unripen', 'weed']

# ── 収量・位置データ読み込み ────────────────────────────────────────────────────
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

# ── GDD 期間ラベル ─────────────────────────────────────────────────────────────
print('GDD 読み込み...', end=' ')
gdd_df = pd.read_csv(GDD_CSV, encoding='utf-8-sig')
gdd_df['date'] = pd.to_datetime(gdd_df['date'])
cum_col = [c for c in gdd_df.columns if 'GDD' in c or 'gdd' in c.lower()][-1]
th1, th2 = GDD_THRESHOLDS
gdd_df['period'] = 1
gdd_df.loc[gdd_df[cum_col] > th1, 'period'] = 2
gdd_df.loc[gdd_df[cum_col] > th2, 'period'] = 3
gdd_df = gdd_df[['field_id', 'year', 'date', 'period']]
print(f'{len(gdd_df):,} 行')

# ── 気象データ読み込み（全 field_id 対象）─────────────────────────────────────
fids  = sorted(df['field_id'].unique().tolist())
print(f'気象データ読み込み ({len(fids)} 圃場)...', end=' ')
conn_w = sqlite3.connect(WEATHER_DB)
fid_ph  = ','.join(['?'] * len(fids))
yr_ph   = ','.join(f"'{y}'" for y in YEARS)
col_str = ', '.join(WEATHER_COLS)
weather_df = pd.read_sql(f'''
    SELECT field_id, date, {col_str} FROM weather_data
    WHERE field_id IN ({fid_ph})
      AND CAST(SUBSTR(date,1,4) AS INTEGER) IN ({yr_ph})
    ORDER BY field_id, date''', conn_w, params=fids)
conn_w.close()
weather_df['field_id'] = weather_df['field_id'].astype(int)
weather_df['date']     = pd.to_datetime(weather_df['date'])
print(f'{len(weather_df):,} 行')

# ── GDD 期間別気象特徴量（全 field_id × year）────────────────────────────────
print('GDD期間別特徴量計算...')
merged_gdd = gdd_df.merge(weather_df[['field_id', 'date'] + WEATHER_COLS],
                          on=['field_id', 'date'], how='left')
agg_dict = {v: stats for v, stats in WEATHER_STAT_MAP.items()}
grp = merged_gdd.groupby(['field_id', 'year', 'period']).agg(agg_dict)
grp_pivot = grp.unstack('period')
grp_pivot.columns = [f'{v}_p{int(p)}_{s}' for v, s, p in grp_pivot.columns]
gdd_feat_cols = [f'{v}_p{p}_{s}'
                 for p in [1, 2, 3]
                 for v, stats in WEATHER_STAT_MAP.items()
                 for s in stats]
for col in gdd_feat_cols:
    if col not in grp_pivot.columns:
        grp_pivot[col] = np.nan
feat_gdd = grp_pivot[gdd_feat_cols].reset_index()
print(f'  気象特徴量次元: {len(gdd_feat_cols)}')

# ── 病害データ読み込み ─────────────────────────────────────────────────────────
print('病害データ読み込み...')
conn = sqlite3.connect(FIELD_DB)
harm_df = pd.read_sql(
    f"SELECT field_id, year, {', '.join(HARM_COLS)} FROM Harm "
    "WHERE field_id IS NOT NULL", conn)
conn.close()
harm_df['field_id'] = harm_df['field_id'].astype(int)
harm_df['year']     = harm_df['year'].astype(int)
for c in HARM_COLS:
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce')
print(f'  病害データ: {len(harm_df)} 件')

# ── df に特徴量をマージして行列化 ─────────────────────────────────────────────
df = df.merge(feat_gdd, on=['field_id', 'year'], how='left')
df = df.merge(harm_df,  on=['field_id', 'year'], how='left')

# 高速参照用の numpy 配列
lats_a       = df['lat'].values
lons_a       = df['lon'].values
years_a      = df['year'].values
fids_a       = df['field_id'].values
N            = len(df)
weather_mat  = df[gdd_feat_cols].to_numpy(dtype=np.float64)   # (N, 135)
harm_mat     = df[HARM_COLS].to_numpy(dtype=np.float64)        # (N, 5)

# ── Haversine 距離行列 ─────────────────────────────────────────────────────────
print('\n距離行列を計算中...')

def haversine_matrix(la, lo):
    R = 6371000.0
    lr  = np.radians(la); lor = np.radians(lo)
    dlat = lr[:,None] - lr[None,:]; dlon = lor[:,None] - lor[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

D = haversine_matrix(lats_a, lons_a)

# ── 出力列名定義 ─────────────────────────────────────────────────────────────
PAST_WX_COLS   = [f'past_{c}' for c in gdd_feat_cols]
PAST_HARM_COLS = [f'past_harm_{c}' for c in HARM_COLS]

# ── 圃場中心アプローチ：各 (fid_t, year_t) の過去記録を検索 ─────────────────
# 条件①: year_j < year_t  （全過去年）
# 条件②: dist <= DIST_THR （近接 300m）
# 条件③なし
# 出力: past_yield_mean / past気象135特徴量 / past病害5特徴量
print('各圃場の過去記録を検索中... [全過去年 × 距離300m]')

def nan_row_wx():
    return {c: np.nan for c in PAST_WX_COLS}

def nan_row_harm():
    return {c: np.nan for c in PAST_HARM_COLS}

feature_rows = []

for t_idx in range(N):
    fid_t  = int(fids_a[t_idx])
    year_t = int(years_a[t_idx])

    # 条件①②
    past_mask = (
        (years_a < year_t) &           # ① 全過去年
        (D[t_idx] <= DIST_THR)         # ② 距離条件
    )
    past_mask[t_idx] = False
    past_indices = np.where(past_mask)[0]

    base = {'field_id': fid_t, 'year': year_t}

    if len(past_indices) == 0:
        row = {**base,
               'past_yield_n': 0, 'past_yield_mean': np.nan,
               'past_yield_max': np.nan, 'past_yield_min': np.nan,
               'past_yield_std': np.nan, 'past_dev_mean': np.nan,
               'past_fids': '',
               **nan_row_wx(), **nan_row_harm()}
    else:
        past_yields = df.iloc[past_indices]['yield'].values
        past_devs   = df.iloc[past_indices]['yield_dev'].values
        past_fids_l = sorted(set(int(fids_a[j]) for j in past_indices))

        # 過去候補の気象特徴量平均
        past_wx   = np.nanmean(weather_mat[past_indices], axis=0)  # (135,)
        # 過去候補の病害特徴量平均
        past_harm = np.nanmean(harm_mat[past_indices],   axis=0)   # (5,)

        wx_dict   = {c: (round(float(v), 4) if not np.isnan(v) else np.nan)
                     for c, v in zip(PAST_WX_COLS, past_wx)}
        harm_dict = {c: (round(float(v), 4) if not np.isnan(v) else np.nan)
                     for c, v in zip(PAST_HARM_COLS, past_harm)}

        row = {**base,
               'past_yield_n'   : len(past_indices),
               'past_yield_mean': round(float(np.mean(past_yields)), 1),
               'past_yield_max' : round(float(np.max(past_yields)), 1),
               'past_yield_min' : round(float(np.min(past_yields)), 1),
               'past_yield_std' : round(float(np.std(past_yields)), 1) if len(past_indices) > 1 else 0.0,
               'past_dev_mean'  : round(float(np.mean(past_devs)), 1),
               'past_fids'      : ','.join(str(f) for f in past_fids_l),
               **wx_dict, **harm_dict}

    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows)
feat_df.to_csv(f'{OUT_DIR}/past_yield_features_v2.csv', index=False, encoding='utf-8-sig')

# ── 結果サマリ ─────────────────────────────────────────────────────────────────
has_past = feat_df[feat_df['past_yield_n'] > 0]
print(f'\n【結果】全過去年 × 距離{DIST_THR}m 以内 / 過去気象135 + 病害5特徴量')
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
print(f'  総列数: {len(feat_df.columns)} 列（yield+気象{len(PAST_WX_COLS)}+病害{len(PAST_HARM_COLS)}）')



# ── 可視化: 拡大エリアで Union-Find vs 圃場中心の比較 ──────────────────────────
# 先ほどの密集エリアで「どの圃場が過去記録として採用されたか」を示す
# ターゲット圃場リスト: 先ほどのエリアの 2017 年記録
TARGET_FIDS = [509, 522, 525, 527, 528, 531, 534, 535, 538, 541]  # 2017年
TARGET_YEAR = 2017

print(f'\n【{TARGET_YEAR}年ターゲット圃場ごとの過去記録】')
print(f'  (距離<={DIST_THR}m / 収量水準条件なし)')
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
    f'（距離<={DIST_THR}m / 収量水準条件なし）',
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
