"""
extract_continuous_fields.py
===========================================================
【目的】
  「距離 ≤ 300m かつ 収量偏差差 ≤ DEV_THR」の条件で同一圃場を定義し、
  年をまたぐ連続記録クラスタを抽出する。
  最終的に「各 (field_id, year) の過去年収量統計特徴量」を CSV で出力し、
  回帰モデルの追加入力特徴量として使えるようにする。

【クラスタリング条件（2重フィルタ）】
  1. 距離条件  : Haversine距離 ≤ DIST_THR (= 300 m)
  2. 収量偏差条件: |年内偏差_i - 年内偏差_j| ≤ DEV_THR (= 70 kg/10a)
       ※ 収量偏差 = 収量 - その年の全圃場平均収量
       ※ 異なる年のペアにのみ適用（同年ペアは除外）

【出力】
  outputs/data_analysis/continuous_clusters.csv
    → クラスタIDごとの年別収量一覧
  outputs/data_analysis/past_yield_features.csv
    → (field_id, year) ごとの「過去年収量の平均・最大・最小・std・件数」
       (回帰モデルの追加特徴量として直接使用可能)
  outputs/data_analysis/continuous_clusters_map.png / _yield.png
"""

import sqlite3, os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント
_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR  = 'outputs/data_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

FIELD_DB = 'data/processed/FieldData_fieldid.db'

# ── パラメータ ─────────────────────────────────────────────────────────────────
DIST_THR = 300   # [m]  距離閾値
DEV_THRS = [50, 70, 100]  # [kg/10a] 収量偏差差の閾値（複数試す）
YEARS    = [2015, 2016, 2017, 2018]

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

# ── 収量偏差を計算（年別平均を引く） ──────────────────────────────────────────
year_means = df.groupby('year')['yield'].mean()
df['yield_dev'] = df.apply(lambda r: r['yield'] - year_means[r['year']], axis=1)

print(f'全サンプル数: {len(df)}')
print('年別平均収量:')
for yr, m in year_means.items():
    print(f'  {yr}年: {m:.1f} kg/10a')
print()

# ── Haversine 距離行列 ─────────────────────────────────────────────────────────
def haversine_matrix(lats, lons):
    R = 6371000.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat  = lat_r[:, None] - lat_r[None, :]
    dlon  = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

print('距離行列を計算中...')
lats     = df['lat'].values
lons     = df['lon'].values
years_a  = df['year'].values
devs     = df['yield_dev'].values

D = haversine_matrix(lats, lons)

# ── Union-Find ────────────────────────────────────────────────────────────────
def union_find_clusters(D, years_a, devs, dist_thr, dev_thr):
    """距離 ≤ dist_thr かつ 偏差差 ≤ dev_thr かつ 異なる年 のペアを繋ぐ"""
    n = len(years_a)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # 条件マスク（下三角のみで十分）
    diff_year  = years_a[:, None] != years_a[None, :]   # 異なる年
    near       = D <= dist_thr                            # 距離条件
    dev_close  = np.abs(devs[:, None] - devs[None, :]) <= dev_thr  # 偏差条件
    mask = diff_year & near & dev_close

    rows, cols = np.where(np.tril(mask, k=-1))
    for r, c in zip(rows, cols):
        union(r, c)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    # 年またぎクラスタのみ返す
    multi = []
    for root, members in clusters.items():
        if len(set(years_a[m] for m in members)) > 1:
            multi.append(members)
    return multi

# ── 各閾値で実験 ──────────────────────────────────────────────────────────────
print(f'距離閾値: {DIST_THR}m\n')

best_dev_thr  = None
best_clusters = None

for dev_thr in DEV_THRS:
    clusters = union_find_clusters(D, years_a, devs, DIST_THR, dev_thr)
    total_fids = sum(len(m) for m in clusters)
    print(f'  偏差閾値 {dev_thr:3d} kg → クラスタ数: {len(clusters):3d}  '
          f'field_id 合計: {total_fids:3d}  '
          f'全体の {100*total_fids/len(df):.1f}%')
    # デフォルトは dev_thr=70
    if dev_thr == 70:
        best_dev_thr  = dev_thr
        best_clusters = clusters

# ── 以降は DEV_THR=70 の結果で詳細分析 ──────────────────────────────────────
DEV_THR  = best_dev_thr
clusters = best_clusters

print(f'\n【詳細分析】距離 {DIST_THR}m / 偏差 {DEV_THR}kg 閾値')
print(f'  連続記録クラスタ数: {len(clusters)}')

# ── 1. continuous_clusters.csv : クラスタ年別収量一覧 ─────────────────────────
rows = []
for cl_idx, members in enumerate(clusters):
    sub = df.iloc[members]

    lat_mean = sub['lat'].mean()
    lon_mean = sub['lon'].mean()

    year_yields = {}
    year_devs   = {}
    year_fids   = {}
    for yr in YEARS:
        s = sub[sub['year'] == yr]
        if len(s) > 0:
            year_yields[yr] = round(s['yield'].mean(), 1)
            year_devs[yr]   = round(s['yield_dev'].mean(), 1)
            year_fids[yr]   = ','.join(str(f) for f in sorted(s['field_id'].tolist()))
        else:
            year_yields[yr] = None
            year_devs[yr]   = None
            year_fids[yr]   = None

    valid_y   = [v for v in year_yields.values() if v is not None]
    n_years   = len(set(sub['year'].values))
    yspan     = max(YEARS[i] for i, yr in enumerate(YEARS) if year_yields[yr] is not None) - \
                min(YEARS[i] for i, yr in enumerate(YEARS) if year_yields[yr] is not None)

    rows.append({
        'cluster_id'  : f'cl{cl_idx+1:03d}',
        'n_years'     : n_years,
        'n_fields'    : len(members),
        'year_span'   : yspan,
        'lat_mean'    : round(lat_mean, 5),
        'lon_mean'    : round(lon_mean, 5),
        'yield_2015'  : year_yields[2015],
        'yield_2016'  : year_yields[2016],
        'yield_2017'  : year_yields[2017],
        'yield_2018'  : year_yields[2018],
        'dev_2015'    : year_devs[2015],
        'dev_2016'    : year_devs[2016],
        'dev_2017'    : year_devs[2017],
        'dev_2018'    : year_devs[2018],
        'yield_diff'  : round(max(valid_y) - min(valid_y), 1),
        'yield_mean'  : round(float(np.mean(valid_y)), 1),
        'yield_std'   : round(float(np.std(valid_y)), 1),
        'fids_2015'   : year_fids[2015],
        'fids_2016'   : year_fids[2016],
        'fids_2017'   : year_fids[2017],
        'fids_2018'   : year_fids[2018],
    })

cl_df = pd.DataFrame(rows).sort_values(['n_years', 'yield_std'], ascending=[False, True])
cl_df.to_csv(f'{OUT_DIR}/continuous_clusters.csv', index=False, encoding='utf-8-sig')
print(f'\n  全クラスタ一覧 CSV: {OUT_DIR}/continuous_clusters.csv')
print(f'  4年連続: {(cl_df["n_years"]==4).sum()} クラスタ')
print(f'  3年連続: {(cl_df["n_years"]==3).sum()} クラスタ')
print(f'  2年連続: {(cl_df["n_years"]==2).sum()} クラスタ')
print()

# ── 2. past_yield_features.csv : 回帰モデル用の過去収量特徴量 ────────────────
#
#  各 (field_id, year) について、
#  「その field_id が属するクラスタの、該当年より前の年の収量記録」
#  を集計して特徴量を作成する。
#
#  出力列:
#    past_yield_mean  : 過去年収量の平均
#    past_yield_max   : 過去年収量の最大
#    past_yield_min   : 過去年収量の最小
#    past_yield_std   : 過去年収量の標準偏差
#    past_yield_n     : 過去年サンプル数（何年分あるか）
#    past_dev_mean    : 過去年収量偏差の平均（年効果を除いた実力値）
#    cluster_id       : 所属クラスタID（なければ NaN）

# field_id → cluster_id のマッピングを先に作る
fid_to_cluster = {}   # field_id -> (cluster_id, members index list)
for cl_idx, members in enumerate(clusters):
    cid = f'cl{cl_idx+1:03d}'
    for m in members:
        fid = int(df.iloc[m]['field_id'])
        fid_to_cluster[fid] = cl_idx

feature_rows = []
for _, row in df.iterrows():
    fid  = int(row['field_id'])
    yr   = int(row['year'])

    if fid not in fid_to_cluster:
        # クラスタ未所属（過去データなし）
        feature_rows.append({
            'field_id'       : fid,
            'year'           : yr,
            'cluster_id'     : None,
            'past_yield_n'   : 0,
            'past_yield_mean': np.nan,
            'past_yield_max' : np.nan,
            'past_yield_min' : np.nan,
            'past_yield_std' : np.nan,
            'past_dev_mean'  : np.nan,
        })
        continue

    cl_idx   = fid_to_cluster[fid]
    cid      = f'cl{cl_idx+1:03d}'
    members  = clusters[cl_idx]

    # クラスタ内の「現在の年より前の年」のデータ
    past = df.iloc[members]
    past = past[past['year'] < yr]

    if len(past) == 0:
        feature_rows.append({
            'field_id'       : fid,
            'year'           : yr,
            'cluster_id'     : cid,
            'past_yield_n'   : 0,
            'past_yield_mean': np.nan,
            'past_yield_max' : np.nan,
            'past_yield_min' : np.nan,
            'past_yield_std' : np.nan,
            'past_dev_mean'  : np.nan,
        })
    else:
        feature_rows.append({
            'field_id'       : fid,
            'year'           : yr,
            'cluster_id'     : cid,
            'past_yield_n'   : len(past),
            'past_yield_mean': round(past['yield'].mean(), 1),
            'past_yield_max' : round(past['yield'].max(), 1),
            'past_yield_min' : round(past['yield'].min(), 1),
            'past_yield_std' : round(past['yield'].std() if len(past) > 1 else 0.0, 1),
            'past_dev_mean'  : round(past['yield_dev'].mean(), 1),
        })

feat_df = pd.DataFrame(feature_rows)
feat_df.to_csv(f'{OUT_DIR}/past_yield_features.csv', index=False, encoding='utf-8-sig')

# 過去データの有無を集計
has_past = feat_df[feat_df['past_yield_n'] > 0]
print(f'  過去収量特徴量 CSV: {OUT_DIR}/past_yield_features.csv')
print(f'  過去データあり: {len(has_past)} 件 / 全体 {len(feat_df)} 件 '
      f'({100*len(has_past)/len(feat_df):.1f}%)')
print()
print('  過去データあり件数（年別）:')
for yr in YEARS:
    n = len(has_past[has_past['year'] == yr])
    t = len(feat_df[feat_df['year'] == yr])
    print(f'    {yr}年: {n}/{t} 件 ({100*n/max(t,1):.1f}%)')

# ── 3. 可視化 ─────────────────────────────────────────────────────────────────
COLORS_YR = {2015: '#4C72B0', 2016: '#55A868', 2017: '#C44E52', 2018: '#8172B2'}

# 図1: 地図プロット（クラスタごとに連線）
fig, ax = plt.subplots(figsize=(12, 9), facecolor='#f0f0f0')
# 全圃場を薄くプロット
ax.scatter(df['lon'], df['lat'], s=8, c='#aaaaaa', alpha=0.4, zorder=1,
           label='全圃場（クラスタ未所属）')

cmap = plt.cm.tab20
for cl_idx, members in enumerate(clusters):
    color = cmap(cl_idx % 20)
    sub   = df.iloc[members]
    # 各年の点
    for _, r in sub.iterrows():
        ax.scatter(r['lon'], r['lat'], s=60, color=color,
                   edgecolors='white', linewidths=0.5, zorder=3, alpha=0.9)
    # クラスタ内の重心を繋ぐ線
    for yr in YEARS:
        s = sub[sub['year'] == yr]
        if len(s) == 0:
            continue
        # 同クラスタ内の全ペアを繋ぐ
    sorted_members = sub.sort_values('year')
    for _, r1 in sorted_members.iterrows():
        for _, r2 in sorted_members.iterrows():
            if r1['year'] < r2['year']:
                ax.plot([r1['lon'], r2['lon']], [r1['lat'], r2['lat']],
                        color=color, alpha=0.35, linewidth=1.0, zorder=2)

# 年ごとのマーカーを凡例に追加
for yr, col in COLORS_YR.items():
    ax.scatter([], [], s=50, c=col, label=f'{yr}年', alpha=0.9)

ax.set_xlabel('経度', fontsize=10)
ax.set_ylabel('緯度', fontsize=10)
ax.set_title(f'連続記録クラスタ地図\n'
             f'（距離 ≤ {DIST_THR}m / 収量偏差差 ≤ {DEV_THR}kg）\n'
             f'同色の点が同クラスタ、線で連結',
             fontsize=11, fontweight='bold')
ax.set_facecolor('#f8f8f8')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/continuous_clusters_map.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  地図図: {OUT_DIR}/continuous_clusters_map.png')

# 図2: 各クラスタの年別収量推移（スパゲッティプロット）
n_clusters = len(clusters)
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), facecolor='#f9f9f9')

# 左: 全クラスタの収量推移
ax_l = axes2[0]
for cl_idx, members in enumerate(clusters):
    sub    = df.iloc[members]
    yr_yld = sub.groupby('year')['yield'].mean().reset_index()
    color  = cmap(cl_idx % 20)
    ax_l.plot(yr_yld['year'], yr_yld['yield'], '-o', color=color,
              alpha=0.6, linewidth=1.5, markersize=5)
ax_l.set_title('全連続記録クラスタの年別収量推移\n（各線 = 1クラスタ）',
               fontsize=11, fontweight='bold')
ax_l.set_xlabel('年', fontsize=10)
ax_l.set_ylabel('収量 (kg/10a)', fontsize=10)
ax_l.set_xticks(YEARS)
ax_l.set_facecolor('#fdfdfd')
ax_l.grid(True, alpha=0.3)

# 右: 年別収量偏差の推移（気象年効果を除いた実力推移）
ax_r = axes2[1]
for cl_idx, members in enumerate(clusters):
    sub    = df.iloc[members]
    yr_dev = sub.groupby('year')['yield_dev'].mean().reset_index()
    color  = cmap(cl_idx % 20)
    ax_r.plot(yr_dev['year'], yr_dev['yield_dev'], '-o', color=color,
              alpha=0.6, linewidth=1.5, markersize=5)
ax_r.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
ax_r.set_title('全連続記録クラスタの年別収量偏差推移\n（気象年効果を除いた実力値）',
               fontsize=11, fontweight='bold')
ax_r.set_xlabel('年', fontsize=10)
ax_r.set_ylabel('収量偏差（平均からの差）', fontsize=10)
ax_r.set_xticks(YEARS)
ax_r.set_facecolor('#fdfdfd')
ax_r.grid(True, alpha=0.3)

fig2.suptitle(f'連続記録クラスタの収量推移分析\n'
              f'（距離 ≤ {DIST_THR}m / 偏差差 ≤ {DEV_THR}kg）',
              fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(f'{OUT_DIR}/continuous_clusters_yield.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'  収量推移図: {OUT_DIR}/continuous_clusters_yield.png')

# ── 4. 閾値別サマリ表 ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('閾値別サマリ（距離300m固定）')
print('='*60)
print(f'{"偏差閾値":>8s} | {"クラスタ数":>8s} | {"field_id数":>9s} | '
      f'{"過去データあり件数":>15s} | {"割合":>6s}')
print('-'*60)
for dev_thr in DEV_THRS:
    cls = union_find_clusters(D, years_a, devs, DIST_THR, dev_thr)
    fids_in_cluster = set()
    for m in cls:
        for idx in m:
            fids_in_cluster.add(int(df.iloc[idx]['field_id']))
    # 過去データあり件数（簡易計算）
    n_past = 0
    for m in cls:
        sub = df.iloc[m]
        for _, r in sub.iterrows():
            past = sub[sub['year'] < int(r['year'])]
            if len(past) > 0:
                n_past += 1
    n_total = len(df)
    print(f'{dev_thr:>6d} kg | {len(cls):>8d} | {len(fids_in_cluster):>9d} | '
          f'{n_past:>15d} | {100*n_past/n_total:>5.1f}%')

print('\n完了')
print(f'  continuous_clusters.csv  : クラスタごとの年別収量一覧')
print(f'  past_yield_features.csv  : 回帰モデル用 過去収量特徴量')
