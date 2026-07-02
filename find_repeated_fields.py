"""
find_repeated_fields.py
lat/lon が近接（同一圃場と見なせる距離）で年をまたぐ field_id を検索する。
"""
import sqlite3, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定
_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'outputs/data_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

FIELD_DB = 'data/processed/FieldData_fieldid.db'
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
df = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)

# ── Haversine 距離 (m) ────────────────────────────────────────────────────────
def haversine_matrix(lats, lons):
    """全ペアのハバーサイン距離行列を返す（メートル）"""
    R = 6371000.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

print(f"全サンプル数: {len(df)}  (lat/lon あり)")
lats = df['lat'].values
lons = df['lon'].values
years = df['year'].values

print("距離行列を計算中...")
D = haversine_matrix(lats, lons)  # (N, N) メートル

THRESHOLDS = [30, 50, 100, 200, 500]   # メートル
COLORS     = {30: '#e74c3c', 50: '#e67e22', 100: '#2ecc71', 200: '#3498db', 500: '#9b59b6'}

results = {}
for thr in THRESHOLDS:
    # 異なる year で距離 < thr の全ペアを抽出
    same_loc = (D < thr) & (years[:, None] != years[None, :])  # (N, N) bool

    # Union-Find で同一地点クラスタを作成
    parent = list(range(len(df)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    rows, cols = np.where(same_loc)
    for r, c in zip(rows, cols):
        union(r, c)

    # クラスタ化
    from collections import defaultdict
    clusters = defaultdict(list)
    for i in range(len(df)):
        clusters[find(i)].append(i)

    # 年をまたぐクラスタのみ抽出
    multi_year_clusters = []
    for root, members in clusters.items():
        member_years = set(df.iloc[members]['year'].values)
        if len(member_years) > 1:
            multi_year_clusters.append({
                'members': members,
                'years': sorted(member_years),
                'n_years': len(member_years),
                'n_fields': len(members),
                'field_ids': sorted(df.iloc[members]['field_id'].values),
                'lat_mean': df.iloc[members]['lat'].mean(),
                'lon_mean': df.iloc[members]['lon'].mean(),
                'yield_mean': df.iloc[members]['yield'].mean(),
            })

    n_locs = len(multi_year_clusters)
    n_fields = sum(c['n_fields'] for c in multi_year_clusters)
    results[thr] = {'clusters': multi_year_clusters, 'n_locs': n_locs, 'n_fields': n_fields}

    print(f"\n=== 閾値 {thr:4d}m ===")
    print(f"  年をまたぐ地点数 (クラスタ数): {n_locs}")
    print(f"  該当 field_id 総数         : {n_fields}")
    if multi_year_clusters:
        df_cl = pd.DataFrame(multi_year_clusters)[['field_ids', 'years', 'n_fields', 'lat_mean', 'lon_mean']]
        df_cl['lat_mean'] = df_cl['lat_mean'].round(4)
        df_cl['lon_mean'] = df_cl['lon_mean'].round(4)
        df_cl = df_cl.sort_values('n_fields', ascending=False)
        print(df_cl.to_string(index=False))

# ── 閾値別サマリ図 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#f9f9f9')

thrs     = THRESHOLDS
n_locs_v = [results[t]['n_locs']   for t in thrs]
n_fids_v = [results[t]['n_fields'] for t in thrs]

axes[0].bar([str(t) for t in thrs], n_locs_v,
            color=[COLORS[t] for t in thrs], edgecolor='white')
axes[0].set_title('閾値別: 年をまたぐ同一地点数\n（同一圃場と見なせるクラスタ数）', fontsize=11, fontweight='bold')
axes[0].set_xlabel('距離閾値 (m)', fontsize=10)
axes[0].set_ylabel('クラスタ数（地点数）', fontsize=10)
for i, (t, v) in enumerate(zip(thrs, n_locs_v)):
    axes[0].text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_facecolor('#fdfdfd')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar([str(t) for t in thrs], n_fids_v,
            color=[COLORS[t] for t in thrs], edgecolor='white')
axes[1].set_title('閾値別: 年をまたぐ field_id 総数\n（連続記録と見なせる圃場数）', fontsize=11, fontweight='bold')
axes[1].set_xlabel('距離閾値 (m)', fontsize=10)
axes[1].set_ylabel('field_id 数', fontsize=10)
for i, (t, v) in enumerate(zip(thrs, n_fids_v)):
    axes[1].text(i, v + 0.3, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_facecolor('#fdfdfd')
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('同一圃場と見なせる閾値別 連続記録圃場数', fontsize=13, fontweight='bold')
fig.tight_layout()
path = os.path.join(OUT_DIR, 'repeated_fields_by_threshold.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n図保存: {path}')

# ── 100m 閾値の詳細: 年跨ぎクラスタを地図上にプロット ────────────────────────
THR_DETAIL = 100
clusters_detail = results[THR_DETAIL]['clusters']
if clusters_detail:
    fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor='#f9f9f9')
    YEAR_COLORS = {2015: '#4C72B0', 2016: '#55A868', 2017: '#C44E52', 2018: '#8172B2'}

    # 背景: 全圃場を薄くプロット
    ax2.scatter(df['lon'], df['lat'], c='#cccccc', s=15, alpha=0.4, zorder=1)

    # 年跨ぎクラスタをハイライト
    for cl in clusters_detail:
        members = cl['members']
        sub = df.iloc[members]
        for _, row in sub.iterrows():
            ax2.scatter(row['lon'], row['lat'],
                        c=YEAR_COLORS[row['year']], s=80, zorder=3,
                        edgecolors='black', linewidths=0.8)
        # クラスタ内を線で結ぶ
        coords = sub[['lon', 'lat']].values
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                ax2.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                         'k--', lw=0.8, alpha=0.5, zorder=2)

    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markeredgecolor='black', markersize=10, label=f'{yr}年')
        for yr, c in YEAR_COLORS.items()
    ] + [Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc',
                markersize=8, label='その他（単年）')]
    ax2.legend(handles=legend_elements, title='Year', fontsize=10, loc='lower right')
    ax2.set_title(f'年をまたぐ連続記録圃場（距離閾値 {THR_DETAIL}m）\n'
                  f'ハイライト: {len(clusters_detail)} クラスタ / {results[THR_DETAIL]["n_fields"]} 圃場',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=10)
    ax2.set_ylabel('Latitude',  fontsize=10)
    ax2.set_facecolor('#f0f0f0')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path2 = os.path.join(OUT_DIR, f'repeated_fields_map_{THR_DETAIL}m.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'地図図保存: {path2}')
