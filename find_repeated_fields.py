"""
find_repeated_fields.py
lat/lon が近接（同一圃場と見なせる距離）で年をまたぐ field_id を検索し、
各クラスタの年別収量をまとめた CSV と収量変動の図を出力する。
"""
import sqlite3, os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 日本語フォント設定 ────────────────────────────────────────────────────────
_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'outputs/data_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ── データ読み込み ─────────────────────────────────────────────────────────────
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
df['yield']    = pd.to_numeric(df['yield'], errors='coerce')
df['lat']      = pd.to_numeric(df['lat'],   errors='coerce')
df['lon']      = pd.to_numeric(df['lon'],   errors='coerce')
df = df.dropna(subset=['lat', 'lon', 'yield']).reset_index(drop=True)

YEARS = [2015, 2016, 2017, 2018]

# ── Haversine 距離行列 (m) ───────────────────────────────────────────────────
def haversine_matrix(lats, lons):
    R = 6371000.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2)**2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

print(f"全サンプル数: {len(df)}  (lat/lon/yield あり)")
lats  = df['lat'].values
lons  = df['lon'].values
years = df['year'].values

print("距離行列を計算中...")
D = haversine_matrix(lats, lons)

THRESHOLDS = [30, 50, 100, 200, 500]
COLORS     = {30: '#e74c3c', 50: '#e67e22', 100: '#2ecc71', 200: '#3498db', 500: '#9b59b6'}

# ── Union-Find ────────────────────────────────────────────────────────────────
def make_clusters(D, years, thr):
    """距離 < thr かつ異なる年のペアを Union-Find でクラスタ化し、
    年をまたぐクラスタのみ返す。"""
    n = len(years)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    same_loc = (D < thr) & (years[:, None] != years[None, :])
    rows, cols = np.where(same_loc)
    for r, c in zip(rows, cols):
        union(r, c)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    multi_year = []
    for root, members in clusters.items():
        member_years = set(df.iloc[members]['year'].values)
        if len(member_years) > 1:
            multi_year.append(members)

    return multi_year

# ── メイン処理 ────────────────────────────────────────────────────────────────
all_summary_rows = []   # 全閾値を通じた1行サマリ（CSV出力用）

for thr in THRESHOLDS:
    clusters = make_clusters(D, years, thr)

    # ---- 1. 各クラスタの年別収量テーブルを作成 --------------------------------
    rows = []
    for cl_idx, members in enumerate(clusters):
        sub = df.iloc[members][['field_id', 'year', 'yield', 'lat', 'lon']]

        # 代表 lat/lon（全メンバーの平均）
        lat_mean = sub['lat'].mean()
        lon_mean = sub['lon'].mean()
        n_fields = len(members)

        # 年ごとの収量（複数 field_id がある年は平均）
        year_yields = {}
        year_fids   = {}
        for yr in YEARS:
            yr_sub = sub[sub['year'] == yr]
            if len(yr_sub) > 0:
                year_yields[yr] = round(yr_sub['yield'].mean(), 1)
                year_fids[yr]   = ','.join(str(f) for f in sorted(yr_sub['field_id'].tolist()))
            else:
                year_yields[yr] = None
                year_fids[yr]   = None

        # 収量変動指標
        valid_yields = [v for v in year_yields.values() if v is not None]
        yield_max  = max(valid_yields)
        yield_min  = min(valid_yields)
        yield_diff = round(yield_max - yield_min, 1)
        yield_std  = round(float(np.std(valid_yields)), 1)
        n_years    = len(set(sub['year'].values))

        row = {
            'threshold_m' : thr,
            'cluster_id'  : f'thr{thr}_cl{cl_idx+1:03d}',
            'n_fields'    : n_fields,
            'n_years'     : n_years,
            'lat_mean'    : round(lat_mean, 5),
            'lon_mean'    : round(lon_mean, 5),
            'yield_2015'  : year_yields[2015],
            'yield_2016'  : year_yields[2016],
            'yield_2017'  : year_yields[2017],
            'yield_2018'  : year_yields[2018],
            'yield_max'   : yield_max,
            'yield_min'   : yield_min,
            'yield_diff'  : yield_diff,   # Max - Min
            'yield_std'   : yield_std,
            'fids_2015'   : year_fids[2015],
            'fids_2016'   : year_fids[2016],
            'fids_2017'   : year_fids[2017],
            'fids_2018'   : year_fids[2018],
        }
        rows.append(row)

    cl_df = pd.DataFrame(rows).sort_values('yield_diff', ascending=False).reset_index(drop=True)

    # ---- 2. CSV 出力 -----------------------------------------------------------
    csv_path = os.path.join(OUT_DIR, f'cluster_yields_thr{thr}m.csv')
    cl_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # BOM付きUTF-8 (Excel対応)
    print(f'\n=== 閾値 {thr}m : {len(clusters)} クラスタ ===')
    print(f'  CSV出力: {csv_path}')

    # サマリ統計を表示
    print(f'  収量変動 (Max-Min) 統計:')
    print(f'    mean  = {cl_df["yield_diff"].mean():.1f}')
    print(f'    median= {cl_df["yield_diff"].median():.1f}')
    print(f'    max   = {cl_df["yield_diff"].max():.1f}')
    print(f'  収量差 > 100 のクラスタ数: {(cl_df["yield_diff"] > 100).sum()}')
    print(f'  収量差 > 150 のクラスタ数: {(cl_df["yield_diff"] > 150).sum()}')

    # 上位5件（変動が大きい順）を表示
    disp_cols = ['cluster_id', 'n_fields', 'n_years',
                 'yield_2015', 'yield_2016', 'yield_2017', 'yield_2018',
                 'yield_diff', 'yield_std']
    print(f'  変動が大きいクラスタ TOP5:')
    print(cl_df[disp_cols].head(5).to_string(index=False))

    all_summary_rows.extend(rows)

# ── 全閾値まとめ CSV ─────────────────────────────────────────────────────────
all_df = pd.DataFrame(all_summary_rows)
all_csv = os.path.join(OUT_DIR, 'cluster_yields_all_thresholds.csv')
all_df.to_csv(all_csv, index=False, encoding='utf-8-sig')
print(f'\n全閾値まとめ CSV: {all_csv}')

# ── 閾値別 収量変動分布の箱ひげ図 ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f9f9f9')

# 左: 閾値別 yield_diff の箱ひげ図
box_data   = []
box_labels = []
for thr in THRESHOLDS:
    sub = all_df[all_df['threshold_m'] == thr]['yield_diff']
    box_data.append(sub.values)
    box_labels.append(f'{thr}m\n(n={len(sub)})')

bp = axes[0].boxplot(box_data, labels=box_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, thr in zip(bp['boxes'], THRESHOLDS):
    patch.set_facecolor(COLORS[thr])
    patch.set_alpha(0.7)
axes[0].set_title('閾値別: クラスタ内収量変動 (Max - Min)\n値が小さいほど同一圃場として安定',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('距離閾値', fontsize=10)
axes[0].set_ylabel('収量差 Max-Min (kg/10a)', fontsize=10)
axes[0].axhline(100, color='red', linestyle='--', alpha=0.6, label='変動 100 ライン')
axes[0].axhline(150, color='orange', linestyle='--', alpha=0.6, label='変動 150 ライン')
axes[0].legend(fontsize=9)
axes[0].set_facecolor('#fdfdfd')
axes[0].grid(True, alpha=0.3, axis='y')

# 右: 閾値別「変動大クラスタ割合」の棒グラフ
pct_100 = []
pct_150 = []
for thr in THRESHOLDS:
    sub = all_df[all_df['threshold_m'] == thr]
    pct_100.append(100 * (sub['yield_diff'] > 100).sum() / max(len(sub), 1))
    pct_150.append(100 * (sub['yield_diff'] > 150).sum() / max(len(sub), 1))

x = range(len(THRESHOLDS))
w = 0.35
bars1 = axes[1].bar([i - w/2 for i in x], pct_100, w,
                    label='収量差 > 100', color='#e74c3c', alpha=0.75)
bars2 = axes[1].bar([i + w/2 for i in x], pct_150, w,
                    label='収量差 > 150', color='#e67e22', alpha=0.75)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([f'{t}m' for t in THRESHOLDS])
axes[1].set_title('閾値別: 収量変動が大きいクラスタの割合\n（割合が高い = 怪しいクラスタが多い）',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('距離閾値', fontsize=10)
axes[1].set_ylabel('クラスタに占める割合 (%)', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].set_facecolor('#fdfdfd')
axes[1].grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.5,
                         f'{h:.0f}%', ha='center', va='bottom', fontsize=9)

fig.suptitle('同一圃場クラスタ内の収量変動分析', fontsize=13, fontweight='bold')
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, 'cluster_yield_variation.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'箱ひげ図保存: {fig_path}')

# ── 全閾値 scatter: cluster内 yield_std vs n_years ───────────────────────────
fig2, axes2 = plt.subplots(1, len(THRESHOLDS), figsize=(18, 4),
                            sharey=True, facecolor='#f9f9f9')
for ax, thr in zip(axes2, THRESHOLDS):
    sub = all_df[all_df['threshold_m'] == thr]
    sc = ax.scatter(sub['n_years'], sub['yield_std'],
                    c=COLORS[thr], s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax.set_title(f'{thr}m\n({len(sub)} クラスタ)', fontsize=10, fontweight='bold')
    ax.set_xlabel('記録年数', fontsize=9)
    if thr == THRESHOLDS[0]:
        ax.set_ylabel('収量 標準偏差 (kg/10a)', fontsize=9)
    ax.set_xticks([2, 3, 4])
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fdfdfd')
    # 水平ライン (std > 50 で要注意)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, linewidth=1)

fig2.suptitle('クラスタの記録年数 vs 収量標準偏差\n（赤破線: std=50 を超えると変動大）',
              fontsize=12, fontweight='bold')
fig2.tight_layout()
fig2_path = os.path.join(OUT_DIR, 'cluster_yield_std_vs_nyears.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'散布図保存: {fig2_path}')

print('\n=== 完了 ===')
print(f'CSV (閾値別): {OUT_DIR}/cluster_yields_thr{{30,50,100,200,500}}m.csv')
print(f'CSV (全まとめ): {all_csv}')
