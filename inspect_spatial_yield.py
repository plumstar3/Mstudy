"""
地理的近傍圃場の収量差を検証するスクリプト
仮説：気象変数がほぼ同じ（地理的に近い）にもかかわらず収量が大きく異なるサンプルペアが存在し、
      それがモデルの混乱を招いているか？
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ── データ取得 ──────────────────────────────────────────────────────────
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
df = pd.read_sql('''
    SELECT field_id, year, lat, lon, yield
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND lat IS NOT NULL AND lon IS NOT NULL
      AND year BETWEEN 2015 AND 2018
    ORDER BY year, field_id
''', conn)
conn.close()

df['field_id'] = df['field_id'].astype(int)
df['year']     = df['year'].astype(int)
print(f"対象サンプル数: {len(df)}  (field_id ユニーク: {df['field_id'].nunique()})")

# ── 年度ごとに「同年・近傍ペア」の収量差を分析 ────────────────────────
# 緯度経度の差 → ユークリッド距離（概算）
# 北海道付近: 1deg lat ≈ 111 km, 1deg lon ≈ 78 km
LAT_KM = 111.0
LON_KM = 78.0

all_pairs = []
DIST_THRESHOLDS_KM = [5, 10, 20, 50]  # 近傍の定義

for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr].reset_index(drop=True)
    n = len(sub)
    if n < 2:
        continue

    # 距離行列計算（km換算）
    lats = sub['lat'].values
    lons = sub['lon'].values
    coords_km = np.column_stack([lats * LAT_KM, lons * LON_KM])
    dist_mat = cdist(coords_km, coords_km)  # (n, n)

    y_arr = sub['yield'].values

    for i in range(n):
        for j in range(i + 1, n):
            d_km = dist_mat[i, j]
            dy   = abs(y_arr[i] - y_arr[j])
            all_pairs.append({
                'year': yr,
                'fid_a': sub['field_id'].iloc[i],
                'fid_b': sub['field_id'].iloc[j],
                'dist_km': d_km,
                'yield_a': y_arr[i],
                'yield_b': y_arr[j],
                'yield_diff': dy,
            })

pairs_df = pd.DataFrame(all_pairs)
print(f"\n全ペア数: {len(pairs_df):,}")

# ── 距離帯ごとの収量差統計 ──────────────────────────────────────────────
print("\n=== 距離帯別の収量差（|yield_A - yield_B|）===")
print(f"  {'距離帯':<15} {'ペア数':>7} {'mean':>7} {'std':>7} {'median':>7} {'max':>7}")
print("  " + "-" * 52)

bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 999)]
for lo, hi in bins:
    sub = pairs_df[(pairs_df['dist_km'] >= lo) & (pairs_df['dist_km'] < hi)]
    if len(sub) == 0:
        continue
    dy = sub['yield_diff']
    print(f"  {lo:3d}-{hi:3d} km       {len(sub):>7,} {dy.mean():>7.1f} {dy.std():>7.1f} "
          f"{dy.median():>7.1f} {dy.max():>7.1f}")

# ── 近傍（< 10km）かつ大きな収量差（> 100）のペア ──────────────────────
close_big = pairs_df[(pairs_df['dist_km'] < 10) & (pairs_df['yield_diff'] > 100)]
print(f"\n=== 10km以内 かつ 収量差 > 100 のペア: {len(close_big)} 件 ===")
print(close_big.sort_values('yield_diff', ascending=False).head(15).to_string(index=False))

# ── 「多収と低収が混在する近傍クラスタ」を定量化 ────────────────────────
# Train 平均（257.9）で High/Low を定義
THRESHOLD = 257.9
pairs_df['high_a'] = pairs_df['yield_a'] >= THRESHOLD
pairs_df['high_b'] = pairs_df['yield_b'] >= THRESHOLD
pairs_df['mixed']  = pairs_df['high_a'] != pairs_df['high_b']  # 異グループペア

print("\n=== 距離帯別「High/Low混在ペア率」===")
print(f"  {'距離帯':<15} {'ペア数':>7} {'混在率':>8}")
print("  " + "-" * 35)
for lo, hi in bins:
    sub = pairs_df[(pairs_df['dist_km'] >= lo) & (pairs_df['dist_km'] < hi)]
    if len(sub) == 0:
        continue
    mixed_rate = sub['mixed'].mean() * 100
    print(f"  {lo:3d}-{hi:3d} km       {len(sub):>7,} {mixed_rate:>7.1f}%")

# ── 収量の空間自己相関（簡易）──────────────────────────────────────────
# 距離と収量差の相関：正なら「遠いほど収量差が大きい＝空間的構造あり」
from scipy.stats import spearmanr
r, p = spearmanr(pairs_df['dist_km'], pairs_df['yield_diff'])
print(f"\n=== 距離 vs 収量差 Spearman 相関 ===")
print(f"  r = {r:.4f}  p = {p:.4e}")
if r > 0.1 and p < 0.05:
    print("  → 距離が大きいほど収量差が大きい傾向あり（弱い空間的構造）")
elif abs(r) < 0.05:
    print("  → 距離と収量差にほとんど相関なし（=地理的に近くても遠くても収量差は同程度）")
else:
    print(f"  → r={r:.3f}")

# ── まとめ ──────────────────────────────────────────────────────────────
print("\n=== 仮説検証サマリー ===")
near10 = pairs_df[pairs_df['dist_km'] < 10]
if len(near10) > 0:
    mixed_near = near10['mixed'].mean() * 100
    mean_dy_near = near10['yield_diff'].mean()
    print(f"  10km以内のペア: {len(near10)}件")
    print(f"  うち High/Low 混在: {mixed_near:.1f}%")
    print(f"  10km以内の平均収量差: {mean_dy_near:.1f} kg/10a")

far = pairs_df[pairs_df['dist_km'] >= 100]
if len(far) > 0:
    print(f"  100km以上のペア: {len(far)}件")
    print(f"  100km以上の平均収量差: {far['yield_diff'].mean():.1f} kg/10a")
