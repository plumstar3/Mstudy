"""各生成ファイルの詳細を確認するスクリプト"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\soybean_ts2vec')
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD', 'SWE', 'SFW']

# ════════════════════════════════════════════════════
print("=" * 60)
print("1. X.npy  (気象時系列テンソル)")
print("=" * 60)
X = np.load(OUT_DIR / 'X.npy')
print(f"  shape      : {X.shape}  → (サンプル数, 日数, 気象変数数)")
print(f"  dtype      : {X.dtype}")
print(f"  サイズ     : {X.nbytes / 1024**2:.2f} MB")
print(f"  NaN 件数   : {np.isnan(X).sum():,} / {X.size:,} ({np.isnan(X).mean()*100:.2f}%)")
print()
print("  【各気象変数の統計（NaN除外）】")
print(f"  {'変数':<12} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'NaN%':>7}")
print(f"  {'-'*57}")
for i, col in enumerate(WEATHER_COLS):
    vals = X[:, :, i].flatten()
    vals_valid = vals[~np.isnan(vals)]
    nan_pct = np.isnan(vals).mean() * 100
    print(f"  {col:<12} {vals_valid.mean():>9.3f} {vals_valid.std():>9.3f} "
          f"{vals_valid.min():>9.3f} {vals_valid.max():>9.3f} {nan_pct:>6.2f}%")

print()
print("  【先頭サンプル (sample[0]) の最初5日間】")
print(f"  {'日付インデックス':<6} " + " ".join(f"{c:>10}" for c in WEATHER_COLS))
for day in range(5):
    vals_str = " ".join(
        f"{X[0, day, i]:>10.3f}" if not np.isnan(X[0, day, i]) else f"{'NaN':>10}"
        for i in range(9)
    )
    print(f"  day[{day:03d}]    {vals_str}")

# ════════════════════════════════════════════════════
print()
print("=" * 60)
print("2. y.npy  (収量ラベル)")
print("=" * 60)
y = np.load(OUT_DIR / 'y.npy')
print(f"  shape      : {y.shape}  → (サンプル数,)")
print(f"  dtype      : {y.dtype}")
print(f"  サイズ     : {y.nbytes / 1024:.2f} KB")
print()
print("  【統計】")
print(f"  件数       : {len(y)}")
print(f"  平均       : {y.mean():.2f} kg/10a")
print(f"  標準偏差   : {y.std():.2f}")
print(f"  最小       : {y.min():.2f}")
print(f"  中央値     : {np.median(y):.2f}")
print(f"  最大       : {y.max():.2f}")
print()
print("  【収量分布（ヒストグラム的）】")
bins = [0, 100, 200, 300, 400, 500, 700]
labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500+']
for i, label in enumerate(labels):
    lo, hi = bins[i], bins[i+1]
    count = ((y >= lo) & (y < hi)).sum()
    bar = '#' * int(count / len(y) * 40)
    print(f"  {label:>10}: {bar:<40} {count:3d}件")

print()
print("  【年別件数】")
meta = pd.read_csv(OUT_DIR / 'meta.csv')
print(meta.groupby('year')['yield'].describe().round(2).to_string())

# ════════════════════════════════════════════════════
print()
print("=" * 60)
print("3. meta.csv  (サンプルメタ情報)")
print("=" * 60)
print(f"  shape      : {meta.shape}  → (サンプル数, 列数)")
print(f"  カラム     : {list(meta.columns)}")
print(f"  サイズ     : {(OUT_DIR / 'meta.csv').stat().st_size / 1024:.2f} KB")
print()
print("  【先頭10行】")
print(meta.head(10).to_string(index=True))
print()
print(f"  ユニーク field_id 数: {meta['field_id'].nunique()}")
print(f"  年別件数:")
print(meta['year'].value_counts().sort_index().to_string())

# ════════════════════════════════════════════════════
print()
print("=" * 60)
print("4. soybean_dataset.parquet  (中間データ)")
print("=" * 60)
df = pd.read_parquet(OUT_DIR / 'soybean_dataset.parquet')
print(f"  shape      : {df.shape}  → (行数, 列数)")
print(f"  カラム     : {list(df.columns)}")
size_mb = (OUT_DIR / 'soybean_dataset.parquet').stat().st_size / 1024**2
print(f"  サイズ     : {size_mb:.2f} MB")
print()
print("  【各カラムの型と欠損数】")
for col in df.columns:
    null_n = df[col].isna().sum()
    print(f"  {col:<20} {str(df[col].dtype):<12} NaN: {null_n:>6,}")
print()
print("  【先頭5行】")
print(df.head(5).to_string(index=False))
print()
print("  【date 範囲】")
print(f"  {df['date'].min()} 〜 {df['date'].max()}")
